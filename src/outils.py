from __future__ import annotations
import pandas as pd 
import os 
import numpy as np
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score, mean_squared_error
from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import itertools

def run_strategy_backtest(
    predictions,
    factor_returns,
    risk_free_rate,
    model_name,
    target_vol=0.20,
    rebalance_freq="D"
):

    rb = rebalance_dates_mask(predictions, rebalance_freq=rebalance_freq)
    print(f"Avg Rebalance/Week: {rb.sum()/52:.2f}")

    # Calcul des Poids Cibles
    weights = mom_vol_target_weights(
        log_var_mom_fcst=predictions,
        rb=rb,
        target_vol_annual=target_vol,
        w_max=2.0,
        no_trade_band=0.05
    )

    daily_weights = expand_daily_weights(weights)

    common_idx = daily_weights.index.intersection(factor_returns.index)
    daily_weights = daily_weights.loc[common_idx]

    # Calcul du PnL
    rp = (
        daily_weights["w_mom"] * factor_returns.loc[common_idx]
        + daily_weights["w_def"] * risk_free_rate.loc[common_idx]
    ).dropna()

    metrics = performance_metrics(rp)
    print(metrics)

    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)

    cost_per_period = 0.001 * turnover

    annual_cost = cost_per_period.mean() * 52

    print(f"Annual Cost: {annual_cost:.5f}")
    print("\n")

    return metrics, annual_cost

def expand_daily_weights(df):
    """
    Prend un DataFrame indexé par date (dates irrégulières)
    et retourne un DataFrame journalier avec forward-fill.
    """
    df = df.sort_index()

    daily_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="D"
    )

    return (
        df
        .reindex(daily_index)
        .ffill()
    )

def rolling_window_forecast(
    y: pd.Series,
    fit_predict_fn,
    window: int,
    X: pd.DataFrame | None = None,
    *,
    horizon: int = 1,
    step: int = 1,
    **model_kwargs,
) -> pd.Series:
    """
    Rolling forecast engine (walk-forward).

    - horizon : horizon de prévision (nb de jours futurs)
    - step    : fréquence de recalcul (1=tous les jours, H=non-overlapping)

    La prévision produite à la date t est TOUJOURS indexée à t + horizon.
    """

    if horizon < 1 or step < 1:
        raise ValueError("horizon and step must be positive integers.")

    # Align / clean
    if X is None:
        y2 = y.dropna().copy()
        X2 = None
    else:
        df = pd.concat([y.rename("y"), X], axis=1).dropna()
        y2 = df["y"].copy()
        X2 = df.drop(columns=["y"]).copy()

    start_end = window - 1
    last_end = len(y2) - horizon - 1
    if last_end < start_end:
        raise ValueError("Not enough data for given window and horizon.")

    dates = []
    preds = []

    for end in range(start_end, last_end + 1, step):
        y_win = y2.iloc[end - window + 1 : end + 1]

        if X2 is None:
            out = fit_predict_fn(y_win, horizon=horizon, **model_kwargs)
        else:
            X_win = X2.iloc[end - window + 1 : end + 1]
            out = fit_predict_fn(y_win, X_win, horizon=horizon, **model_kwargs)

        # accepte yhat seul ou (yhat, diag)
        yhat = out[0] if isinstance(out, tuple) else out

        forecast_date = y2.index[end + horizon]  #  fin d'horizon

        dates.append(forecast_date)
        preds.append(float(yhat))

    fcast = pd.Series(preds, index=pd.Index(dates, name="date"), name="forecast")
    return fcast

def rebalance_dates_mask(
    signal: pd.Series,
    rebalance_freq: str = "M",  # "M", "MS", "W-FRI", "W-MON", "D"
) -> pd.Series:
    """
    Renvoie une série booléenne indexée comme `signal` :
    True = date de rebalancement, False sinon.

    - "D" : tous les jours
    - "M" : dernier jour de trading du mois
    - "MS": premier jour de trading du mois (approx via resample, voir note)
    - "W-FRI", "W-MON" : rebal hebdo sur le jour indiqué
    """
    signal = signal.dropna()
    idx = signal.index

    if rebalance_freq == "D":
        return pd.Series(True, index=idx, name="rebalance")

    dummy = pd.Series(1.0, index=idx)

    # Dates candidates générées par resample (puis intersect avec idx)
    rb_dates = dummy.resample(rebalance_freq).last().dropna().index
    rb_dates = rb_dates.intersection(idx)

    rb = pd.Series(False, index=idx, name="rebalance")
    rb.loc[rb_dates] = True
    return rb

def mom_vol_target_weights(
    log_var_mom_fcst: pd.Series,      # log(sigma^2) prévu
    rb: pd.Series,                    # série booléenne True = date de rebalancement
    target_vol_annual: float = 0.10,  # cible vol annualisée
    w_max: float = 2.0,               # cap levier
    no_trade_band: float | None = 0.10,
    trading_days: int = 52,
) -> pd.DataFrame:
    """
    Stratégie baseline :
    Portefeuille = w_mom * MOM + (1 - w_mom) * DEF

    Le poids w_mom est déterminé uniquement par la volatilité prévue du facteur MOM.
    Cette version prend directement en entrée le masque de rebalancement rb (True/False).

    Timing (à gérer hors fonction si besoin):
    - si log_var_mom_fcst[t] correspond à la prévision pour t+1 mais est datée à t,
      alors appliquer un shift(1) sur les poids au moment du backtest.
    """

    # Alignement et nettoyage
    df = pd.concat({"logvar": log_var_mom_fcst, "rb": rb}, axis=1).dropna()
    idx = df.index
    rb_aligned = df["rb"].astype(bool)

    # Vol prévue
    sigma_hat = np.sqrt(np.exp(df["logvar"].astype(float)))

    # Cible daily
    sigma_star_daily = target_vol_annual / np.sqrt(trading_days)

    # Poids MOM aux dates de rebalance
    w_mom_rb = pd.Series(np.nan, index=idx)
    w_mom_rb.loc[rb_aligned] = sigma_star_daily / sigma_hat.loc[rb_aligned]

    # Bornes
    w_mom_rb = w_mom_rb.clip(lower=0.0, upper=w_max)

    # No-trade band (appliqué uniquement aux dates rb=True)
    if no_trade_band is not None:
        last = np.nan
        for t in idx:
            if not rb_aligned.loc[t]:
                continue
            cur = w_mom_rb.loc[t]
            if np.isnan(cur):
                continue
            if np.isnan(last):
                last = cur
                w_mom_rb.loc[t] = last
            else:
                if abs(cur - last) < no_trade_band:
                    w_mom_rb.loc[t] = last
                else:
                    last = cur
                    w_mom_rb.loc[t] = last

    # Poids constants entre rebalances
    w_mom = w_mom_rb.ffill().fillna(0.0)
    w_def = 1.0 - w_mom

    return pd.DataFrame({"w_mom": w_mom, "w_def": w_def}, index=idx)

def performance_metrics(
    returns: pd.Series,
    rf: pd.Series | None = None,
    trading_days: int = 252,
) -> pd.Series:
    """
    Calcule des métriques de performance classiques à partir d'une série de rendements.

    Paramètres
    ----------
    returns : pd.Series
        Rendements du portefeuille (daily).
    rf : pd.Series | None, optional
        Rendements sans risque (daily). Si None, rf = 0.
    trading_days : int
        Nombre de jours de trading par an (252 par défaut).

    Retour
    ------
    pd.Series
        Série contenant les métriques de performance.
    """

    returns = returns.dropna()

    if rf is None:
        rf = pd.Series(0.0, index=returns.index)
    else:
        rf = rf.reindex(returns.index).fillna(0.0)

    excess_ret = returns - rf

    # --- Moyennes et volatilités ---
    mean_daily = returns.mean()
    vol_daily = returns.std()

    mean_annual = mean_daily * trading_days
    vol_annual = vol_daily * np.sqrt(trading_days)

    # --- Sharpe ratio ---
    sharpe = np.nan
    if vol_annual > 0:
        sharpe = (excess_ret.mean() * trading_days) / vol_annual

    # --- Drawdown ---
    cum_ret = (1 + returns).cumprod()
    running_max = cum_ret.cummax()
    drawdown = cum_ret / running_max - 1
    max_drawdown = drawdown.min()

    # --- Autres métriques utiles ---
    skew = returns.skew()
    kurt = returns.kurtosis()

    hit_ratio = (returns > 0).mean()

    metrics = pd.Series(
        {
            "Mean annual return": mean_annual,
            "Annual volatility": vol_annual,
            "Sharpe ratio": sharpe,
            "Max drawdown": max_drawdown,
            "Skewness": skew,
            "Kurtosis": kurt,
            "Hit ratio": hit_ratio,
        }
    )

    return metrics


def sklearn_walk_forward(
        base_model,
        param_grid,
        X,
        y,
        train_window_size,
        val_window_size,
        refit_step=10
):
    """
    Walk-Forward avec optimisation Rolling Window et extraction des coef/features importance.
    Retourne :
    - df_preds : Prédictions vs Réel
    - df_params : Historique des meilleurs hyperparamètres
    - df_importances : Historique des coefficients ou feature importances
    """

    # Génération des combinaisons
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    n_samples = len(X)
    total_window = train_window_size + val_window_size
    start_index = total_window

    predictions = []
    actuals = []
    dates = []

    # Stockage des historiques
    best_params_history = []
    history_importances = []
    history_dates = []

    current_best_model = None

    # Boucle temporelle
    for t in tqdm(range(start_index, n_samples)):

        current_date = X.index[t]

        # 1 : OPTIMISATION & REFIT
        if (t - start_index) % refit_step == 0:

            # Définition des bornes
            end_val_idx = t
            start_val_idx = t - val_window_size
            end_train_idx = start_val_idx
            start_train_idx = end_train_idx - train_window_size
            
            if start_train_idx < 0:
                continue 

            # Données Train et Val
            X_train = X.iloc[start_train_idx: end_train_idx]
            y_train = y.iloc[start_train_idx: end_train_idx]
            X_val = X.iloc[start_val_idx: end_val_idx]
            y_val = y.iloc[start_val_idx: end_val_idx]

            # Grid Search manuel
            best_score = float('inf')
            best_params = None

            for params in param_combinations:
                model = clone(base_model)
                model.set_params(**params)
                model.fit(X_train, y_train)

                pred_val = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, pred_val))

                if rmse < best_score:
                    best_score = rmse
                    best_params = params

            # REFIT FINAL sur (Train + Val)
            X_total = X.iloc[start_train_idx: end_val_idx]
            y_total = y.iloc[start_train_idx: end_val_idx]

            current_best_model = clone(base_model)
            current_best_model.set_params(**best_params)
            current_best_model.fit(X_total, y_total)

            # Sauvegarde
            best_params_history.append({**best_params, 'date': current_date})

            # sauvegarde des coefs
            imps = get_model_coefs(current_best_model, X.columns)
            history_importances.append(imps)
            history_dates.append(current_date)

        # 2 Prédiction
        X_test = X.iloc[t: t + 1]
        y_test = y.iloc[t: t + 1]

        if current_best_model is not None:
            pred = current_best_model.predict(X_test)[0]
        else:
            pred = np.nan

        predictions.append(pred)
        actuals.append(y_test.values[0])
        dates.append(current_date)

    df_preds = pd.DataFrame({"Actual": actuals, "Pred": predictions}, index=dates)
    df_params = pd.DataFrame(best_params_history).set_index('date').reindex(dates).ffill()

    df_importances = pd.DataFrame(history_importances, index=history_dates)
    df_importances = df_importances.reindex(dates).ffill()

    return df_preds, df_params, df_importances


def engineer_features(df_raw, price_cols=None):
    """
    Génère automatiquement les features techniques et macro.
    - price_cols : liste des colonnes à traiter comme des prix
    - Les autres colonnes sont traitées comme des niveaux (Taux, VIX...)
    """
    df = df_raw.copy()
    if price_cols is None: price_cols = []

    # Séparation automatique si non fournie
    rate_cols = [c for c in df.columns if c not in price_cols]

    X_out = pd.DataFrame(index=df.index)

    # A. Traitement des PRIX (Rendements, Tendance)
    for col in price_cols:
        if col in df.columns:
            X_out[f'{col}_ret1d'] = df[col].pct_change(fill_method=None)
            X_out[f'{col}_vol21'] = df[col].pct_change(fill_method=None).rolling(21).std()
            X_out[f'{col}_vol5'] = df[col].pct_change(fill_method=None).rolling(5).std()
            X_out[f'{col}_vol1'] = df[col].pct_change(fill_method=None).rolling(2).std()

    # B. Traitement des TAUX (Variations, Cycle)
    for col in rate_cols:
        if col in df.columns:
            X_out[f'{col}_chg1d'] = df[col].diff(1)
            X_out[f'{col}_vol21'] = df[col].pct_change(fill_method=None).rolling(21).std()
            X_out[f'{col}_vol5'] = df[col].pct_change(fill_method=None).rolling(5).std()
            X_out[f'{col}_vol1'] = df[col].pct_change(fill_method=None).rolling(2).std()
            X_out[f'{col}_level'] = df[col]

    return X_out

def add_targeted_lags(df, suffixes, extra_cols=None, lags=[1, 2, 3, 5]):
    """
    Ajoute des lags (t-1, t-2...) sur les colonnes qui :
    1. Finissent par un des 'suffixes' donnés (ex: _ret1d)
    2. Sont listées dans 'extra_cols' (ex: RV 1j)
    """
    df_out = df.copy()
    if extra_cols is None: extra_cols = []

    # Identifier les colonnes à lagger
    cols_to_lag = []
    for col in df.columns:
        # Condition 1 : Suffixe
        if any(col.endswith(s) for s in suffixes):
            cols_to_lag.append(col)
        # Condition 2 : Colonne explicite (si présente)
        elif col in extra_cols:
            cols_to_lag.append(col)

    # Création des lags
    print(f"Lagging de {len(cols_to_lag)} variables sur {df.shape[1]}")

    for col in cols_to_lag:
        for lag in lags:
            new_col_name = f"{col}_lag{lag}"
            df_out[new_col_name] = df[col].shift(lag)

    return df_out.dropna()


def get_model_coefs(model, feature_names):

    """
    Récupère l'importance des variables (coeffs ou feature_importances)
    de manière agnostique (Pipeline ou modèle, Linéaire ou Arbre).
    """
    # Si on a un Pipeline (ex: Scaler + ElasticNet), il faut aller chercher le modèle à la fin
    if hasattr(model, 'steps'):
        estimator = model.steps[-1][1]
    else:
        estimator = model

    # Modèles paramétriques (Regressions, ElasticNet, Lasso...)
    if hasattr(estimator, 'coef_'):
        return pd.Series(estimator.coef_, index=feature_names)

    # Modèles à base d'arbres (RandomForest, XGBoost...)
    elif hasattr(estimator, 'feature_importances_'):
        return pd.Series(estimator.feature_importances_, index=feature_names)

    # Fallback si le modèle n'a rien de tout ça
    else:
        return pd.Series(np.nan, index=feature_names)
    

def expanding_window_forecast(
    y: pd.Series,
    fit_predict_fn,
    min_window: int,
    X: pd.DataFrame | None = None,
    *,
    horizon: int = 1,
    step: int = 1,
    **model_kwargs,
) -> pd.Series:
    """
    Expanding forecast engine (walk-forward).

    - horizon : horizon de prévision (nb de jours futurs)
    - step    : fréquence de recalcul (1=tous les jours, H=non-overlapping)

    La prévision produite à la date t est TOUJOURS indexée à t + horizon.
    """

    if horizon < 1 or step < 1:
        raise ValueError("horizon and step must be positive integers.")

    # Align / clean
    if X is None:
        y2 = y.dropna().copy()
        X2 = None
    else:
        df = pd.concat([y.rename("y"), X], axis=1).dropna()
        y2 = df["y"].copy()
        X2 = df.drop(columns=["y"]).copy()

    start_end = min_window - 1
    last_end = len(y2) - horizon - 1
    if last_end < start_end:
        raise ValueError("Not enough data for given min_window and horizon.")

    dates = []
    preds = []

    for end in range(start_end, last_end + 1, step):
        y_win = y2.iloc[: end + 1]  # expanding window

        if X2 is None:
            out = fit_predict_fn(y_win, horizon=horizon, **model_kwargs)
        else:
            X_win = X2.iloc[: end + 1]
            out = fit_predict_fn(y_win, X_win, horizon=horizon, **model_kwargs)

        yhat = out[0] if isinstance(out, tuple) else out

        forecast_date = y2.index[end + horizon]  # TOUJOURS fin d'horizon

        dates.append(forecast_date)
        preds.append(float(yhat))

    fcast = pd.Series(preds, index=pd.Index(dates, name="date"), name="forecast")
    return fcast
