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



def forecast_quality_oos(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """
    Standard OOS forecast metrics (finance).
    """
    df = pd.concat(
        [y_true.rename("y"), y_pred.rename("yhat")],
        axis=1
    ).dropna()

    err = df["y"] - df["yhat"]

    mspe = np.mean(err**2)
    rmse = np.sqrt(mspe)
    mae  = np.mean(np.abs(err))

    # Historical-mean benchmark (Welch–Goyal)
    y_bar = df["y"].mean()
    mspe_hm = np.mean((df["y"] - y_bar)**2)
    r2_oos = 1 - mspe / mspe_hm

    return pd.Series({
        "N_OOS": len(df),
        "MSPE": mspe,
        "RMSE": rmse,
        "MAE": mae,
        "R2_OOS": r2_oos
    })


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

