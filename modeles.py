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


def har_fit_predict(
    y_win: pd.Series,
    X_win: pd.DataFrame,
    horizon: int = 1) -> tuple[float, dict]:
    """
    HAR: y_{t+1} = a + B' X_t + e_{t+1}
    y_win : Series of target (same freq as X_win)
    X_win : DataFrame of regressors (e.g. [D, W, M] or [D, W])
    Returns 1-step ahead forecast and in-sample diagnostics.
    """

    # build (X_t, y_{t+1}) pairs
    X = X_win.iloc[:-1].to_numpy()
    Y = y_win.iloc[1:].to_numpy()

    n = len(Y)
    k = X.shape[1] + 1  # +1 for intercept

    if n <= k:
        raise ValueError("Window too small for HAR estimation.")

    Xmat = np.column_stack([np.ones(n), X])

    # OLS
    beta, *_ = np.linalg.lstsq(Xmat, Y, rcond=None)

    # In-sample fit + R2
    Yhat = Xmat @ beta
    resid = Y - Yhat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((Y - Y.mean())**2))
    r2_is = 1 - sse / sst if sst > 0 else np.nan

    # classic OLS t-stats (same logic as your AR(1))
    sigma2 = sse / (n - k)
    XtX_inv = np.linalg.pinv(Xmat.T @ Xmat)  # robust to collinearity
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    tstats = np.where(se > 0, beta / se, np.nan)

    # 1-step ahead forecast
    x_last = X_win.iloc[-1].to_numpy()
    yhat_next = float(beta[0] + beta[1:] @ x_last)

    return yhat_next


def ar1_fit_predict(y_win: pd.Series, horizon: int = 1) -> tuple[float, dict]:
    """
    AR(1): y_t = a + b y_{t-1} + e_t, estimated by OLS on the window.
    Returns forecast for next point (t+1) and diagnostics (R2_IS, beta, t_beta).
    """
    X = y_win.iloc[:-1].to_numpy()
    Y = y_win.iloc[1:].to_numpy()

    n = len(X)
    Xmat = np.column_stack([np.ones_like(X), X])  # const + lag

    beta, *_ = np.linalg.lstsq(Xmat, Y, rcond=None)
    a, b = beta

    # In-window fitted + R2 (classic SCE/SCT)
    Yhat = Xmat @ beta
    resid = Y - Yhat
    sse = float(np.sum(resid**2))                 # SCR
    sst = float(np.sum((Y - Y.mean())**2))        # SCT
    r2_is = 1 - sse/sst if sst > 0 else np.nan

    # t-stat for b (classic OLS)
    k = 2
    sigma2 = sse / (n - k)
    XtX_inv = np.linalg.inv(Xmat.T @ Xmat)
    se_b = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    t_beta = float(b / se_b) if se_b > 0 else np.nan

    # 1-step ahead forecast
    yhat_next = a + b * y_win.iloc[-1]

    return float(yhat_next)


def garch_fit_predict(
    r_win: pd.Series,
    horizon: int,
    p: int = 1,
    o: int =0,
    q: int = 1,
    mean: str = "Constant",
    dist: str = "normal",
    vol:str = "GARCH"
) -> float:
    """
    Fit GARCH(p,q) sur des rendements journaliers (en décimal) et prédit la
    realized variance (proxy) sur `horizon` jours:

        RV_hat_{t->t+h} = sum_{i=1..h} E_t[r_{t+i}^2] ~= sum_{i=1..h} sigma^2_{t+i|t}

    Paramètres
    ----------
    r_win : pd.Series
        Rendements en décimal (0.001 = 0.1%).
    horizon : int
        Horizon de prévision (nb de jours de trading), ex: 5.
    p, q : int
        Ordres GARCH.
    mean : {"constant","zero"}
    dist : {"normal","t","skewt"}

    Retour
    ------
    RV_hat : float
        Prévision de variance cumulée sur `horizon` jours (en unités décimales^2).
        Ex: 0.0004 ~ (2%)^2 sur l'horizon si tu prends sqrt ensuite.
    """
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")

    r_win = r_win.dropna().astype(float)
    if len(r_win) < max(50, 10 * (p + q + 1)):
        raise ValueError("Window too small for stable GARCH estimation.")

    # arch est plus stable en pourcents
    r_pct = 100.0 * r_win

    mean_spec = "Constant" if mean.lower() == "constant" else "Zero"
    dist_key = dist.lower()
    if dist_key not in {"normal", "t", "skewt"}:
        raise ValueError("dist must be one of {'normal','t','skewt'}.")
    dist_spec = {"normal": "normal", "t": "t", "skewt": "skewt"}[dist_key]

    am = arch_model(
        r_pct,
        mean=mean_spec,
        vol=vol,
        p=p,
        o = o,
        q=q,
        dist=dist_spec,
        rescale=False,  # déjà en %
    )
    res = am.fit(disp="off")

    if vol.upper() == "EGARCH" and horizon > 1:
        f = res.forecast(horizon=horizon, method="simulation", simulations=2000, reindex=False)
    else:
        f = res.forecast(horizon=horizon, reindex=False)
        
    vars_h = f.variance.iloc[-1].to_numpy()  # longueur = horizon
    var_H_pct2 = float(np.sum(vars_h))       # variance cumulée sur H jours en (%^2)

    # Retour en unités décimales^2 (car r_pct = 100 * r_dec)
    RV_hat = var_H_pct2 / (100.0 ** 2)

    return RV_hat

def msar1_fit_predict(
    y_win: pd.Series,
    horizon: int = 1,
    k_regimes: int = 2,
    switching_variance: bool = False,
    trend: str = "n",
    maxiter: int = 200,
    disp: bool = False,
    debug: bool = False,
):
    if horizon != 1:
        raise ValueError("Cette fonction est faite pour horizon=1 (t+1).")

    y = y_win.astype(float).to_numpy()
    if (len(y) < 120) or (not np.isfinite(y).all()):
        return np.nan

    # winsorize léger
    lo, hi = np.quantile(y, [0.001, 0.999])
    y = np.clip(y, lo, hi)

    # centrage (stabilité)
    y_mean = y.mean()
    y0 = y - y_mean

    mod = MarkovRegression(
        endog=y0,
        k_regimes=k_regimes,
        trend=trend,
        order=1,
        switching_variance=switching_variance,
    )

    try:
        res = mod.fit(disp=disp, maxiter=maxiter, em_iter=10, search_reps=20, search_iter=5)
        yhat_last = float(np.asarray(res.predict())[-1])  # stable
        return yhat_last + y_mean
    except Exception as e:
        if debug:
            print("MSAR t+1 failed:", repr(e))
        return np.nan

