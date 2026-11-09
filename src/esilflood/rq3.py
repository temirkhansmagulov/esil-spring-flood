"""RQ3: drivers of observed spring discharge (Table 6) with XGBoost + SHAP."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import (all_metrics, block_bootstrap_ci, peak_errors, rmse, spearman,
                      wilcoxon_paired)


def build_features(forcing: pd.DataFrame, cfg) -> pd.DataFrame:
    """Lagged ERA5-Land features (lags computed on the full series, then subset)."""
    rq = cfg["rq3"]
    cols = {}
    for var in rq["features"]:
        for lag in rq["lags"]:
            cols[f"{var}_lag{lag}"] = forcing[var].shift(lag)
    X = pd.DataFrame(cols, index=forcing.index)
    if rq.get("include_doy", True):
        X["doy"] = forcing.index.dayofyear
    return X


def base_variable(feature: str) -> str:
    return feature.rsplit("_lag", 1)[0]


def prepare_dataset(forcing: pd.DataFrame, discharge: pd.DataFrame, cfg) -> pd.DataFrame:
    X = build_features(forcing, cfg)
    df = X.join(discharge[["q_mm", "q_m3s", "hydro_year", "split", "spring"]], how="inner")
    return df[df["spring"]].dropna(subset=list(X.columns) + ["q_mm"])


def climatology_baseline(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Day-of-year mean of the training years."""
    doy_mean = train.groupby(train.index.dayofyear)["q_mm"].mean()
    doy_mean = doy_mean.reindex(range(1, 367)).interpolate(limit_direction="both")
    return doy_mean.loc[target.index.dayofyear].values


def fit_xgboost(train, val, features, cfg, seed):
    import xgboost as xgb

    p = dict(cfg["rq3"]["xgboost"])
    es = p.pop("early_stopping_rounds", None)
    model = xgb.XGBRegressor(**p, random_state=seed, early_stopping_rounds=es,
                             objective="reg:squarederror", tree_method="hist")
    model.fit(train[features], train["q_mm"], eval_set=[(val[features], val["q_mm"])],
              verbose=False)
    return model


def evaluate(pred: np.ndarray, test: pd.DataFrame, cfg) -> dict:
    obs, sim = test["q_mm"], pd.Series(pred, index=test.index)
    m = all_metrics(obs, sim, test["hydro_year"])
    df = pd.DataFrame({"obs": obs, "sim": sim, "year": test["hydro_year"]})
    n = cfg["project"]["bootstrap_iterations"]
    from .metrics import kge, nse, pbias
    for name, fn in (("NSE", nse), ("KGE", kge), ("PBIAS", pbias)):
        m[f"{name}_ci"] = block_bootstrap_ci(df, lambda d, f=fn: f(d["obs"], d["sim"]), n=n)
    pe = peak_errors(obs, sim, test["hydro_year"])
    for name in ("Ep", "dtp"):
        vals = pe[name].values
        from .metrics import bootstrap_ci_values
        m[f"{name}_ci"] = bootstrap_ci_values(vals, stat=np.mean, n=n)
    return m


def per_year_rmse(pred, test) -> pd.Series:
    d = pd.DataFrame({"o": test["q_mm"], "s": pred, "y": test["hydro_year"]})
    return d.groupby("y").apply(lambda g: rmse(g["o"], g["s"]), include_groups=False)


def run_rq3(forcing: pd.DataFrame, discharge: pd.DataFrame, cfg,
            swat_sim_mm: pd.Series | None = None) -> dict:
    data = prepare_dataset(forcing, discharge, cfg)
    features = [c for c in data.columns if c not in
                ("q_mm", "q_m3s", "hydro_year", "split", "spring")]
    train, val, test = (data[data["split"] == s] for s in ("train", "validation", "test"))
    if len(test) == 0 or len(train) == 0:
        raise ValueError("Empty train or test subset; check period.split and discharge data.")

    preds, results = {}, {}
    preds["Climatology (day-of-year mean)"] = climatology_baseline(train, test)
    lin = make_pipeline(StandardScaler(), LinearRegression()).fit(train[features], train["q_mm"])
    preds["Linear regression"] = np.clip(lin.predict(test[features]), 0, None)
    if swat_sim_mm is not None:
        preds["SWAT+ (calibrated)"] = swat_sim_mm.reindex(test.index).values

    seed_preds, seed_metrics, shap_rows = [], [], []
    for seed in cfg["project"]["random_seeds"]:
        model = fit_xgboost(train, val, features, cfg, seed)
        p = np.clip(model.predict(test[features]), 0, None)
        seed_preds.append(p)
        seed_metrics.append(all_metrics(test["q_mm"], pd.Series(p, index=test.index),
                                        test["hydro_year"]))
        shap_rows.append(shap_importance(model, test[features]))
    xgb_mean = np.mean(seed_preds, axis=0)
    preds["XGBoost (proposed)"] = xgb_mean

    for name, p in preds.items():
        results[name] = evaluate(np.asarray(p, float), test, cfg)
    sm = pd.DataFrame(seed_metrics)
    results["XGBoost (proposed)"]["seed_mean"] = sm.mean().to_dict()
    results["XGBoost (proposed)"]["seed_sd"] = sm.std(ddof=1).to_dict()

    # Paired Wilcoxon on per-year RMSE: XGBoost vs each baseline.
    xgb_err = per_year_rmse(xgb_mean, test)
    wil = {name: wilcoxon_paired(xgb_err.values, per_year_rmse(np.asarray(p, float), test)
                                 .loc[xgb_err.index].values)
           for name, p in preds.items() if name != "XGBoost (proposed)"}

    shap_df = pd.DataFrame(shap_rows)
    shap_rank = shap_df.mean().sort_values(ascending=False)
    pred_frame = pd.DataFrame(preds, index=test.index)
    pred_frame.insert(0, "observed", test["q_mm"])
    return {"metrics": results, "wilcoxon": wil, "shap": shap_rank,
            "shap_by_seed": shap_df, "predictions": pred_frame,
            "n_train": len(train), "n_val": len(val), "n_test": len(test)}


def shap_importance(model, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per base variable (summed over its lags)."""
    import shap

    vals = shap.TreeExplainer(model).shap_values(X)
    imp = pd.Series(np.abs(vals).mean(axis=0), index=X.columns)
    return imp.groupby([base_variable(c) for c in imp.index]).sum()


def agreement_with_scenarios(shap_rank: pd.Series, cfg) -> dict:
    """Spearman rho between SHAP ranking and SWAT+ perturbation sensitivity for the
    drivers common to both (needs swat.perturbation_csv: variable,delta_peak_pct)."""
    path = cfg.path(cfg["swat"].get("perturbation_csv"))
    common = cfg["rq3"]["common_drivers"]
    if path is None or not path.exists():
        return {"rho": np.nan, "p": np.nan, "note": "SWAT+ perturbation results not provided"}
    pert = pd.read_csv(path).set_index("variable")["delta_peak_pct"].abs()
    vars_ = [v for v in common if v in pert.index and v in shap_rank.index]
    rho, p = spearman(shap_rank.loc[vars_].values, pert.loc[vars_].values)
    return {"rho": rho, "p": p, "variables": vars_, "note": f"n = {len(vars_)} drivers"}
