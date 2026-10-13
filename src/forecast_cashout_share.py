"""
Forecasting VA Cash-Out Refinance Share Across States
DSCI 6800 Final Project | D'Andre Burdin | Hawaii Pacific University

Question: can supervised ML forecast next year's state-level share of VA loans
that are cash-out refinances, and does it beat a simple persistence baseline?

Design (matches the written report, Sections III, VI, VII):
  - Panel: 54 states/territories x 2020-2024 (VA Loan Volume by State reports)
  - Target: Pct_Cashout for state s in year t
  - Predictors: lag-1 (t-1) versions of internal loan-mix variables plus a
    state one-hot encoding. Nothing from year t is used, so there is no leakage.
  - Baseline: persistence, yhat_t = y_(t-1)
  - Models: Ridge, decision tree, random forest, gradient boosting
  - Tuning: time-aware, on training years only (fit 2021-2022, validate 2023)
  - Final test: train 2021-2023, evaluate once on the 2024 holdout
  - Extension: rolling-origin check (test 2023, then 2024) and a residual
    model that predicts the year-over-year change on top of persistence

Usage: python src/forecast_cashout_share.py
"""

import os, warnings, itertools
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RS = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "va_state_year.xlsx")
OUT = os.path.join(ROOT, "outputs"); FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# Chart styling: thin marks, recessive grid, text in ink colors (not series colors)
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10,
                     "axes.axisbelow": True})

# ---------------- 1. LOAD ----------------
df = pd.read_excel(DATA).sort_values(["State", "Year"]).reset_index(drop=True)
print(f"[1] Loaded {len(df)} state-years: {df.State.nunique()} states x "
      f"{df.Year.min()}-{df.Year.max()}")
trend = df.groupby("Year").Pct_Cashout.mean()
print("    Mean cash-out share by year:", trend.round(4).to_dict())

# ---------------- 2. LAG FEATURES ----------------
LAG_SRC = ["Pct_Cashout", "Pct_IRRRL", "Pct_Purchase", "Total_Count", "Total_Amount",
           "Avg Loan Amount Cash-Out", "Avg Loan Amount IRRRL", "Avg Loan Amount Purchase"]
g = df.groupby("State")
for c in LAG_SRC:
    df[f"lag1_{c}"] = g[c].shift(1)
df["lag1_log_Total_Count"] = np.log1p(df["lag1_Total_Count"])
df = df[df.Year >= 2021].copy()              # 2020 has no prior year
NUM = ["lag1_Pct_Cashout", "lag1_Pct_IRRRL", "lag1_Pct_Purchase", "lag1_log_Total_Count",
       "lag1_Avg Loan Amount Cash-Out", "lag1_Avg Loan Amount IRRRL",
       "lag1_Avg Loan Amount Purchase"]
CAT = ["State"]
TARGET = "Pct_Cashout"
print(f"[2] Modeling rows (2021-2024): {len(df)} | {len(NUM)} lag features + state one-hot")

def prep():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])

GRIDS = {
    "Ridge regression": (lambda p: Ridge(alpha=p["alpha"]),
                         {"alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}),
    "Decision tree": (lambda p: DecisionTreeRegressor(max_depth=p["max_depth"], random_state=RS),
                      {"max_depth": [2, 3, 4, 6, 8, 10]}),
    "Random forest": (lambda p: RandomForestRegressor(n_estimators=p["n_estimators"],
                      max_depth=p["max_depth"], random_state=RS, n_jobs=-1),
                      {"n_estimators": [200, 500, 1000], "max_depth": [2, 4, 6, 10]}),
    "Gradient boosting": (lambda p: GradientBoostingRegressor(learning_rate=p["learning_rate"],
                          n_estimators=p["n_estimators"], max_depth=p["max_depth"], random_state=RS),
                          {"learning_rate": [0.01, 0.05, 0.1, 0.2],
                           "n_estimators": [200, 500, 1000], "max_depth": [2, 3, 5]}),
}

def metrics(y, yhat):
    return {"MAE": mean_absolute_error(y, yhat),
            "RMSE": float(np.sqrt(mean_squared_error(y, yhat))), "R2": r2_score(y, yhat)}

def tune_and_test(train_years, test_year, target_kind="level"):
    """Tune on the last training year, refit on all training years, test once."""
    tr, te = df[df.Year.isin(train_years)], df[df.Year == test_year]
    fit_y, val_y = train_years[:-1], train_years[-1]
    fit, val = df[df.Year.isin(fit_y)], df[df.Year == val_y]
    def y_of(d):   # level target, or change relative to persistence
        return d[TARGET] - d["lag1_Pct_Cashout"] if target_kind == "delta" else d[TARGET]
    def back(d, pred):
        return pred + d["lag1_Pct_Cashout"].values if target_kind == "delta" else pred
    rows, fitted = [], {}
    for name, (make, grid) in GRIDS.items():
        best, best_mae = None, np.inf
        for combo in itertools.product(*grid.values()):
            p = dict(zip(grid.keys(), combo))
            m = Pipeline([("p", prep()), ("m", make(p))]).fit(fit[NUM + CAT], y_of(fit))
            mae = mean_absolute_error(val[TARGET], back(val, m.predict(val[NUM + CAT])))
            if mae < best_mae: best, best_mae = p, mae
        m = Pipeline([("p", prep()), ("m", make(best))]).fit(tr[NUM + CAT], y_of(tr))
        fitted[name] = m
        rows.append({"model": name, **metrics(te[TARGET], back(te, m.predict(te[NUM + CAT]))),
                     "best_params": best, "val_MAE": best_mae})
    rows.insert(0, {"model": "Persistence baseline (y_t-1)",
                    **metrics(te[TARGET], te["lag1_Pct_Cashout"]),
                    "best_params": "n/a", "val_MAE": np.nan})
    return pd.DataFrame(rows), fitted, tr, te

# ---------------- 3. MAIN EXPERIMENT: train 2021-2023, test 2024 ----------------
print("[3] Main experiment: tune on 2021-2022 -> 2023, refit 2021-2023, test 2024")
main, fitted, tr, te = tune_and_test([2021, 2022, 2023], 2024, "level")
main.to_csv(os.path.join(OUT, "results_2024_holdout.csv"), index=False)
print(main[["model", "MAE", "RMSE", "R2", "best_params"]].round(4).to_string(index=False))

# ---------------- 4. ROLLING ORIGIN + RESIDUAL (DELTA) MODELS ----------------
print("[4] Rolling-origin check and residual-on-persistence models")
roll = []
for kind in ["level", "delta"]:
    for train_years, test_year in [([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]:
        if len(train_years) < 2: continue
        r, *_ = tune_and_test(train_years, test_year, kind)
        r["target"] = "level" if kind == "level" else "change vs. persistence"
        r["test_year"] = test_year
        roll.append(r)
roll = pd.concat(roll)
roll = roll[~((roll.target != "level") & roll.model.str.startswith("Persistence"))]
roll.to_csv(os.path.join(OUT, "results_rolling_origin.csv"), index=False)
print(roll[["test_year", "target", "model", "MAE", "R2"]].round(4).to_string(index=False))

# ---------------- 5. EXPLAINABILITY ----------------
print("[5] Feature importance")
rf = fitted["Random forest"]
names = rf.named_steps["p"].get_feature_names_out()
imp = pd.DataFrame({"feature": names,
                    "importance": rf.named_steps["m"].feature_importances_})
imp["group"] = np.where(imp.feature.str.startswith("cat__"), "State identity", imp.feature)
grp = imp.groupby("group").importance.sum().sort_values(ascending=False)
grp.to_csv(os.path.join(OUT, "feature_importance_random_forest.csv"))
print(grp.round(3).to_string())
try:
    import shap
    X = rf.named_steps["p"].transform(te[NUM + CAT])
    X = X.toarray() if hasattr(X, "toarray") else X
    sv = shap.TreeExplainer(rf.named_steps["m"]).shap_values(X)
    s = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)})
    s["group"] = np.where(s.feature.str.startswith("cat__"), "State identity", s.feature)
    s.groupby("group").mean_abs_shap.sum().sort_values(ascending=False).to_csv(
        os.path.join(OUT, "shap_importance_random_forest.csv"))
except Exception as e:
    print("    SHAP skipped:", e)

# ---------------- 6. FIGURES ----------------
clean = lambda f: (f.replace("num__lag1_", "Prior-year ").replace("Pct_Cashout", "share Cash-Out").replace("Pct_", "share ")
                   .replace("log_Total_Count", "volume (log)").replace("Avg Loan Amount", "avg amount"))

# 6a. Cash-out share trend
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(trend.index, trend.values * 100, color=BLUE, lw=2, marker="o", ms=8)
for x, y in trend.items():
    ax.annotate(f"{y*100:.1f}%", (x, y * 100), textcoords="offset points", xytext=(0, 9),
                ha="center", color=INK, fontsize=9)
ax.set_xticks(trend.index); ax.set_ylabel("Mean cash-out share across states (%)")
ax.set_ylim(0, trend.max() * 100 * 1.25)
ax.set_title("VA cash-out share more than doubled in 2022, then retreated", loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "cashout_share_trend.png"), dpi=150); plt.close()

# 6b. Model comparison (MAE, lower is better)
m = main.sort_values("MAE")
fig, ax = plt.subplots(figsize=(7, 3.8))
colors = [ORANGE if s.startswith("Persistence") else BLUE for s in m.model]
ax.barh(m.model, m.MAE * 100, color=colors, height=0.55)
for i, v in enumerate(m.MAE * 100):
    ax.text(v + 0.15, i, f"{v:.2f} pts", va="center", color=INK, fontsize=9)
ax.invert_yaxis(); ax.set_xlabel("Mean absolute error on 2024 holdout (percentage points)")
ax.set_title("No model beat last year's value on the 2024 holdout", loc="left")
ax.grid(axis="y", visible=False); ax.set_xlim(0, m.MAE.max() * 100 * 1.2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "model_comparison_2024.png"), dpi=150); plt.close()

# 6c. Persistence: last year vs this year
fig, ax = plt.subplots(figsize=(5.5, 5.5))
for yr, c in [(2023, ORANGE), (2024, BLUE)]:
    d = df[df.Year == yr]
    ax.scatter(d.lag1_Pct_Cashout * 100, d.Pct_Cashout * 100, s=40, color=c, alpha=0.85,
               edgecolor="#fcfcfb", linewidth=1, label=f"{yr} vs {yr-1}")
lim = [0, max(df.Pct_Cashout.max(), df.lag1_Pct_Cashout.max()) * 105]
ax.plot(lim, lim, color=INK2, lw=1, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Prior-year cash-out share (%)"); ax.set_ylabel("Current-year cash-out share (%)")
ax.set_title("Persistence holds in calm years, breaks after a shock", loc="left")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "persistence_scatter.png"), dpi=150); plt.close()

# 6d. Feature importance
top = grp.head(8)[::-1]
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh([clean(i) for i in top.index], top.values, color=BLUE, height=0.55)
ax.set_xlabel("Random forest importance (state dummies summed)")
ax.set_title("Prior-year loan mix drives the forecast", loc="left")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "feature_importance.png"), dpi=150); plt.close()

print("[6] Done. Outputs written to outputs/")
