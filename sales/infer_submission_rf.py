# -*- coding: utf-8 -*-
"""
Sales Forecasting — Inferência com Random Forest (gera submission.csv)
Autor: Igor Costa

Fluxo:
1) Lê train e test
2) KMeans 1D para Store/Dept (fit no treino) -> Store_cluster / Dept_cluster
3) Features temporais: weekofyear + lags/rolling por Store×Dept (com shift(1))
4) Treina RandomForestRegressor no treino completo
5) Prevê Weekly_Sales no template e salva CSV de submissão
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# CONFIGURAÇÕES
# =========================
DATA_TRAIN = Path("../data/train_sales.csv")
DATA_TEST  = Path("../data/score_template.csv")
OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION = OUTPUT_DIR / "submission.csv"

TARGET        = "Weekly_Sales"
K_STORE       = 3
K_DEPT        = 6
RANDOM_STATE  = 42

# Hiperparâmetros do RF 
RF_PARAMS = dict(
    n_estimators=600,
    max_depth=12,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# =========================
# FUNÇÕES AUXILIARES
# =========================
def read_train_test(train_path: Path, test_path: Path):
    df_tr = pd.read_csv(train_path, parse_dates=["Date"], dayfirst=True)
    df_te = pd.read_csv(test_path,  parse_dates=["Date"], dayfirst=True)
    req_tr = {"Store", "Dept", "Date", "IsHoliday", TARGET}
    req_te = {"Store", "Dept", "Date", "IsHoliday"}
    if not req_tr.issubset(df_tr.columns):
        raise ValueError(f"Train faltando colunas: {req_tr - set(df_tr.columns)}")
    if not req_te.issubset(df_te.columns):
        raise ValueError(f"Test faltando colunas: {req_te - set(df_te.columns)}")
    return df_tr, df_te

def fit_kmeans_on_category_stat(df_train: pd.DataFrame, cat: str, y: str,
                                n_clusters=5, stat="median", random_state=RANDOM_STATE):
    if stat not in ("mean", "median"):
        raise ValueError("stat deve ser 'mean' ou 'median'")
    agg = df_train.groupby(cat)[y].median() if stat == "median" else df_train.groupby(cat)[y].mean()
    stats = agg.rename("stat_value").reset_index()
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    stats["cluster"] = km.fit_predict(stats[["stat_value"]])
    order = (
        stats.groupby("cluster")["stat_value"].mean()
        .sort_values().reset_index()
        .assign(cluster_ordinal=lambda d: range(len(d)))
    )
    stats = stats.merge(order, on="cluster", how="left")
    return dict(stats[[cat, "cluster_ordinal"]].values)

def apply_mapping(df: pd.DataFrame, cat: str, new_col: str, mapping: dict) -> pd.DataFrame:
    out = df.copy()
    out[new_col] = out[cat].map(mapping)
    return out

def add_basic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["weekofyear"] = out["Date"].dt.isocalendar().week.astype(int)
    return out

def add_time_features_per_group(df: pd.DataFrame, target_col: str = TARGET) -> pd.DataFrame:
    """Lags/rolling por Store×Dept usando APENAS passado (shift(1))."""
    df = df.sort_values(["Store", "Dept", "Date"]).copy()
    g = df.groupby(["Store", "Dept"])[target_col]
    # Lags
    df["lag_1"] = g.transform(lambda s: s.shift(1))
    df["lag_4"] = g.transform(lambda s: s.shift(4))
    # Médias móveis (sempre com shift para usar só o passado)
    df["ma_4"]  = g.transform(lambda s: s.shift(1).rolling(4,  min_periods=1).mean())
    df["ma_8"]  = g.transform(lambda s: s.shift(1).rolling(8,  min_periods=1).mean())
    df["ma_12"] = g.transform(lambda s: s.shift(1).rolling(12, min_periods=1).mean())
    return df

def sanitize_X_for_rf(X: pd.DataFrame) -> pd.DataFrame:
    """Converte bools para int, garante numérico e preenche NaNs (lags iniciais)."""
    Xc = X.copy()
    for c in Xc.select_dtypes(include=["bool"]).columns:
        Xc[c] = Xc[c].astype(int)
    for c in Xc.columns:
        Xc[c] = pd.to_numeric(Xc[c], errors="raise")
    fill_map = {c: Xc[c].mean() for c in Xc.columns}
    Xc = Xc.fillna(fill_map)
    return Xc

# =========================
# PIPELINE
# =========================
def main():
    # 1) Lê dados
    df_train, df_test = read_train_test(DATA_TRAIN, DATA_TEST)

    # 2) Clusters KMeans 1D (fit no treino)
    store_map = fit_kmeans_on_category_stat(df_train, "Store", TARGET, n_clusters=K_STORE, stat="median")
    dept_map  = fit_kmeans_on_category_stat(df_train, "Dept",  TARGET, n_clusters=K_DEPT,  stat="median")

    df_train_fe = apply_mapping(df_train, "Store", "Store_cluster", store_map)
    df_train_fe = apply_mapping(df_train_fe, "Dept",  "Dept_cluster",  dept_map)
    df_test_fe  = apply_mapping(df_test,  "Store", "Store_cluster", store_map)
    df_test_fe  = apply_mapping(df_test_fe,"Dept",  "Dept_cluster",  dept_map)

    # 3) Variável temporal simples
    df_train_fe = add_basic_time_features(df_train_fe)
    df_test_fe  = add_basic_time_features(df_test_fe)

    # 4) Lags/Rollings por Store×Dept — gerar em train+test juntos (sem vazamento via shift)
    df_all = pd.concat([df_train_fe, df_test_fe], ignore_index=True)
    df_all = add_time_features_per_group(df_all, target_col=TARGET)

    # re-separa: no test o TARGET não existe no CSV, logo fica NaN após concat
    mask_test = df_all[TARGET].isna()
    df_train_all = df_all.loc[~mask_test].copy()
    df_test_all  = df_all.loc[ mask_test].copy()

    # 5) Seleção de features
    features = [
        "Store_cluster", "Dept_cluster", "IsHoliday", "weekofyear",
        "lag_1", "ma_4", "ma_8", "ma_12"
        # opcional: "lag_4"
    ]
    X_train = sanitize_X_for_rf(df_train_all[features])
    y_train = df_train_all[TARGET].copy()

    X_test  = sanitize_X_for_rf(df_test_all[features])

    # 6) Treina Random Forest no treino completo
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)

    # (opcional) métrica in-sample só para sanity-check
    pred_tr = rf.predict(X_train)
    mae  = mean_absolute_error(y_train, pred_tr)
    rmse = np.sqrt(mean_squared_error(y_train, pred_tr))
    print(f"[INFO] RF — Treino: MAE={mae:,.0f} | RMSE={rmse:,.0f}")

    # 7) Previsão no TEST e grava submissão
    preds_test = rf.predict(X_test)
    submission = df_test.copy()
    submission[TARGET] = preds_test
    submission.to_csv(SUBMISSION, index=False, float_format="%.6f")
    print(f"[OK] Submissão RF gerada em: {SUBMISSION.resolve()}")
    print(submission.head())

if __name__ == "__main__":
    main()
