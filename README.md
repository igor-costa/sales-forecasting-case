# 🛒 Sales Forecasting Case

## 🎯 Objetivo
Prever as vendas semanais (`Weekly_Sales`) por **loja** e **departamento** para o período de **Mai–Out/2012**, usando dados históricos de **Fev/2010–Abr/2012**.  

A métrica usada para avaliação foi o **SMAPE (Symmetric Mean Absolute Percentage Error)**.

---

## 📊 Metodologia

### 1. Preparação dos Dados
- **Split temporal (OOT)**: últimas 12 semanas reservadas para avaliação futura.
- **Redução de cardinalidade**: clusterização de `Store` e `Dept` via **KMeans 1D**, criando `Store_cluster` (3 grupos) e `Dept_cluster` (6 grupos).
- **Features temporais**:
  - `weekofyear` (sazonalidade semanal).
  - Lags: `lag_1`.
  - Médias móveis: `ma_4`, `ma_8`, `ma_12`.

### 2. Modelagem
Foram testados modelos de árvore:
- **Random Forest** (baseline final escolhido).
- **LightGBM** (comparação com boosting).

### 3. Validação
- **Validação temporal**: últimas 4 semanas do treino usadas como validação para early stopping e tuning.

---

## 🧠 Resultados

| Modelo                 | OOT SMAPE | OOT MAE | OOT RMSE |
|-------------------------|-----------|---------|----------|
| Random Forest + Lags    | ~17%      | ~2.5k   | ~5k      |
| LightGBM + Lags (tuned) | ~19.5%    | ~2.5k   | ~5k      |

📌 O **Random Forest** foi escolhido para submissão final por apresentar **SMAPE menor** e alta robustez.

---


