"""
Random Forest болон XGBoost-оор PM2.5-ийг урьдчилан таамаглаж, MAE, RMSE, R² тооцох скрипт.

"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ===================== ТОХИРГОО =====================
DATA_FILE = "ub_clean_dataset.csv"
TARGET = "PM2.5"
HORIZON = 1                     # хэдэн цагийн дараахыг таамаглах вэ (1 = дараагийн цаг)
LAGS = [0, 1, 2, 3, 6, 12, 24] # PM2.5-ийн хэдэн цагийн өмнөх утга (0 = одоогийн цаг)
ROLL_WINDOWS = [3, 6, 24]       # гулсах дундаж (цагаар)
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15   # үлдсэн 15% нь туршилтын багц
RANDOM_STATE = 42
# ====================================================


def build_features(path):
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time").sort_index()

    # Алга болсон цагийг NaN болгож, дараалал алдагдахаас сэргийлнэ
    df = df.asfreq("h")

    feats = pd.DataFrame(index=df.index)

    # Цаг уурын хувьсагч (одоогийн цагийнх)
    for c in ["Temp", "Humidity", "Wind", "Pressure", "WindDir", "Precip", "DewPoint"]:
        if c in df.columns:
            feats[c] = df[c]

    # Салхины чиг нь тойрог хэмжигдэхүүн (0° ба 360° ижил) тул sin/cos болгоно
    if "WindDir" in feats.columns:
        rad = np.deg2rad(feats["WindDir"])
        feats["WindDir_sin"] = np.sin(rad)
        feats["WindDir_cos"] = np.cos(rad)
        feats = feats.drop(columns="WindDir")

    # Цагийн мөчлөг
    hour = df.index.hour
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # PM2.5-ийн өмнөх утгууд (lag) ба гулсах дундаж
    for k in LAGS:
        feats[f"PM25_lag{k}"] = df[TARGET].shift(k)
    for w in ROLL_WINDOWS:
        feats[f"PM25_roll{w}"] = df[TARGET].rolling(w, min_periods=w).mean()

    # Зорилтот хувьсагч: HORIZON цагийн дараах PM2.5
    feats["target"] = df[TARGET].shift(-HORIZON)

    feats = feats.dropna()
    return feats


def split_by_time(feats):
    # Цаг хугацааны дарааллыг хадгална. Санамсаргүй холихгүй (өгөгдөл алдагдахаас сэргийлнэ).
    n = len(feats)
    i_train = int(n * TRAIN_FRAC)
    i_val = int(n * (TRAIN_FRAC + VAL_FRAC))
    train, val, test = feats.iloc[:i_train], feats.iloc[i_train:i_val], feats.iloc[i_val:]
    return train, val, test


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    feats = build_features(DATA_FILE)
    train, val, test = split_by_time(feats)
    print(f"Нийт: {len(feats)} мөр | сургалт: {len(train)} | баталгаажуулалт: {len(val)} | туршилт: {len(test)}")
    print(f"Сургалт: {train.index.min()} ~ {train.index.max()}")
    print(f"Туршилт: {test.index.min()} ~ {test.index.max()}")

    X_cols = [c for c in feats.columns if c != "target"]
    X_tr, y_tr = train[X_cols], train["target"]
    X_va, y_va = val[X_cols], val["target"]
    X_te, y_te = test[X_cols], test["target"]

    cv = TimeSeriesSplit(n_splits=3)   # хугацааны дарааллыг хадгалсан cross-validation

    # ---------- Жишиг загвар (Persistence): дараагийн цаг = одоогийн цаг ----------
    results = {}
    persist_col = "PM25_lag0"
    results["Persistence (жишиг)"] = {
        "val": metrics(y_va, X_va[persist_col]),
        "test": metrics(y_te, X_te[persist_col]),
    }

    # ---------- Random Forest ----------
    rf_grid = {
        "n_estimators": [200, 400],
        "max_depth": [6, 10, None],
        "min_samples_split": [2, 5, 10],
    }
    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        rf_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    rf_search.fit(X_tr, y_tr)
    rf = rf_search.best_estimator_
    print("\nRandom Forest хамгийн сайн параметр:", rf_search.best_params_)
    results["Random Forest"] = {
        "val": metrics(y_va, rf.predict(X_va)),
        "test": metrics(y_te, rf.predict(X_te)),
    }

    # ---------- XGBoost ----------
    xgb_grid = {
        "learning_rate": [0.03, 0.1],
        "max_depth": [3, 5],
        "n_estimators": [200, 400],
        "subsample": [0.8, 1.0],
    }
    xgb_search = GridSearchCV(
        XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, objective="reg:squarederror"),
        xgb_grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    xgb_search.fit(X_tr, y_tr)
    xgb = xgb_search.best_estimator_
    print("XGBoost хамгийн сайн параметр:", xgb_search.best_params_)
    results["XGBoost"] = {
        "val": metrics(y_va, xgb.predict(X_va)),
        "test": metrics(y_te, xgb.predict(X_te)),
    }

    # ---------- Үр дүнгийн хүснэгт ----------
    rows = []
    for name, r in results.items():
        for part in ["val", "test"]:
            rows.append({"Загвар": name, "Багц": "Баталгаажуулалт" if part == "val" else "Туршилт",
                         **{k: round(v, 3) for k, v in r[part].items()}})
    table = pd.DataFrame(rows)
    print("\n=== ҮР ДҮНГИЙН ХАРЬЦУУЛАЛТ ===")
    print(table.to_string(index=False))
    table.to_csv("model_results.csv", index=False, encoding="utf-8-sig")

    # ---------- Хувьсагчийн ач холбогдол ----------
    imp = pd.DataFrame({
        "feature": X_cols,
        "Random Forest": rf.feature_importances_,
        "XGBoost": xgb.feature_importances_,
    }).sort_values("XGBoost", ascending=False)
    print("\n=== ХУВЬСАГЧИЙН АЧ ХОЛБОГДОЛ (топ 10) ===")
    print(imp.head(10).round(3).to_string(index=False))
    imp.to_csv("feature_importance.csv", index=False, encoding="utf-8-sig")

    # ---------- График ----------
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(y_te.index, y_te.values, label="Бодит", color="black", linewidth=1.2)
    ax[0].plot(y_te.index, rf.predict(X_te), label="Random Forest", alpha=0.8)
    ax[0].plot(y_te.index, xgb.predict(X_te), label="XGBoost", alpha=0.8)
    ax[0].set_title(f"Туршилтын багц: {HORIZON} цагийн дараах PM2.5 (µg/m³)")
    ax[0].legend()
    ax[0].set_ylabel("PM2.5")

    top = imp.head(10).iloc[::-1]
    ax[1].barh(top["feature"], top["XGBoost"])
    ax[1].set_title("XGBoost: хувьсагчийн ач холбогдол (топ 10)")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    print("\nГрафик хадгаллаа: model_comparison.png")

    # ---------- Хамгийн сайн загварыг хадгалах (туршилтын MAE-ээр) ----------
    candidates = {"Random Forest": rf, "XGBoost": xgb}
    best_name = min(candidates, key=lambda n: results[n]["test"]["MAE"])
    joblib.dump({"model": candidates[best_name], "features": X_cols, "horizon": HORIZON},
                "best_model.joblib")
    print(f"Хамгийн сайн загвар: {best_name} -> best_model.joblib")


if __name__ == "__main__":
    main()