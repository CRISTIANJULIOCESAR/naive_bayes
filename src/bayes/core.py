from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import expit


# ----------------------------
# Targets compuestos (y)
# ----------------------------
def make_target_and(df: pd.DataFrame, cols: List[str], positive: int = 1, name: str = "target_AND") -> pd.Series:
    """
    y = 1 si TODAS las columnas == positive (AND).
    """
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"No existe columna en df: {c}")
    y = np.ones(len(df), dtype=bool)
    for c in cols:
        y &= (pd.to_numeric(df[c], errors="coerce") == positive).to_numpy()
    return pd.Series(y.astype(int), index=df.index, name=name)


def make_target_or(df: pd.DataFrame, cols: List[str], positive: int = 1, name: str = "target_OR") -> pd.Series:
    """
    y = 1 si AL MENOS UNA columna == positive (OR).
    """
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"No existe columna en df: {c}")
    y = np.zeros(len(df), dtype=bool)
    for c in cols:
        y |= (pd.to_numeric(df[c], errors="coerce") == positive).to_numpy()
    return pd.Series(y.astype(int), index=df.index, name=name)


def make_target_equals(df: pd.DataFrame, col: str, positive_value, name: Optional[str] = None) -> pd.Series:
    """
    y = 1 si df[col] == positive_value.
    Sirve para categorías tipo 'Sí', 'M', 'F', etc.
    """
    if col not in df.columns:
        raise ValueError(f"No existe columna en df: {col}")
    if name is None:
        name = f"{col}_equals_{positive_value}"
    y = (df[col] == positive_value).astype(int)
    y.name = name
    return y


# ----------------------------
# Alineación segura X/y por IDs
# ----------------------------
def align_xy_by_ids(
    X_dummies: pd.DataFrame,
    df_targets: pd.DataFrame,
    *,
    id_cols: List[str],
    target_col: str,
    how: str = "inner",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Alinea X y y por llaves (ej. FOLIO_I, FOLIO_INT). 100% robusto contra reorder/reset.

    Requisitos:
      - X_dummies debe contener id_cols como columnas (o al menos se las agregas antes)
      - df_targets debe contener id_cols y target_col
    """
    for c in id_cols:
        if c not in X_dummies.columns:
            raise ValueError(f"X_dummies NO tiene columna ID requerida: {c}")
        if c not in df_targets.columns:
            raise ValueError(f"df_targets NO tiene columna ID requerida: {c}")
    if target_col not in df_targets.columns:
        raise ValueError(f"df_targets no tiene target_col={target_col}")

    XY = X_dummies.merge(df_targets[id_cols + [target_col]], on=id_cols, how=how)

    y = pd.to_numeric(XY[target_col], errors="coerce")
    mask = y.notna()
    XY = XY.loc[mask].copy()

    y_arr = XY[target_col].astype(int).to_numpy()
    X = XY.drop(columns=id_cols + [target_col])

    # fuerza a binario (por si acaso)
    X = (X.fillna(0) > 0).astype(int)

    return X, y_arr


# ----------------------------
# Core Bayes: tabla + pesos + proba
# ----------------------------
def _group_sizes(columns: Iterable[str], sep: str = "|") -> Dict[str, int]:
    gs: Dict[str, int] = {}
    for c in columns:
        g = str(c).split(sep, 1)[0]
        gs[g] = gs.get(g, 0) + 1
    return gs


def bayes_fit(
    X: pd.DataFrame,
    y: Union[np.ndarray, pd.Series, list],
    *,
    min_cases: int = 5,
    alpha: float = 1.0,
    sep: str = "|",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Entrena: calcula tabla estilo Proyecto 42 + pesos (Score por regla).
    Asume y binaria {0,1} y X dummies {0,1}.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X debe ser DataFrame.")
    y_arr = np.asarray(y).astype(int)
    if len(y_arr) != len(X):
        raise ValueError("X y y deben tener mismo número de filas.")

    Xv = (X.fillna(0) > 0).astype(int)

    mask = (y_arr == 1)
    Nc = int(mask.sum())
    N = int(len(y_arr))
    if N == 0:
        raise ValueError("X está vacío.")
    Nnc = N - Nc
    prior = (Nc / N) if N else 0.0

    gs = _group_sizes(Xv.columns, sep=sep)

    rows = []
    for col in Xv.columns:
        parts = str(col).split(sep, 1)
        group = parts[0]
        category = parts[1] if len(parts) > 1 else ""

        vals = Xv[col].to_numpy(dtype=int)
        Nx = int(vals.sum())
        nCx = int((vals[mask] == 1).sum())
        n_x_nc = Nx - nCx

        Kx = int(gs.get(group, 1))

        p_c = (nCx + alpha) / (Nc + alpha * Kx) if (Nc + alpha * Kx) > 0 else 0.0
        p_nc = (n_x_nc + alpha) / (Nnc + alpha * Kx) if (Nnc + alpha * Kx) > 0 else 0.0

        score = float(np.log(p_c / p_nc)) if (p_c > 0 and p_nc > 0) else 0.0
        if nCx < int(min_cases):
            score = 0.0

        p_c_given_x = (nCx / Nx) if Nx > 0 else 0.0
        denom = np.sqrt(Nx * prior * (1.0 - prior))
        epsilon = float((Nx * (p_c_given_x - prior) / denom) if denom > 0 else 0.0)

        rows.append({
            "Subcategoría": group,
            "Valor_Variable": category,
            "Descripción": group,   # tú luego puedes mapearlo a nombres largos
            "Respuesta": category,
            "N(CX)": nCx,
            "N(X)": Nx,
            "P(C|X)": p_c_given_x,
            "P(C)": prior,
            "Epsilon": epsilon,
            "Score": score,
            "rule": col,
        })

    tabla = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    weights = tabla.set_index("rule")["Score"].astype(float)

    return tabla, weights


def bayes_predict_proba(X: pd.DataFrame, weights: Union[pd.Series, Dict[str, float]]) -> np.ndarray:
    """
    Probabilidad final por fila: sigma(sum Scores de reglas activas).
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights)
    Xv = (X.fillna(0) > 0).astype(int)
    common = Xv.columns.intersection(weights.index)
    log_odds = Xv[common].dot(weights[common]).astype(float).to_numpy()
    return expit(log_odds)


def bayes_run_and_export(
    X: pd.DataFrame,
    y: Union[np.ndarray, pd.Series, list],
    *,
    model_name: str = "modelo",
    min_cases: int = 5,
    alpha: float = 1.0,
    sep: str = "|",
    out_xlsx: Optional[str] = None,
    threshold: float = 0.5,
) -> Dict[str, pd.DataFrame]:
    """
    Corre todo y opcionalmente exporta Excel estilo Proyecto 42.

    Excel sheets:
      - Resumen_modelo
      - Tabla_Resultados
      - Weights
      - Predicciones
    """
    y_arr = np.asarray(y).astype(int)
    N = int(len(y_arr))
    Nc = int((y_arr == 1).sum())
    prior = (Nc / N) if N else 0.0

    resumen = pd.DataFrame([{
        "Modelo": model_name,
        "N(C)": Nc,
        "P(C)": round(prior, 6),
        "N": N,
        "N(-C)": N - Nc,
        "min_cases": int(min_cases),
        "alpha": float(alpha),
        "threshold": float(threshold),
    }])

    tabla, weights = bayes_fit(X, y_arr, min_cases=min_cases, alpha=alpha, sep=sep)
    proba = pd.Series(bayes_predict_proba(X, weights), name=f"P({model_name})")
    pred = (proba >= threshold).astype(int).rename(f"Pred_{threshold}")

    tabla_out = tabla.drop(columns=["rule"]).copy()
    weights_out = weights.sort_values(ascending=False).rename("Score").reset_index().rename(columns={"index": "rule"})
    pred_out = pd.concat([proba, pred], axis=1)

    if out_xlsx:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            resumen.to_excel(writer, sheet_name="Resumen_modelo", index=False)
            tabla_out.to_excel(writer, sheet_name="Tabla_Resultados", index=False)
            weights_out.to_excel(writer, sheet_name="Weights", index=False)
            pred_out.to_excel(writer, sheet_name="Predicciones", index=False)

    return {
        "resumen": resumen,
        "tabla_resultados": tabla_out,
        "weights": weights_out,
        "predicciones": pred_out,
    }
