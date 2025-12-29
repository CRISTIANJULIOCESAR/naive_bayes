from __future__ import annotations

from typing import Dict, Iterable, Union

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin


def _group_sizes(columns: Iterable[str], sep: str = "|") -> Dict[str, int]:
    """Cuenta cuántas columnas pertenecen a cada grupo. Si no hay `sep`, cada columna es su propio grupo."""
    gs: Dict[str, int] = {}
    for c in columns:
        cs = str(c)
        group = cs.split(sep, 1)[0] if sep in cs else cs
        gs[group] = gs.get(group, 0) + 1
    return gs


class BayesClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador Bayes-like para matrices dummy (0/1), compatible con scikit-learn.

    Score por feature:
      Score = log( P(x|C) / P(x|~C) ) con Laplace smoothing (alpha)
    y la prob final:
      P(y=1|X) = sigmoid( intercept + sum(Score_i * x_i) )

    Parámetros
    ---------
    alpha : float
        Laplace smoothing.
    min_cases : int
        Si N(C∩X) < min_cases => Score=0 (robustez).
    sep : str
        Separador opcional para columnas tipo `grupo|categoria` (sirve para ajustar smoothing por grupo).
    threshold : float
        Umbral para predict().
    use_prior : bool
        Si True, agrega intercept = log(prior/(1-prior)).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        min_cases: int = 5,
        sep: str = "|",
        threshold: float = 0.5,
        use_prior: bool = True,
    ):
        self.alpha = alpha
        self.min_cases = min_cases
        self.sep = sep
        self.threshold = threshold
        self.use_prior = use_prior

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[np.ndarray, pd.Series, list]):
        X_df = self._to_df(X)
        y_arr = np.asarray(y).astype(int)

        if len(y_arr) != len(X_df):
            raise ValueError("X y y deben tener mismo número de filas.")
        if set(np.unique(y_arr)) - {0, 1}:
            raise ValueError("y debe ser binaria {0,1}.")

        # scikit-learn convention
        self.classes_ = np.array([0, 1], dtype=int)

        Xv = (X_df.fillna(0) > 0).astype(int)

        mask = (y_arr == 1)
        Nc = int(mask.sum())
        N = int(len(y_arr))
        if N == 0:
            raise ValueError("X está vacío.")
        Nnc = N - Nc

        prior = (Nc / N) if N else 0.0
        self.prior_ = float(prior)

        # intercept
        if self.use_prior and 0.0 < prior < 1.0:
            self.intercept_ = float(np.log(prior / (1.0 - prior)))
        else:
            self.intercept_ = 0.0

        gs = _group_sizes(Xv.columns, sep=self.sep)

        rows = []
        weights = {}

        # Edge: single class only
        if Nc == 0 or Nnc == 0:
            for col in Xv.columns:
                weights[str(col)] = 0.0
                rows.append({
                    "feature": str(col),
                    "group": str(col).split(self.sep, 1)[0] if self.sep in str(col) else str(col),
                    "N(CX)": 0,
                    "N(X)": int(Xv[col].sum()),
                    "P(C|X)": float(prior),
                    "P(C)": float(prior),
                    "Epsilon": 0.0,
                    "Score": 0.0,
                })
            self.feature_table_ = pd.DataFrame(rows)
            self.weights_ = pd.Series(weights, dtype=float)
            self.feature_names_in_ = np.array(list(Xv.columns), dtype=object)
            return self

        for col in Xv.columns:
            col_s = str(col)
            group = col_s.split(self.sep, 1)[0] if self.sep in col_s else col_s

            vals = Xv[col].to_numpy(dtype=int)
            Nx = int(vals.sum())
            nCx = int((vals[mask] == 1).sum())
            n_x_nc = Nx - nCx

            Kx = int(gs.get(group, 1))

            p_c = (nCx + self.alpha) / (Nc + self.alpha * Kx) if (Nc + self.alpha * Kx) > 0 else 0.0
            p_nc = (n_x_nc + self.alpha) / (Nnc + self.alpha * Kx) if (Nnc + self.alpha * Kx) > 0 else 0.0

            score = float(np.log(p_c / p_nc)) if (p_c > 0 and p_nc > 0) else 0.0
            if nCx < int(self.min_cases):
                score = 0.0

            p_c_given_x = (nCx / Nx) if Nx > 0 else 0.0
            denom = np.sqrt(Nx * prior * (1.0 - prior))
            epsilon = float((Nx * (p_c_given_x - prior) / denom) if denom > 0 else 0.0)

            rows.append({
                "feature": col_s,
                "group": group,
                "N(CX)": nCx,
                "N(X)": Nx,
                "P(C|X)": p_c_given_x,
                "P(C)": prior,
                "Epsilon": epsilon,
                "Score": score,
            })
            weights[col_s] = score

        self.feature_table_ = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
        self.weights_ = pd.Series(weights, dtype=float)
        self.feature_names_in_ = np.array(list(Xv.columns), dtype=object)
        return self

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X_df = self._to_df(X)
        Xv = (X_df.fillna(0) > 0).astype(int)

        common = Xv.columns.intersection(self.weights_.index)
        log_odds = Xv[common].dot(self.weights_[common]).astype(float).to_numpy() + float(self.intercept_)
        return log_odds

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        p1 = expit(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= float(self.threshold)).astype(int)

    @staticmethod
    def _to_df(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        X = np.asarray(X)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])


# Compat: nombre anterior (no recomendado)
BayesDummyClassifier = BayesClassifier
