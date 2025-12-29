# bayes (minimal)

Paquete **general** y minimalista.

## Input
- `X`: `pandas.DataFrame` con **0/1** (filas = instancias, columnas = variables).
- `y`: vector **0/1** (target ya preparado).

## Import
```python
import bayes
```

## API (compatible con scikit-learn)
- `BayesClassifier.fit(X, y)`
- `BayesClassifier.predict_proba(X)` -> `Nx2` `[P(y=0), P(y=1)]`
- `BayesClassifier.predict(X)` -> `0/1`
- `BayesClassifier.decision_function(X)` -> log-odds continuo
- Atributos después de fit: `weights_`, `feature_table_`

## ROC/AUC con CV=3 (sklearn)
```python
import bayes
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve

clf = bayes.BayesClassifier(alpha=1.0, min_cases=5)

y_proba = cross_val_predict(
    clf, X, y,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    method="predict_proba"
)[:, 1]

auc_val = roc_auc_score(y, y_proba)
fpr, tpr, _ = roc_curve(y, y_proba)
```
