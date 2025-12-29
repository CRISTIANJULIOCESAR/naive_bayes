# bayes

Paquete para scoring Bayesiano sobre matrices **dummy (0/1)** y export estilo “Proyecto 42”.

## Filosofía (simple)
- `X` **siempre** es tu `df_dummies` (0/1).
- El usuario **ya debe traer su target `y` listo** (0/1) o una columna `target` en su dataframe de targets.
- El paquete solo se encarga de:
  1) alinear `X` y `y` por IDs (si lo necesitas),
  2) calcular `Epsilon` y `Score`,
  3) sacar **una probabilidad final** por fila,
  4) exportar Excel.

## Instalación (desde GitHub)
En Colab:
```bash
pip install git+https://github.com/TU_USUARIO/bayes.git
```

## Uso (recomendado: alinear por IDs)
```python
import bayes

id_cols = ["FOLIO_I","FOLIO_INT"]

# df_targets ya trae la columna target (0/1) lista
X, y = bayes.align_xy_by_ids(
    X_dummies=df_dummies,       # incluye id_cols + reglas dummy
    df_targets=df_targets,      # incluye id_cols + target
    id_cols=id_cols,
    target_col="target"
)

out = bayes.bayes_run_and_export(
    X, y,
    model_name="diabetes_Y_hipertension",
    min_cases=5,
    alpha=1.0,
    threshold=0.5,
    out_xlsx="/content/resultados.xlsx"
)
```

## Output Excel
- `Resumen_modelo`
- `Tabla_Resultados`
- `Weights`
- `Predicciones`
