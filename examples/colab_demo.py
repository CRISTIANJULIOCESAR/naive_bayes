import bayes

# 👇 El usuario ya debe traer df_targets con target (0/1) listo.
# df_dummies debe incluir FOLIO_I y FOLIO_INT como columnas.

id_cols = ["FOLIO_I", "FOLIO_INT"]

X, y = bayes.align_xy_by_ids(
    X_dummies=df_dummies,
    df_targets=df_targets,
    id_cols=id_cols,
    target_col="target"
)

out = bayes.bayes_run_and_export(
    X, y,
    model_name="mi_modelo",
    min_cases=5,
    alpha=1.0,
    threshold=0.5,
    out_xlsx="/content/resultados.xlsx"
)

print(out["resumen"])
print(out["tabla_resultados"].head(10))
