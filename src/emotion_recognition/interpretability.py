"""Optional, data-dependent feature-importance helpers."""

import pandas as pd


def permutation_importance_table(model, features: pd.DataFrame, labels, *, random_seed: int = 42) -> pd.DataFrame:
    """Calculate permutation importance for a scikit-learn compatible model."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(model, features, labels, random_state=random_seed, scoring="f1_macro")
    return pd.DataFrame({"feature": features.columns, "importance_mean": result.importances_mean, "importance_std": result.importances_std}).sort_values("importance_mean", ascending=False)