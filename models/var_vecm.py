import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import (
    coint_johansen,
    VECM
)

# ---------------- VAR ----------------
def fit_var(returns, maxlags=10, ic="aic"):
    """
    Fit VAR model with lag selection
    """
    model = VAR(returns)
    results = model.fit(maxlags=maxlags, ic=ic)
    return results


# ---------------- Johansen Cointegration ----------------
def johansen_test(returns, det_order=0, k_ar_diff=1):
    """
    Johansen cointegration test
    """
    result = coint_johansen(returns, det_order, k_ar_diff)
    return result


# ---------------- VECM ----------------
def fit_vecm(returns, coint_rank, k_ar_diff=1):
    """
    Fit VECM model
    """
    model = VECM(
        returns,
        k_ar_diff=k_ar_diff,
        coint_rank=coint_rank,
        deterministic="co"
    )
    results = model.fit()
    return results
