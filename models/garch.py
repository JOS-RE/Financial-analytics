from arch import arch_model

def fit_garch(returns, p=1, q=1):
    """
    Fit GARCH(p,q) model
    Returns fitted model
    """
    model = arch_model(
        returns * 100,
        mean="Constant",
        vol="Garch",
        p=p,
        q=q,
        dist="normal"
    )

    results = model.fit(disp="off")
    return results
