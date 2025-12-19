import numpy as np
import pandas as pd
import cvxpy as cp

# --------------------------------------------------
# Helper: portfolio statistics
# --------------------------------------------------
def portfolio_stats(weights, mean_returns, cov_matrix, rf):
    portfolio_return = weights @ mean_returns
    portfolio_vol = cp.quad_form(weights, cov_matrix)
    sharpe = (portfolio_return - rf) / cp.sqrt(portfolio_vol)
    return portfolio_return, portfolio_vol, sharpe

def random_portfolios(returns, n_portfolios=2000):
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    n_assets = returns.shape[1]

    results = []

    for _ in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= weights.sum()  # full investment, long-only

        port_return = weights @ mean_returns
        port_risk = np.sqrt(weights @ cov_matrix @ weights)
        concentration = np.sum(weights ** 2)  # HHI

        results.append({
            "Return": port_return,
            "Risk": port_risk,
            "Concentration": concentration
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# Mean-Variance (Minimum Variance)
# --------------------------------------------------
def min_variance_portfolio(returns):
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    n = len(mean_returns)

    w = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(w, cov_matrix))
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return pd.Series(w.value, index=returns.columns)


# --------------------------------------------------
# Maximum Sharpe Ratio (Long-only)
# --------------------------------------------------
def max_sharpe_portfolio(returns, rf, n_targets=50):
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    n = len(mean_returns)

    target_returns = np.linspace(
        mean_returns.min(),
        mean_returns.max(),
        n_targets
    )

    best_sharpe = -np.inf
    best_weights = None

    for target in target_returns:
        w = cp.Variable(n)

        portfolio_return = w @ mean_returns
        portfolio_variance = cp.quad_form(w, cov_matrix)

        problem = cp.Problem(
            cp.Minimize(portfolio_variance),
            constraints=[
                cp.sum(w) == 1,
                w >= 0,
                portfolio_return >= target
            ]
        )

        problem.solve(solver=cp.SCS, verbose=False)

        if w.value is not None:
            ret = w.value @ mean_returns
            vol = np.sqrt(w.value @ cov_matrix @ w.value)
            sharpe = (ret - rf) / vol if vol > 0 else -np.inf

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = w.value

    return pd.Series(best_weights, index=returns.columns)


# --------------------------------------------------
# Efficient Frontier
# --------------------------------------------------
def efficient_frontier(returns, n_portfolios=25):
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    n = len(mean_returns)

    target_returns = np.linspace(
        mean_returns.min(),
        mean_returns.max(),
        n_portfolios
    )

    weights_list = []
    risks = []
    rets = []

    for target in target_returns:
        w = cp.Variable(n)

        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w @ mean_returns == target
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        if w.value is not None:
            weights_list.append(w.value)
            risks.append(np.sqrt(w.value @ cov_matrix @ w.value))
            rets.append(target)

    ef = pd.DataFrame(weights_list, columns=returns.columns)
    ef["Return"] = rets
    ef["Risk"] = risks

    return ef
