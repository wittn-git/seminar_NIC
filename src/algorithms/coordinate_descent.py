from sklearn.linear_model import Lasso, LassoLarsCV
import numpy as np

def run_coorddesc(X, y, args):

    time_steps = args["time_steps"]

    lasso_lars_cv = LassoLarsCV(cv=5)
    lasso_lars_cv.fit(X, y)
    alpha = lasso_lars_cv.alpha_

    params = {
        "alpha": float(alpha)
    }

    coefficients = []

    for _ in range(time_steps):
        lasso = Lasso(alpha=alpha, warm_start=True, max_iter=1)
        lasso.fit(X, y)
        coefficients.append(lasso.coef_)
    
    return np.array(coefficients), params