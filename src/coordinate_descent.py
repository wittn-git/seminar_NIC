from sklearn.linear_model import Lasso, LassoLarsCV
import numpy as np

def run_coorddesc(X, y, args):

    time_steps = args["time_steps"]

    lasso_lars_cv = LassoLarsCV(cv=5)
    lasso_lars_cv.fit(X, y)
    alpha = lasso_lars_cv.alpha_
    
    lasso = Lasso(alpha=alpha, warm_start=True)
    lasso.fit(X, y)
    coefficients = np.tile(lasso.coef_, (time_steps, 1)).T
    return coefficients