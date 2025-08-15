import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def blocked_cv_slices(n, n_splits=3):
    fold = n // (n_splits + 1)
    for i in range(n_splits):
        train_end = fold * (i + 1)
        val_end = fold * (i + 2)
        yield slice(0, train_end), slice(train_end, val_end)
