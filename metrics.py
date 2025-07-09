import numpy

def redshift_error(y_true, y_pred):
    nominator = y_pred - y_true
    denominator = y_true + 1
    redshift_err = nominator / denominator 
    mean_readshift_err = float(redshift_err.mean())
    redshift_err_std = float(redshift_err.std())
    return (mean_readshift_err, redshift_err_std)