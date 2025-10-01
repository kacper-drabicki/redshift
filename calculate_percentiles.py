import tensorflow as tf
import numpy as np
import pandas as pd
import data_frame
import models
from utils import load_config
from sklearn.preprocessing import StandardScaler

config_path = "configs/config.yml"
config = load_config(config_path)

DATAFRAME_PATH = "../dataframes/experiment3/MG_3_components_3.csv"
MODEL_PATH = f"../models/experiment2/MG_{config["num_components"]}_components_3/variables/variables"

PERCENTILES = np.arange(0.01, 1, 0.01)

def percentile(dist, low=0.0, high=10.0, tol=1e-6, max_iter=100):

    low = tf.constant(low, shape=(PERCENTILES.shape[0], dist.shape[0], 1), dtype=tf.float32)
    high = tf.constant(high, shape=(PERCENTILES.shape[0], dist.shape[0], 1), dtype=tf.float32)
    p = tf.constant(PERCENTILES, shape=(PERCENTILES.shape[0], 1), dtype=tf.float32)
    
    for _ in range(100):
        mid = (low + high) / 2
        cdf_mid = tf.exp(dist.log_cdf(mid))
        high = tf.where((cdf_mid > p)[:, :, None], mid, high)
        low = tf.where((cdf_mid < p)[:, :, None], mid, low)

    return ((low+high)/2).numpy().T

def load_df(dataPath):
    df = data_frame.DataFrame("../KiDS-DR5-WCScut_x_DESI-DR1-small.fits", "QSO",)
    df.load_df(dataPath)

    return df

def load_model(modelPath, df):
    model = models.MLModelContext(models.MixtureGaussian(df, config))
    model.load_weights(modelPath)
    
    return model.strategy.network

def calculate_percentiles(df, model):
    X_train, y_train = df.get_train_dataset()
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_test, y_test = df.get_test_dataset()
    index = X_test.index
    X_test = scaler.transform(X_test)

    dist = model(X_test)
    percentiles = percentile(dist)

    for i in range(0,99):
        df.data.loc[index, f"percentile_{i+1}"] = percentiles[:,:, i].reshape(-1)

df = load_df(DATAFRAME_PATH)
model = load_model(MODEL_PATH, df)
calculate_percentiles(df, model)
df.data.to_csv(f"{DATAFRAME_PATH}")