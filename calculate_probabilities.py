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

PHOTOZ = np.linspace(0.05, 5, 100).reshape(-1,1,1)

def load_df(dataPath):
    df = data_frame.DataFrame("../KiDS-DR5-WCScut_x_DESI-DR1-small.fits", "QSO",)
    df.load_df(dataPath)

    return df

def load_model(modelPath, df):
    model = models.MLModelContext(models.MixtureGaussian(df, config))
    model.load_weights(modelPath)
    
    return model.strategy.network

def calculate_probabilities(df, model):
    X_train, y_train = df.get_train_dataset()
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_test, y_test = df.get_test_dataset()
    index = X_test.index
    X_test = scaler.transform(X_test)

    dist = model(X_test)
    probabilities = dist.log_prob(PHOTOZ).numpy().T

    for i, photoz in enumerate(PHOTOZ):
        df.data.loc[index, f"prob_{photoz}"] = np.exp(probabilities[:, i]).reshape(-1)

df = load_df(DATAFRAME_PATH)
model = load_model(MODEL_PATH, df)
calculate_probabilities(df, model)
df.data.to_csv(f"{DATAFRAME_PATH}")