import numpy as np
import data_frame
import models
import evaluator as ev
import matplotlib.pyplot as plt
from utils import load_config

filePath = "../KiDS-DR5-WCScut_x_DESI-DR1-small.fits"
config_path = "configs/config.yml"

config = load_config(config_path)

for i in range(1,6):
    df = data_frame.DataFrame(filePath, "QSO", data_frame.MaxFiller())

    model = models.MLModelContext(strategy=models.MixtureGaussian(df, config))
    model.train()
    model.test_predict()

    model.strategy.network.save(f"../models/experiment2/{model.getModelName()}_{i}")
    df.data.to_csv(f"../dataframes/experiment2/{model.getModelName()}_{i}.csv")