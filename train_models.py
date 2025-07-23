import numpy as np
import data_frame
import models
import evaluator as ev
import matplotlib.pyplot as plt

filePath = "../KiDS-DR5-WCScut_x_DESI-DR1-small.fits"

for i in range(1,6):
    df = data_frame.DataFrame(filePath, "QSO", data_frame.MaxFiller())

    model = models.MLModelContext(strategy=models.ANNDoubleGauss(df))
    model.train()
    model.test_predict()

    model.strategy.network.save(f"../models/double_gauss_{i}")
    df.data.to_csv(f"../dataframes/double_gauss_{i}")