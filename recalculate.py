import numpy as np
import data_frame
import models

filePath = "../KiDS-DR5-WCScut_x_DESI-DR1-small.fits"


config = {"num_components":2}
for i in range(1,6):
    df = data_frame.DataFrame(filePath, "QSO", data_frame.MaxFiller())

    model = models.MLModelContext(strategy=models.ANNRegressor(df, config))
    model.load_weights(f"../models/experiment2/ANN_{i}/variables/variables")
    model.test_predict()

    df.data.to_csv(f"../dataframes/experiment2.1/{model.getModelName()}_{i}.csv")