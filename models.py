import tensorflow as tf
import xgboost as xgb
import tensorflow_probability as tfp
import tf_keras
import numpy as np
import datetime
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tf_keras.layers import Dense, Dropout
from plotting_functions import plotTrainHistory

tfd = tfp.distributions

class MLStrategy(ABC):

    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        self.X_train, self.y_train = dataFrame.get_train_dataset()
        self.X_val, self.y_val = dataFrame.get_val_dataset()
        self.X_test, self.y_test = dataFrame.get_test_dataset()
    
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test_predict(self):
        pass

class XGBRegressor(MLStrategy):

    def __init__(self, dataFrame):
        super().__init__(dataFrame)
        self.model = xgb.XGBRegressor(n_jobs=16)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def test_predict(self):
        self.dataFrame.data.loc[self.X_test.index, "Z_pred"] = self.model.predict(self.X_test)

class ANNRegressor(MLStrategy):

    def __init__(self, dataFrame):
        super().__init__(dataFrame)
        self.network = None
        self.scaler = StandardScaler()
        self.epochs = 300
        self.batch_size = 128
        self.lr = 0.0001
        self.callbacks = []
        self.tensorboard_callback = None

        self.callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                           patience=30, verbose=1, start_from_epoch=1))

        self.create_network()
        
    def create_network(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(256, kernel_initializer='normal', activation='relu', input_dim=55))
        for i in range(10):
            model.add(tf.keras.layers.Dense(256, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(1))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss=tf.keras.losses.MeanSquaredError())
        self.network = model

    def train(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        
        self.network.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=self.epochs, batch_size=self.batch_size,
                         callbacks=self.callbacks, verbose=0)
        
    def test_predict(self):
        indexes = self.X_test.index
        self.X_test = self.scaler.transform(self.X_test)
        self.dataFrame.data.loc[indexes, "Z_pred"] = self.network.predict(self.X_test)

class ANNSingleGauss(MLStrategy):

    def __init__(self, dataFrame):
        super().__init__(dataFrame)
        self.network = None
        self.scaler = StandardScaler()
        self.epochs = 300
        self.batch_size = 128
        self.lr = 0.0001
        self.callbacks = []
        log_dir = "../../logs/fit/" + "SG" +datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf_keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.callbacks.append(self.tensorboard_callback)
        self.callbacks.append(tf_keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                           patience=30, verbose=1, start_from_epoch=1, restore_best_weights=True))

        self.create_network()
        
    def create_network(self):
        model = tf_keras.Sequential([Dense(256, kernel_initializer='normal', activation='relu', input_shape=(55,)),
                                     Dense(256, kernel_initializer='normal', activation='relu'),
                                     Dense(256, kernel_initializer='normal', activation='relu'),
                                     Dense(256, kernel_initializer='normal', activation='relu'),
                                     Dense(256, kernel_initializer='normal', activation='relu'),
                                     Dense(256, kernel_initializer='normal', activation='relu'),
                                     Dense(256, kernel_initializer='normal', activation='relu'),
                                     Dense(256, kernel_initializer='normal', activation='relu'),
                                     Dense(256, kernel_initializer='normal', activation='relu'),
                                     Dense(256, kernel_initializer='normal', activation='relu'),
                                     Dense(256, kernel_initializer='normal', activation='relu'),
                                     Dense(256, kernel_initializer='normal', activation='relu'),
                                     Dense(2),
                                     tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
                                                                                        scale=1e-3 + tf.math.softplus(0.05 * t[...,1:]))),
                                    ])
        
        # model = tf_keras.Sequential([Dense(256, kernel_initializer='normal', activation='relu', input_shape=(55,)),
        #                              Dense(512, kernel_initializer='normal', activation='relu'),
        #                              Dropout(0.2),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dropout(0.2),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dropout(0.2),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dropout(0.2),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dropout(0.2),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(2),
        #                              tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
        #                                                                                 scale=1e-3 + tf.math.softplus(0.05 * t[...,1:]))),
        #                             ])

        negloglik = lambda y, p_y: -p_y.log_prob(y)

        model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=self.lr),
                      loss=negloglik)
        
        self.network = model
        

    def train(self):
        X_train = self.scaler.fit_transform(self.X_train)
        X_val = self.scaler.transform(self.X_val)
        
        history = self.network.fit(X_train, self.y_train, validation_data=(X_val, self.y_val), epochs=self.epochs, batch_size=self.batch_size,
                         callbacks=self.callbacks, verbose=0)

        plotTrainHistory(history)
        
    def test_predict(self):
        indexes = self.X_test.index
        X_test = self.scaler.transform(self.X_test)
        y_model = self.network(X_test)
        y_pred = y_model.mean().numpy()
        y_std = y_model.stddev().numpy()
        
        self.dataFrame.data.loc[indexes, "Z_pred"] = y_pred
        self.dataFrame.data.loc[indexes, "Z_pred_std"] = y_std
        self.dataFrame.data.loc[indexes, "log_prob"] = -y_model.log_prob(self.y_test.values.reshape(-1,1)).numpy()

class ANNDoubleGauss(MLStrategy):
    def __init__(self, dataFrame):
        super().__init__(dataFrame)
        self.network = None
        self.scaler = StandardScaler()
        self.epochs = 300
        self.batch_size = 128
        self.lr = 0.0001
        self.callbacks = []
        log_dir = "../../logs/fit/" + "DG" +datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf_keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.callbacks.append(self.tensorboard_callback)
        self.callbacks.append(tf_keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                           patience=30, verbose=1, start_from_epoch=1, restore_best_weights=True))

        self.create_network()
        
    def create_network(self):
        num_components = 2
        event_shape = [1]
        params_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)
        
        # model = tf_keras.Sequential([Dense(256, kernel_initializer='normal', activation='relu', input_shape=(55,)),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(params_size),
        #                              tfp.layers.MixtureNormal(num_components, event_shape)])
        
        model = tf_keras.Sequential([Dense(512, kernel_initializer='normal', activation='relu', input_shape=(55,)),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(params_size),
                                     tfp.layers.MixtureNormal(num_components, event_shape)])

        negloglik = tf.autograph.experimental.do_not_convert(lambda y, p_y: -p_y.log_prob(y))

        model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=self.lr),
                      loss=negloglik)
        
        self.network = model
        

    def train(self):
        X_train = self.scaler.fit_transform(self.X_train)
        X_val = self.scaler.transform(self.X_val)
        
        history = self.network.fit(X_train, self.y_train, validation_data=(X_val, self.y_val), epochs=self.epochs, batch_size=self.batch_size,
                         callbacks=self.callbacks, verbose=0)

        plotTrainHistory(history)
        
    def test_predict(self):
        # mean(), stddev()
        indexes = self.X_test.index
        X_test = self.scaler.transform(self.X_test)
        y_model = self.network(X_test)
        y_pred = y_model.mean().numpy()
        y_std = y_model.stddev().numpy()
        self.dataFrame.data.loc[indexes, "Z_pred"] = y_pred
        self.dataFrame.data.loc[indexes, "Z_pred_std"] = y_std
        self.dataFrame.data.loc[indexes, "log_prob"] = -y_model.log_prob(self.y_test.values.reshape(-1,1)).numpy()

        # # Median, stddev()
        # indexes = self.X_test.index
        # X_test = self.scaler.transform(self.X_test)
        # y_model = self.network(X_test)
        # y_samples = y_model.sample(10000).numpy().reshape(-1, 30044)
        # y_pred = np.median(y_samples, axis=0)
        # y_std = y_model.stddev().numpy()
        # self.dataFrame.data.loc[indexes, "Z_pred"] = y_pred
        # self.dataFrame.data.loc[indexes, "Z_pred_std"] = y_std

        # # Mean of narrower gaussian, stddev()
        # indexes = self.X_test.index
        # X_test = self.scaler.transform(self.X_test)
        # y_model = self.network(X_test)
        # mu = y_model.components_distribution.mean().numpy().reshape(-1,2)                
        # sigma = y_model.components_distribution.stddev()          
        # min_std_indices = tf.argmin(sigma[:], axis=1).numpy().reshape(-1)
        # y_pred = mu[np.arange(X_test.shape[0]), min_std_indices]
        # y_std = y_model.stddev().numpy()
        # self.dataFrame.data.loc[indexes, "Z_pred"] = y_pred
        # self.dataFrame.data.loc[indexes, "Z_pred_std"] = y_std

class ANNTripleGauss(MLStrategy):
    def __init__(self, dataFrame):
        super().__init__(dataFrame)
        self.network = None
        self.scaler = StandardScaler()
        self.epochs = 300
        self.batch_size = 128
        self.lr = 0.0001
        self.callbacks = []
        log_dir = "../../logs/fit/" + "TG" +datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf_keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.callbacks.append(self.tensorboard_callback)
        self.callbacks.append(tf_keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                           patience=30, verbose=1, start_from_epoch=1, restore_best_weights=True))

        self.create_network()
        
    def create_network(self):
        num_components = 3
        event_shape = [1]
        params_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)
        
        # model = tf_keras.Sequential([Dense(256, kernel_initializer='normal', activation='relu', input_shape=(55,)),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(params_size),
        #                              tfp.layers.MixtureNormal(num_components, event_shape)])
        
        model = tf_keras.Sequential([Dense(512, kernel_initializer='normal', activation='relu', input_shape=(55,)),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(params_size),
                                     tfp.layers.MixtureNormal(num_components, event_shape)])

        negloglik = tf.autograph.experimental.do_not_convert(lambda y, p_y: -p_y.log_prob(y))

        model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=self.lr),
                      loss=negloglik)
        
        self.network = model
        

    def train(self):
        X_train = self.scaler.fit_transform(self.X_train)
        X_val = self.scaler.transform(self.X_val)
        
        history = self.network.fit(X_train, self.y_train, validation_data=(X_val, self.y_val), epochs=self.epochs, batch_size=self.batch_size,
                         callbacks=self.callbacks, verbose=0)

        plotTrainHistory(history)
        
    def test_predict(self):
        # mean(), stddev()
        indexes = self.X_test.index
        X_test = self.scaler.transform(self.X_test)
        y_model = self.network(X_test)
        y_pred = y_model.mean().numpy()
        y_std = y_model.stddev().numpy()
        self.dataFrame.data.loc[indexes, "Z_pred"] = y_pred
        self.dataFrame.data.loc[indexes, "Z_pred_std"] = y_std
        self.dataFrame.data.loc[indexes, "log_prob"] = -y_model.log_prob(self.y_test.values.reshape(-1,1)).numpy()

class ANNQuadrupleGauss(MLStrategy):
    def __init__(self, dataFrame):
        super().__init__(dataFrame)
        self.network = None
        self.scaler = StandardScaler()
        self.epochs = 300
        self.batch_size = 128
        self.lr = 0.0001
        self.callbacks = []
        log_dir = "../../logs/fit/" + "QG" +datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf_keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.callbacks.append(self.tensorboard_callback)
        self.callbacks.append(tf_keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                           patience=30, verbose=1, start_from_epoch=1, restore_best_weights=True))

        self.create_network()
        
    def create_network(self):
        num_components = 4
        event_shape = [1]
        params_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)
        
        # model = tf_keras.Sequential([Dense(256, kernel_initializer='normal', activation='relu', input_shape=(55,)),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(256, kernel_initializer='normal', activation='relu'),
        #                              Dense(params_size),
        #                              tfp.layers.MixtureNormal(num_components, event_shape)])
        
        model = tf_keras.Sequential([Dense(512, kernel_initializer='normal', activation='relu', input_shape=(55,)),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(0.2),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(params_size),
                                     tfp.layers.MixtureNormal(num_components, event_shape)])

        negloglik = tf.autograph.experimental.do_not_convert(lambda y, p_y: -p_y.log_prob(y))

        model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=self.lr),
                      loss=negloglik)
        
        self.network = model
        

    def train(self):
        X_train = self.scaler.fit_transform(self.X_train)
        X_val = self.scaler.transform(self.X_val)
        
        history = self.network.fit(X_train, self.y_train, validation_data=(X_val, self.y_val), epochs=self.epochs, batch_size=self.batch_size,
                         callbacks=self.callbacks, verbose=0)

        plotTrainHistory(history)
        
    def test_predict(self):
        # mean(), stddev()
        indexes = self.X_test.index
        X_test = self.scaler.transform(self.X_test)
        y_model = self.network(X_test)
        y_pred = y_model.mean().numpy()
        y_std = y_model.stddev().numpy()
        self.dataFrame.data.loc[indexes, "Z_pred"] = y_pred
        self.dataFrame.data.loc[indexes, "Z_pred_std"] = y_std
        self.dataFrame.data.loc[indexes, "log_prob"] = -y_model.log_prob(self.y_test.values.reshape(-1,1)).numpy()
        
class MLModelContext:
    def __init__(self, strategy:MLStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy:MLStrategy):
        self.strategy = strategy

    def train(self):
        self.strategy.train()

    def test_predict(self):
        self.strategy.test_predict()
        
    