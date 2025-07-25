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
from plotting_functions import diag_plot, dist_plot
from utils import plot_to_image

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

    @abstractmethod
    def getModelName(self):
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
        log_dir = "../logs/fit/" + f"{self.getModelName()}" +datetime.datetime.now().strftime("-%m%d-%H-%M")
        self.tensorboard_callback = tf_keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        self.callbacks.append(self.tensorboard_callback)
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

    def getModelName(self):
        return "ANN"

class MixtureGaussian(MLStrategy):
    def __init__(self, dataFrame, config):
        super().__init__(dataFrame)
        self.config = config
        self.network = None
        self.scaler = StandardScaler()
        self.epochs = 2
        self.batch_size = 128
        self.lr = 0.0001
        self.callbacks = []
    
        def log_plot(epoch, logs):
            X_test, y_test = self.dataFrame.get_random_test_dataset()
            X_faint, y_faint = self.dataFrame.get_faint_test_dataset()
            X_test = self.scaler.transform(X_test)
            X_faint = self.scaler.transform(X_faint)
    
            y_model_test = self.network(X_test)
            test_pred = y_model_test.mean().numpy()
            test_std = y_model_test.stddev().numpy()
    
            y_model_faint = self.network(X_faint)
            faint_pred = y_model_faint.mean().numpy()
            faint_std = y_model_faint.stddev().numpy()
    
            figure = diag_plot(y_test, test_pred, test_std, y_faint, faint_pred, faint_std)
            image = plot_to_image(figure)
    
            with file_writer.as_default():
                tf.summary.image(f"{self.getModelName()}_diag_plot", image, step=epoch)
    
            X_all = self.X_test.copy()
            y_all = self.y_test.copy()
            N = 10  

            np.random.seed(2)
            selected_indices = np.random.choice(X_all.index, size=N, replace=False)
    
            for idx in selected_indices:
                x_input = self.scaler.transform(X_all.loc[[idx]].values)
                true_y = y_all.loc[idx]
                fig = dist_plot(x_input, true_y, self.network)
                image = plot_to_image(fig)
                with file_writer.as_default():
                    tf.summary.image(f"{self.getModelName()}_distribution_sample_{idx}", image, step=epoch)
    
        
        timestamp = datetime.datetime.now().strftime("-%m%d-%H-%M")
        loss_log_dir = f"../../logs/fit/loss/{self.getModelName()}{timestamp}"
        plots_log_dir = f"../../logs/fit/plots/{self.getModelName()}{timestamp}"
        file_writer = tf.summary.create_file_writer(plots_log_dir)
    
        self.diag_plot_callback = tf_keras.callbacks.LambdaCallback(on_epoch_end=log_plot)
        self.tensorboard_callback = tf_keras.callbacks.TensorBoard(log_dir=loss_log_dir, histogram_freq=1)
    
        self.callbacks.append(self.tensorboard_callback)
        self.callbacks.append(self.diag_plot_callback)
        self.callbacks.append(tf_keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='min',
            patience=30, verbose=1,
            start_from_epoch=1, restore_best_weights=True
        ))
    
        self.create_network()
        
    def create_network(self):
        num_components = self.config["num_components"]
        event_shape = [1]
        params_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)

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

    def test_predict(self):
        indexes = self.X_test.index
        X_test = self.scaler.transform(self.X_test)
        y_model = self.network(X_test)
        y_pred = y_model.mean().numpy()
        y_std = y_model.stddev().numpy()
        
        self.dataFrame.data.loc[indexes, "Z_pred"] = y_pred
        self.dataFrame.data.loc[indexes, "Z_pred_std"] = y_std
        self.dataFrame.data.loc[indexes, "Z_spec_prob"] = np.exp(y_model.log_prob(self.y_test.values.reshape(-1,1)).numpy())

    def getModelName(self):
        return f"MG_{self.config["num_components"]}_components"
        

        
class MLModelContext:
    def __init__(self, strategy:MLStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy:MLStrategy):
        self.strategy = strategy

    def train(self):
        self.strategy.train()

    def test_predict(self):
        self.strategy.test_predict()

    def getModelName(self):
        return self.strategy.getModelName()
        
    