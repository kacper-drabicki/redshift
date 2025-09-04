import tensorflow as tf
import xgboost as xgb
import tensorflow_probability as tfp
import tf_keras
import numpy as np
import datetime
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tf_keras.layers import Dense, Dropout, BatchNormalization
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

    @abstractmethod
    def load_weights(self, path):
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
    def __init__(self, dataFrame, config):
        super().__init__(dataFrame)
        self.config = config
        self.network = None
        self.scaler = StandardScaler()
        self.epochs = 300
        self.batch_size = 128
        self.lr = 0.0001
        self.callbacks = []
        timestamp = datetime.datetime.now().strftime("-%m%d-%H-%M")
        loss_log_dir = f"../logs/fit/loss/{self.getModelName()}{timestamp}"
        self.tensorboard_callback = tf_keras.callbacks.TensorBoard(log_dir=loss_log_dir, histogram_freq=1)
        
        self.callbacks.append(self.tensorboard_callback)
        self.callbacks.append(tf_keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                           patience=30, verbose=1, start_from_epoch=1))

        self.create_network()
        self.scaler.fit(self.X_train)
        
    def load_weights(self, path):
        self.network.load_weights(path).expect_partial()
    
    def create_network(self):
        model = tf_keras.Sequential()
        model.add(tf_keras.layers.Dense(256, kernel_initializer='normal', activation='relu', input_dim=55))
        for i in range(10):
            model.add(tf_keras.layers.Dense(256, kernel_initializer='normal', activation='relu'))
        model.add(tf_keras.layers.Dense(1))

        model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=self.lr),
                      loss=tf_keras.losses.MeanSquaredError())
        self.network = model

    def train(self):
        X_train = self.scaler.transform(self.X_train)
        X_val = self.scaler.transform(self.X_val)
        
        history = self.network.fit(X_train, self.y_train, validation_data=(X_val, self.y_val), epochs=self.epochs, batch_size=self.batch_size,
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
        self.epochs = 300
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
    
            with plots_file_writer.as_default():
                tf.summary.image(f"Diag_plot", image, step=epoch)
    
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
                with plots_file_writer.as_default():
                    tf.summary.image(f"Distribution_sample_{idx}", image, step=epoch)

        class MyCustomCallback(tf_keras.callbacks.Callback):
            def __init__(self, scaler, X_faint_val, y_faint_val, log_dir):
                super().__init__()
                self.scaler = scaler
                self.X_test = X_faint_val
                self.y_test = y_faint_val
                self.writer = tf.summary.create_file_writer(log_dir)
        
            def on_epoch_end(self, epoch, logs=None):
                X_scaled = self.scaler.transform(self.X_test)
                loss = self.model.evaluate(X_scaled, self.y_test, verbose=0)
        
                with self.writer.as_default():
                    tf.summary.scalar(f"faint_epoch_loss", loss, step=epoch)
      
        timestamp = datetime.datetime.now().strftime("-%m%d-%H-%M")
        loss_log_dir = f"../logs/fit/loss/{self.getModelName()}{timestamp}"
        plots_log_dir = f"../logs/fit/plots/{self.getModelName()}{timestamp}"
        val_loss_log_dir = f"../logs/fit/loss/{self.getModelName()}{timestamp}/faint_validation"
        plots_file_writer = tf.summary.create_file_writer(plots_log_dir)

        X_faint_val, y_faint_val = self.dataFrame.get_faint_val_dataset()
        self.diag_plot_callback = tf_keras.callbacks.LambdaCallback(on_epoch_end=log_plot)
        self.tensorboard_callback = tf_keras.callbacks.TensorBoard(log_dir=loss_log_dir, histogram_freq=1)
        self.val_callback = MyCustomCallback(self.scaler, X_faint_val, y_faint_val, val_loss_log_dir)
    
        self.callbacks.append(self.tensorboard_callback)
        self.callbacks.append(self.diag_plot_callback)
        self.callbacks.append(self.val_callback)
        self.callbacks.append(tf_keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='min',
            patience=30, verbose=1,
            start_from_epoch=1, restore_best_weights=True
        ))
    
        self.create_network()
        self.scaler.fit(self.X_train)
        
    def create_network(self):
        class DropoutDict(dict):
            def __missing__(self, key):
                return 0 if key == 1 else 0.2
        dropout_dict = DropoutDict()
        
        num_components = self.config["num_components"]
        event_shape = [1]
        params_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)
        dropout_rate = dropout_dict[num_components]
        
        model = tf_keras.Sequential([Dense(512, kernel_initializer='normal', activation='relu', input_shape=(55,)),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(dropout_rate),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(dropout_rate),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(dropout_rate),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(dropout_rate),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dropout(dropout_rate),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     Dense(params_size),
                                     tfp.layers.MixtureNormal(num_components, event_shape)])
        
        ## Added batch normalization
        # model = tf_keras.Sequential([Dense(512, kernel_initializer='normal', activation='relu', input_shape=(55,)), 
        #                              BatchNormalization(),
        #                              Dense(512, kernel_initializer='normal', activation='relu'), 
        #                              BatchNormalization(),
        #                              Dropout(dropout_rate),
        #                              Dense(512, kernel_initializer='normal', activation='relu'),
        #                              BatchNormalization(),
        #                              Dense(512, kernel_initializer='normal', activation='relu'),
        #                              BatchNormalization(),
        #                              Dropout(dropout_rate),
        #                              Dense(512, kernel_initializer='normal', activation='relu'), 
        #                              BatchNormalization(),
        #                              Dense(512, kernel_initializer='normal', activation='relu'), 
        #                              BatchNormalization(),
        #                              Dropout(dropout_rate), 
        #                              Dense(512, kernel_initializer='normal', activation='relu'), 
        #                              BatchNormalization(),
        #                              Dense(512, kernel_initializer='normal', activation='relu'), 
        #                              BatchNormalization(),
        #                              Dropout(dropout_rate), 
        #                              Dense(512, kernel_initializer='normal', activation='relu'),
        #                              BatchNormalization(),
        #                              Dense(512, kernel_initializer='normal', activation='relu'), 
        #                              BatchNormalization(),
        #                              Dropout(dropout_rate), 
        #                              Dense(512, kernel_initializer='normal', activation='relu'), 
        #                              BatchNormalization(),
        #                              Dense(512, kernel_initializer='normal', activation='relu'),
        #                              BatchNormalization(),
        #                              Dense(params_size), 
        #                              tfp.layers.MixtureNormal(num_components, event_shape)])
        
        
        negloglik = tf.autograph.experimental.do_not_convert(lambda y, p_y: -p_y.log_prob(y))

        model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=self.lr),
                      loss=negloglik)
        
        self.network = model

    def load_weights(self, path):
        self.network.load_weights(path).expect_partial()
        
    def train(self):
        X_train = self.scaler.transform(self.X_train)
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


class BayesianNN(MLStrategy):
    def __init__(self, dataFrame, config):
        super().__init__(dataFrame)
        self.config = config
        self.network = None
        self.scaler = StandardScaler()
        self.epochs = 1000
        self.batch_size = 128
        self.lr = 0.0001
        self.callbacks = []
      
        timestamp = datetime.datetime.now().strftime("-%m%d-%H-%M")
        loss_log_dir = f"../logs/fit/loss/{self.getModelName()}{timestamp}"
        self.tensorboard_callback = tf_keras.callbacks.TensorBoard(log_dir=loss_log_dir, histogram_freq=1)
    
        self.callbacks.append(self.tensorboard_callback)
        self.callbacks.append(tf_keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='min',
            patience=30, verbose=1,
            start_from_epoch=1, restore_best_weights=True
        ))
    
        self.create_network()
        self.scaler.fit(self.X_train)

    def create_network(self):        
        num_components = 3
        event_shape = [1]
        params_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)

        # Bayesian params
        scale = self.X_train.shape[0]
        activation = "tanh"
        neurons = 512
        
        model = tf_keras.Sequential([Dense(512, kernel_initializer='normal', activation='relu', input_shape=(55,)),
                                     BatchNormalization(),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     BatchNormalization(),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     BatchNormalization(),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     BatchNormalization(),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     BatchNormalization(),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     BatchNormalization(),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     BatchNormalization(),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     BatchNormalization(),
                                     Dense(512, kernel_initializer='normal', activation='relu'),
                                     BatchNormalization(),
                                     tfp.layers.DenseFlipout(neurons, activation=activation,
                                                       kernel_divergence_fn=lambda q,p,ignore: tfp.distributions.kl_divergence(q, p) / scale,
                                                       bias_divergence_fn=lambda q,p,ignore: tfp.distributions.kl_divergence(q, p) / scale),
                                     tfp.layers.DenseFlipout(neurons, activation=activation,
                                                       kernel_divergence_fn=lambda q,p,ignore: tfp.distributions.kl_divergence(q, p) / scale,
                                                       bias_divergence_fn=lambda q,p,ignore: tfp.distributions.kl_divergence(q, p) / scale),
                                     tfp.layers.DenseFlipout(neurons, activation=activation,
                                                       kernel_divergence_fn=lambda q,p,ignore: tfp.distributions.kl_divergence(q, p) / scale,
                                                       bias_divergence_fn=lambda q,p,ignore: tfp.distributions.kl_divergence(q, p) / scale),
                                     Dense(params_size),
                                     tfp.layers.MixtureNormal(num_components, event_shape)])

        
        
        negloglik = tf.autograph.experimental.do_not_convert(lambda y, p_y: -p_y.log_prob(y))

        batch_num = int(self.X_train.shape[0] / self.batch_size)

        boundaries = [200 * batch_num] #, 50 * batch_num]
        values = [0.001, 0.0001] # ,0.00001]
        lr_schedule = tf_keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, values)

        model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=lr_schedule), #self.lr
                      loss=negloglik)
        
        self.network = model

    def load_weights(self, path):
        self.network.load_weights(path).expect_partial()
      
    def train(self):
        X_train = self.scaler.transform(self.X_train)
        X_val = self.scaler.transform(self.X_val)
        
        history = self.network.fit(X_train, self.y_train, validation_data=(X_val, self.y_val), epochs=self.epochs, batch_size=self.batch_size,
                         callbacks=self.callbacks, verbose=1)

    def test_predict(self):
        indexes = self.X_test.index
        X_test = self.scaler.transform(self.X_test)
        means = []
        variances = []
        probs = []
        for _ in range(100):
            dist = self.network(X_test)
            means.append(dist.mean().numpy())
            variances.append(dist.variance().numpy())
            probs.append(np.exp(dist.log_prob(self.y_test.values.reshape(-1,1)).numpy()))
        
        y_pred = np.array(means).mean(axis=0)
        y_aleatoric_variance = np.array(variances).mean(axis=0)
        y_epistemic_variance = np.array(means).std(axis=0) ** 2
        y_std = np.sqrt(y_epistemic_variance + y_aleatoric_variance)
        y_prob = np.array(probs).mean(axis=0)

        self.dataFrame.data.loc[indexes, "Z_pred"] = y_pred
        self.dataFrame.data.loc[indexes, "Z_pred_std"] = y_std
        self.dataFrame.data.loc[indexes, "Z_spec_prob"] = y_prob
        
    def getModelName(self):
        return "BNN"    
        
class MLModelContext:
    def __init__(self, strategy:MLStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy:MLStrategy):
        self.strategy = strategy

    def load_weights(self, path):
        self.strategy.load_weights(path)

    def train(self):
        self.strategy.train()

    def test_predict(self):
        self.strategy.test_predict()

    def getModelName(self):
        return self.strategy.getModelName()
    