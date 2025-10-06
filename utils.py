import matplotlib.pyplot as plt
import io
import tensorflow as tf
import yaml

import models

def plot_to_image(figure):
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  plt.close(figure)
  buf.seek(0)
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  image = tf.expand_dims(image, 0)
  return image

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.full_load(file)
    return config

def load_model(modelPath, modelStrategy, df, config):
    model = models.MLModelContext(strategy=modelStrategy(df, config))
    model.load_weights(modelPath)
    return model.strategy.network