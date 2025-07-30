import matplotlib.pyplot as plt
import io
import tensorflow as tf
import yaml

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