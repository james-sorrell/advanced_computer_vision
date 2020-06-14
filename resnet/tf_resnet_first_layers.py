# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision
# Rewritten in TF2 by James Sorrell

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# from keras.applications.resnet50 import ResNet50
# from keras.models import Model
# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input, decode_predictions

from tf_resnet_convblock import ConvLayer, BatchNormLayer, ConvBlock

# NOTE: dependent on your Keras version
#       this script used 2.1.1
# [<keras.engine.topology.InputLayer at 0x112fe4358>,
#  <keras.layers.convolutional.Conv2D at 0x112fe46a0>,
#  <keras.layers.normalization.BatchNormalization at 0x112fe4630>,
#  <keras.layers.core.Activation at 0x112fe4eb8>,
#  <keras.layers.pooling.MaxPooling2D at 0x10ed4be48>,
#  <keras.layers.convolutional.Conv2D at 0x1130723c8>,
#  <keras.layers.normalization.BatchNormalization at 0x113064710>,
#  <keras.layers.core.Activation at 0x113092dd8>,
#  <keras.layers.convolutional.Conv2D at 0x11309e908>,
#  <keras.layers.normalization.BatchNormalization at 0x11308a550>,
#  <keras.layers.core.Activation at 0x11312ac88>,
#  <keras.layers.convolutional.Conv2D at 0x1131207b8>,
#  <keras.layers.convolutional.Conv2D at 0x1131b8da0>,
#  <keras.layers.normalization.BatchNormalization at 0x113115550>,
#  <keras.layers.normalization.BatchNormalization at 0x1131a01d0>,
#  <keras.layers.merge.Add at 0x11322f0f0>,
#  <keras.layers.core.Activation at 0x113246cf8>]


# define some additional layers so they have a forward function
class ReLULayer(keras.layers.Layer):
  def __init__(self):
    super(ReLULayer, self).__init__()

  def forward(self, X):
    return tf.nn.relu(X)

  def get_params(self):
    return []

class MaxPoolLayer(keras.layers.Layer):
  def __init__(self, dim):
    super(MaxPoolLayer, self).__init__()
    self.dim = dim

  def call(self, X):
    return tf.nn.max_pool(
      X,
      ksize=[1, self.dim, self.dim, 1],
      strides=[1, 2, 2, 1],
      padding='SAME'
    )

  def get_params(self):
    return []

class PartialResNet:
  def __init__(self):
    self.layers = [
      # before conv block
      ConvLayer(d=7, mi=3, mo=64, stride=2, padding='SAME'),
      BatchNormLayer(64),
      ReLULayer(),
      MaxPoolLayer(dim=3),
      # conv block
      ConvBlock(mi=64, fm_sizes=[64, 64, 256], stride=1),
    ]
    self.input_ = keras.Input(shape=(224, 224, 3), dtype=tf.float32)
    self.createModel_(self.input_)

  def createModel_(self, X):
    Y = X
    print("\nLayers.Size: {}".format(len(self.layers)))
    for layer in self.layers:
      Y = layer(Y)
      print("\t\t{}_____{}".format(layer.__class__.__name__, Y.shape))
    print("\nCreated.\n")
    self.model = keras.Model(inputs=[X], outputs=[Y]) 

  def copyFromKerasLayers(self, layers):
    print("\nCopying from keras layers:\n")
    index = 0
    for layer in layers:
      print("\t{}\t{}\t:{}".format(index , layer.output_shape, layer))
      index+=1
    self.layers[0].copyFromKerasLayers(layers[2])
    self.layers[1].copyFromKerasLayers(layers[3])
    self.layers[4].copyFromKerasLayers(layers[7:])
    print("\nCopied.")

  def forward(self, X):
    for layer in self.layers:
      X = layer.forward(X)
    return X

  def predict(self, X):
    return self.model(X)


if __name__ == '__main__':
  # you can also set weights to None, it doesn't matter
  resnet = keras.applications.resnet50.ResNet50(weights='imagenet')

  # you can determine the correct layer
  # by looking at resnet.layers in the console
  partial_model = keras.Model(
    inputs=resnet.input,
    outputs=resnet.layers[17].output
  )
  print(partial_model.summary())
  # for layer in partial_model.layers:
  #   layer.trainable = False

  my_partial_resnet = PartialResNet()

  # make a fake image
  X = np.random.random((1, 224, 224, 3))

  # get keras output
  keras_output = partial_model.predict(X)

  # # get my model output
  # init = tf.variables_initializer(my_partial_resnet.get_params())

  # # note: starting a new session messes up the Keras model
  # session = keras.backend.get_session()
  # my_partial_resnet.set_session(session)
  # session.run(init)

  # first, just make sure we can get any output
  first_output = my_partial_resnet.predict(X)

  # copy params from Keras model
  my_partial_resnet.copyFromKerasLayers(partial_model.layers)

  # compare the 2 models
  output = my_partial_resnet.predict(X)

  print("\nKeras Output Shape: {}".format(keras_output.shape))
  print("My Output Shape: {}\n".format(output.shape))

  diff = np.abs(output - keras_output).sum()
  if diff < 1e-10:
    print("Everything's great!")
  else:
    print("diff = %s" % diff)
