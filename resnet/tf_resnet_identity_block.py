# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision
# Rewritten in Tensorflow 2.2.0 by James Sorrell

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from tf_resnet_convblock import ConvLayer, BatchNormLayer


class IdentityBlock(keras.layers.Layer):
  def __init__(self, mi, fm_sizes, activation=tf.nn.relu):
    super(IdentityBlock, self).__init__()
    # conv1, conv2, conv3
    # note: # feature maps shortcut = # feauture maps conv 3
    assert(len(fm_sizes) == 3)

    # note: kernel size in 2nd conv is always 3
    #       so we won't bother including it as an arg

    self.session = None
    self.f = tf.nn.relu
    
    # init main branch
    # Conv -> BN -> F() ---> Conv -> BN -> F() ---> Conv -> BN
    self.conv1 = ConvLayer(1, mi, fm_sizes[0], 1)
    self.bn1   = BatchNormLayer(fm_sizes[0])
    self.conv2 = ConvLayer(3, fm_sizes[0], fm_sizes[1], 1, 'SAME')
    self.bn2   = BatchNormLayer(fm_sizes[1])
    self.conv3 = ConvLayer(1, fm_sizes[1], fm_sizes[2], 1)
    self.bn3   = BatchNormLayer(fm_sizes[2])

    # in case needed later
    self.layers = [
      self.conv1, self.bn1,
      self.conv2, self.bn2,
      self.conv3, self.bn3,
    ]

  def call(self, X):
    FX = self.conv1(X)
    FX = self.bn1(FX)
    FX = self.f(FX)
    FX = self.conv2(FX)
    FX = self.bn2(FX)
    FX = self.f(FX)
    FX = self.conv3(FX)
    FX = self.bn3(FX)
    # sum + activation
    Y = self.f(tf.math.add(FX, X))
    return Y

  def copyFromKerasLayers(self, layers):
    #assert(len(layers) == 10)
    # <keras.layers.convolutional.Conv2D at 0x7fa44255ff28>,
    # <keras.layers.normalization.BatchNormalization at 0x7fa44250e7b8>,
    # <keras.layers.core.Activation at 0x7fa44252d9e8>,
    # <keras.layers.convolutional.Conv2D at 0x7fa44253af60>,
    # <keras.layers.normalization.BatchNormalization at 0x7fa4424e4f60>,
    # <keras.layers.core.Activation at 0x7fa442494828>,
    # <keras.layers.convolutional.Conv2D at 0x7fa4424a2da0>,
    # <keras.layers.normalization.BatchNormalization at 0x7fa44244eda0>,
    # <keras.layers.merge.Add at 0x7fa44245d5c0>,
    # <keras.layers.core.Activation at 0x7fa44240aba8>
    self.conv1.copyFromKerasLayers(layers[0])
    self.bn1.copyFromKerasLayers(layers[1])
    self.conv2.copyFromKerasLayers(layers[3])
    self.bn2.copyFromKerasLayers(layers[4])
    self.conv3.copyFromKerasLayers(layers[6])
    self.bn3.copyFromKerasLayers(layers[7])

  def get_params(self):
    params = []
    for layer in self.layers:
      params += layer.get_params()
    return params


if __name__ == '__main__':
  identity_block = IdentityBlock(mi=256, fm_sizes=[64, 64, 256])
  # make a fake image
  X = np.random.random((1, 224, 224, 256))
  output = identity_block(X)
  print("output.shape:", output.shape)
