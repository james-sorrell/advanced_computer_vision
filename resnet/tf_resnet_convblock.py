# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision
# Rewritten in Tensorflow 2.2.0 by James Sorrell

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def init_filter(d, mi, mo, stride):
  # Filter initialisation, Xavier
  return (np.random.randn(d, d, mi, mo) * np.sqrt(2.0 / (d * d * mi))).astype(np.float32)

class ConvLayer(keras.layers.Layer):
  # Kernel Size, Input Depth, Number of Filters
  def __init__(self, d, mi, mo, stride=2, padding='SAME'):
    super(ConvLayer, self).__init__()
    self.W = tf.Variable(init_filter(d, mi, mo, stride), trainable=True)
    self.b = tf.Variable(np.zeros(mo, dtype=np.float32), trainable=True)
    self.stride = stride
    self.padding = padding

  def call(self, X):
    Y = tf.nn.conv2d(
      X,
      self.W,
      strides=[1, self.stride, self.stride, 1],
      padding=self.padding
    )
    Y += self.b
    return Y

  def copyFromKerasLayers(self, layer):
    # only 1 layer to copy from
    W, b = layer.get_weights()
    self.W.assign(W)
    self.b.assign(b)

  def get_params(self):
    return [self.W, self.b]


class BatchNormLayer(keras.layers.Layer):
  def __init__(self, D):
    super(BatchNormLayer, self).__init__()
    self.running_mean = tf.Variable(np.zeros(D, dtype=np.float32), trainable=False)
    self.running_var  = tf.Variable(np.ones(D, dtype=np.float32), trainable=False)
    self.gamma        = tf.Variable(np.ones(D, dtype=np.float32), trainable=True)
    self.beta         = tf.Variable(np.zeros(D, dtype=np.float32), trainable=True)

  def call(self, X):
    return tf.compat.v1.nn.batch_normalization(
      X,
      self.running_mean,
      self.running_var,
      self.beta,
      self.gamma,
      1e-3
    )

  def copyFromKerasLayers(self, layer):
    # only 1 layer to copy from
    # order:
    # gamma, beta, moving mean, moving variance
    gamma, beta, running_mean, running_var = layer.get_weights()
    self.running_mean.assign(running_mean)
    self.running_var.assign(running_var)
    self.gamma.assign(gamma)
    self.beta.assign(beta)

  def get_params(self):
    return [self.running_mean, self.running_var, self.gamma, self.beta]


class ConvBlock(keras.layers.Layer):
  def __init__(self, mi, fm_sizes, stride=2, activation=tf.compat.v1.nn.relu):
    super(ConvBlock, self).__init__()
    # conv1, conv2, conv3
    # note: # feature maps shortcut = # feauture maps conv 3
    assert(len(fm_sizes) == 3)

    # note: kernel size in 2nd conv is always 3
    #       so we won't bother including it as an arg

    # note: stride only applies to conv 1 in main branch
    #       and conv in shortcut, otherwise stride is 1

    self.f = tf.compat.v1.nn.relu
    
    # init main branch
    # Conv -> BN -> F() ---> Conv -> BN -> F() ---> Conv -> BN
    self.conv1 = ConvLayer(1, mi, fm_sizes[0], stride)
    self.bn1   = BatchNormLayer(fm_sizes[0])
    self.conv2 = ConvLayer(3, fm_sizes[0], fm_sizes[1], 1, 'SAME')
    self.bn2   = BatchNormLayer(fm_sizes[1])
    self.conv3 = ConvLayer(1, fm_sizes[1], fm_sizes[2], 1)
    self.bn3   = BatchNormLayer(fm_sizes[2])

    # init shortcut branch
    # Conv -> BN
    self.convs = ConvLayer(1, mi, fm_sizes[2], stride)
    self.bns   = BatchNormLayer(fm_sizes[2])

    # in case needed later
    self.layers = [
      self.conv1, self.bn1,
      self.conv2, self.bn2,
      self.conv3, self.bn3,
      self.convs, self.bns
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
    # shortcut branch
    SX = self.convs(X)
    SX = self.bns(SX)
    # sum + activation
    Y = self.f(tf.math.add(FX, SX))
    return Y

  def copyFromKerasLayers(self, layers):
    # print("\nCopying from keras layers:\n")
    # index = 0
    # for layer in layers:
    #   print("\t{} __ {}".format(index , layer))
    #   index+=1
    # [<keras.layers.convolutional.Conv2D at 0x117bd1978>,
    #  <keras.layers.normalization.BatchNormalization at 0x117bf84a8>,
    #  <keras.layers.core.Activation at 0x117c15fd0>,
    #  <keras.layers.convolutional.Conv2D at 0x117c23be0>,
    #  <keras.layers.normalization.BatchNormalization at 0x117c51978>,
    #  <keras.layers.core.Activation at 0x117c93518>,
    #  <keras.layers.convolutional.Conv2D at 0x117cc1518>,
    #  <keras.layers.convolutional.Conv2D at 0x117d21630>,
    #  <keras.layers.normalization.BatchNormalization at 0x117cd2a58>,
    #  <keras.layers.normalization.BatchNormalization at 0x117d44b00>,
    #  <keras.layers.merge.Add at 0x117dae748>,
    #  <keras.layers.core.Activation at 0x117da2eb8>]
    self.conv1.copyFromKerasLayers(layers[0])
    self.bn1.copyFromKerasLayers(layers[1])
    self.conv2.copyFromKerasLayers(layers[3])
    self.bn2.copyFromKerasLayers(layers[4])
    self.conv3.copyFromKerasLayers(layers[6])
    self.bn3.copyFromKerasLayers(layers[8])
    self.convs.copyFromKerasLayers(layers[7])
    self.bns.copyFromKerasLayers(layers[9])

  def get_params(self):
    params = []
    for layer in self.layers:
      params += layer.get_params()
    return params


if __name__ == '__main__':
  conv_block = ConvBlock(mi=3, fm_sizes=[64, 64, 256], stride=1)
  # make a fake image
  X = np.random.random((1, 224, 224, 3))
  output = conv_block(X)
  print("output.shape:", output.shape)
