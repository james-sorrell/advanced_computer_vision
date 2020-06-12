from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class CNN:
  def __init__(self, k):
    self.model = keras.Sequential(
      [
        keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(units=200, activation=None),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(units=k, activation='softmax')
      ]
    )
    self.model.compile(
      loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy']
    )

# helper
def y2indicator(Y):
  N = len(Y)
  K = len(set(Y))
  I = np.zeros((N, K))
  I[np.arange(N), Y] = 1
  return I

def getData():
  data = pd.read_csv('../large_files/fashionmnist/fashion-mnist_train.csv')
  data = data.values
  X = data[:, 1:].reshape(-1, 28, 28, 1) / 255.0
  Y = data[:, 0].astype(np.int32)
  K = len(set(Y))
  # there's another cost function we can use
  # where we can just pass in the integer labels directly
  # just like Tensorflow / Theano
  Y = y2indicator(Y)
  return X, Y, K

def main():
    """
    Main Function
    """
    # Inputs, Outputs, Num Classes
    X, Y, K = getData()
    classifier = CNN(K)
    r = classifier.model.fit(X, Y, validation_split=0.33, epochs=15, batch_size=32)
    print("Returned:", r)

    # print the available keys
    # should see: dict_keys(['val_loss', 'acc', 'loss', 'val_acc'])
    print(r.history.keys())

    # plot some data
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    # accuracies
    plt.plot(r.history['accuracy'], label='acc')
    plt.plot(r.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()