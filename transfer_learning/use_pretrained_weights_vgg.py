# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision

from tensorflow import keras
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

# re-size all the images to this
IMAGE_SIZE = [100, 100] # feel free to change depending on dataset

# training config:
EPOCHS = 5
BATCH_SIZE = 32

def getData(display=True):
  # https://www.kaggle.com/paultimothymooney/blood-cells
  # train_path = '../large_files/blood_cell_images/TRAIN'
  # test_path = '../large_files/blood_cell_images/TEST'

  # https://www.kaggle.com/moltean/fruits
  # train_path = '../large_files/fruits-360/Training'
  # test_path = '../large_files/fruits-360/Test'
  train_path = '../large_files/fruits-360-small/Training'
  test_path = '../large_files/fruits-360-small/Test'

  # useful for getting number of files
  image_files = glob(train_path + '/*/*.jp*g')
  test_image_files = glob(test_path + '/*/*.jp*g')

  # useful for getting number of classes
  folders = glob(train_path + '/*')

  if display is True:
    # look at an image for fun
    plt.imshow(image.img_to_array(image.load_img(np.random.choice(image_files))).astype('uint8'))
    plt.show(block=False)

  return train_path, test_path, image_files, test_image_files, len(folders)

class VGGTransferModel:
  def __init__(self, k):
    # Get VGG feature extractor
    vgg = keras.applications.VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    for layer in vgg.layers:
      layer.trainable = False
    x = keras.layers.Flatten()(vgg.output)
    prediction = keras.layers.Dense(k, activation='softmax')(x)
    # create model
    self.model = keras.models.Model(inputs=vgg.input, outputs=prediction)
    # view the structure of the model
    self.model.summary()
    # tell the model what cost and optimization method to use
    self.model.compile(
      loss='categorical_crossentropy',
      optimizer='rmsprop',
      metrics=['accuracy']
    )

class TransferLearner:
  def __init__(self):
    self.train_path, self.test_path, self.image_files, self.test_image_files, k = getData()
    self.transfer_model = VGGTransferModel(k)
    self.createImageGenerator()

  def createImageGenerator(self):
    # create an instance of ImageDataGenerator
    self.gen = keras.preprocessing.image.ImageDataGenerator(
      rotation_range=20,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=0.2,
      horizontal_flip=True,
      vertical_flip=True,
      preprocessing_function=keras.applications.vgg16.preprocess_input
    )

  def get_confusion_matrix(self, data_path, N):
    # we need to see the data in the same order
    # for both predictions and targets
    print("Generating confusion matrix", N)
    predictions = []
    targets = []
    i = 0
    for x, y in self.gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=BATCH_SIZE * 2):
      i += 1
      if i % 50 == 0:
        print(i)
      p = self.transfer_model.model.predict(x)
      p = np.argmax(p, axis=1)
      y = np.argmax(y, axis=1)
      predictions = np.concatenate((predictions, p))
      targets = np.concatenate((targets, y))
      if len(targets) >= N:
        break

    cm = confusion_matrix(targets, predictions)
    return cm

  def createGenerators(self):
    # create generators
    self.train_generator = self.gen.flow_from_directory(
      self.train_path,
      target_size=IMAGE_SIZE,
      shuffle=True,
      batch_size=BATCH_SIZE,
    )
    self.test_generator = self.gen.flow_from_directory(
      self.test_path,
      target_size=IMAGE_SIZE,
      shuffle=True,
      batch_size=BATCH_SIZE,
    )

  def fit(self):
    """ Returns results and confusion matricies """
    r = self.transfer_model.model.fit_generator(
      self.train_generator,
      validation_data=self.test_generator,
      epochs=EPOCHS,
      steps_per_epoch=len(self.image_files) // BATCH_SIZE,
      validation_steps=len(self.test_image_files) // BATCH_SIZE,
    )
    cm = self.get_confusion_matrix(self.train_path, len(self.image_files))
    print(cm)
    test_cm = self.get_confusion_matrix(self.test_path, len(self.test_image_files))
    print(test_cm)
    return r, cm, test_cm, self.labels

  def test_generator(self):
    # test generator to see how it works and some other useful things
    # get label mapping for confusion matrix plot later
    test_gen = self.gen.flow_from_directory(self.test_path, target_size=IMAGE_SIZE)

    print(test_gen.class_indices)
    self.labels = [None] * len(test_gen.class_indices)
    for k, v in test_gen.class_indices.items():
      self.labels[v] = k

    # should be a strangely colored image (due to VGG weights being BGR)
    for x, y in test_gen:
      print("min:", x[0].min(), "max:", x[0].max())
      plt.title(self.labels[np.argmax(y[0])])
      plt.imshow(x[0])
      plt.show(block=False)
      break

def main():
  """
  Main Function
  """
  learner = TransferLearner()
  learner.test_generator()
  learner.createGenerators()
  r, cm, test_cm, labels = learner.fit()

  # loss
  plt.plot(r.history['loss'], label='train loss')
  plt.plot(r.history['val_loss'], label='val loss')
  plt.legend()
  plt.show()

  # accuracies
  plt.plot(r.history['accuracy'], label='train acc')
  plt.plot(r.history['val_accuracy'], label='val acc')
  plt.legend()
  plt.show()

  from util import plot_confusion_matrix
  plot_confusion_matrix(cm, labels, title='Train confusion matrix')
  plot_confusion_matrix(test_cm, labels, title='Validation confusion matrix')

if __name__ == '__main__':
    main()