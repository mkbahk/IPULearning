#
# Acquiring dataset
#
# wget https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip
# unzip gzip.zip 
# rm gzip.zip
# pip install python-mnist sklearn
#

#
# Preparing dataset
#
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dropout, Dense
from mnist import MNIST
import os

# load the entire EMNIST dataset as numpy arrays (this might take a while)
print("Loading Dataset,...Please wait for moment...")
emnist_data = MNIST(path='/home/mkbahk/emnist_png', return_type='numpy')
emnist_data.select_emnist('byclass')
x_train, y_train = emnist_data.load_training()
x_test, y_test = emnist_data.load_testing()
print("Loading complete...")

# print the shapes
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

img_side = 28

# Reshape tensors to [n, y, x, 1] and normalize the pixel values between [0, 1]
x_train = x_train.reshape(-1, img_side, img_side, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, img_side, img_side, 1).astype('float32') / 255.0

print(x_train.shape, x_test.shape)

# get number of classes
unique_classes = np.unique(y_train)
num_classes = len(unique_classes)

input_shape = (img_side, img_side, 1)

# weight the classes (to combat the imbalance)
class_weights = dict(enumerate(compute_class_weight('balanced', unique_classes, y_train)))

# Convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

#
# Create Model Structure(layers)
#
kernel_size = (5, 5)

def createmodel():
    return Sequential([
        Convolution2D(16, kernel_size=kernel_size, padding='same', input_shape=input_shape, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Dropout(0.4),
        Convolution2D(32, kernel_size=kernel_size, padding='same', activation= 'relu'), #strides=2,
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Dropout(0.4),
        Convolution2D(64, kernel_size=kernel_size, padding='same', activation= 'relu'),
        MaxPooling2D(pool_size =(2,2)),
        BatchNormalization(),
        Dropout(0.4),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax'),
    ])
###end of def:

# setting up model to run on cpu, or gpu when avaiable
print("Building Model Structure...")
model = createmodel()

print("Compiling Model...")
model.compile(loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])

model.summary()
# tf.keras.utils.plot_model(model, show_shapes=True)


#
# Train Model
#
es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=10,
    restore_best_weights=True)

print("Training Model...")
model.fit(x_train, y_train,
          #class_weight=class_weights,
          batch_size=10000,
          epochs=200,
          verbose=1,
          shuffle=True,
          validation_data=(x_test, y_test),
          callbacks=[es])
print("Done Model Training...")

#
# Evaluate Model
#
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_pred = model.predict(x_test)
print(y_pred)

#
# Convert Model to Javascript
#
print("Save Model, name cnn_emnist.h5...")
model.save("cnn_emnist.h5")

#
# pip install tensorflowjs
# rm -rf jsmodel/
# tensorflowjs_converter --input_format keras "cnn_emnist.h5" ./jsmodel
# zip -r jsmodel.zip jsmodel/
#
