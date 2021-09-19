#
# Copyleft 2021 Megazone Cloud Corp.
#

#Import Python Modules
import tensorflow as tf
from tensorflow import keras
import time
import os

if tf.__version__[0] != '2':
    raise ImportError("TensorFlow 2 is required for this example")
#end of if  

# Define GPU ditribution strategy    
print(tf.config.list_physical_devices("GPU"))
gpus = tf.config.experimental.list_logical_devices("GPU")
strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
print('\n\nRunning on multiple GPUs ', [gpu.name for gpu in gpus])
print("\n\nNumber of accelerators(GPU): ", strategy.num_replicas_in_sync,"\n\n")

# The input data and labels.
mnist = tf.keras.datasets.mnist
print("\n\n==============================Downloading MNIST DataSet from Internet. It's will take 10 Sec..==============================\n\n")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, x_test) = (x_train / 255.0, x_test / 255.0)

# Add a channels dimension.
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

def create_train_dataset():
    print("==============================Processing Training DataSet==============================\n\n")
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(2000, drop_remainder=True)
    train_ds = train_ds.map(lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))
    return train_ds.repeat()
###end of def:

def create_test_dataset():
    print("==============================Processing Test  DataSet==============================\n\n")
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(10000).batch(2000, drop_remainder=True)
    test_ds = test_ds.map(lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))
    return test_ds.repeat()
###end of def:

# standard tf.keras.Sequential class
def create_model():
    model = tf.keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')])
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), 
                  optimizer = tf.keras.optimizers.Adam(),
                  metrics=['sparse_categorical_accuracy'])
    return model
###end of def:

start = time.time() ## Save time point of starting

def main():
    # Get the training dataset.
    print("==============================Getting Training DataSet==============================\n\n")
    ds1 = create_train_dataset()

    print("==============================Getting Test DataSet==============================\n\n")
    ds2 = create_test_dataset()

    # Create an instance of the model.
    print("==============================Building Model==============================\n\n")
    model = create_model()

    print("==============================Model Training==============================\n\n")


    with strategy.scope():
      model.fit(ds1, steps_per_epoch=30, epochs=50)

      print("\n\n==============================Checking the result==============================\n\n")
      (loss, accuracy) = model.evaluate(ds2, steps=5)

      print("Validation loss: {}".format(loss))

      print("Validation accuracy: {}%".format(100.0 * accuracy))
      print("\n\n==============================Job Done...==============================")
   #end of with:
###end of def:

if __name__ == '__main__':
    main()
###end of if

print("Running Time :", round(time.time() - start, 2),"(Sec.)")  ## Total running time

#
###end of codes
#