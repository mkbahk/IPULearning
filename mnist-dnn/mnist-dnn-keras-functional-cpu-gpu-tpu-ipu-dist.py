import tensorflow as tf
from tensorflow import keras
import time
import os
from tensorflow.python import ipu

# cpu, gpu, tpu, ipu통합 검출 및 수행환경 예
if tf.__version__[0] != '2':
    raise ImportError("TensorFlow 2 is required for this example")
###end of if

print("Tensorflow version " + tf.__version__)
print(tf.config.list_physical_devices("CPU"))

if tf.config.list_physical_devices("GPU") != []:
   print(tf.config.list_physical_devices("GPU"))
else:
   print("GPU가 없어라...")
###end of if

if tf.config.list_physical_devices("TPU") != []:
   print(tf.config.list_physical_devices("TPU"))
else:
   print("TPU가 없은께...알아서 하랑께")
###end of if

if tf.config.list_physical_devices("IPU") != []:
   print(tf.config.list_physical_devices("IPU"))
else:
   print("IPU를 없어...빨랑 사야제...그래야 인생이 편히..")
###end of if

# Detect hardware
try:
   tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
except ValueError:
   tpu_resolver = None
   gpus = tf.config.experimental.list_logical_devices("GPU")
   ipus = tf.config.experimental.list_logical_devices("IPU")
###end of try

# Select appropriate distribution strategy
if tpu_resolver:
   tf.config.experimental_connect_to_cluster(tpu_resolver)
   tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
   strategy = tf.distribute.experimental.TPUStrategy(tpu_resolver)
   print('\n\nRunning on TPU ', tpu_resolver.cluster_spec().as_dict()['worker'])
elif len(ipus) > 1:
   print('\n\nRunning on multiple IPUs ', [ipu.name for ipu in ipus])
   # Configure the IPU system
   cfg = ipu.utils.create_ipu_config()
   cfg = ipu.utils.auto_select_ipus(cfg, 16)
   ipu.utils.configure_ipu_system(cfg)
   # Create an IPU distribution strategy.
   strategy = ipu.ipu_strategy.IPUStrategy()
elif len(gpus) > 1:
   strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
   print('\n\nRunning on multiple GPUs ', [gpu.name for gpu in gpus])
elif len(gpus) == 1:
   strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
   print('\n\nRunning on single GPU ', gpus[0].name)
else:
   strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
   print('\n\nRunning on CPU\n\n')
###end of if

print("\n\nNumber of accelerators: ", strategy.num_replicas_in_sync,"\n\n")

# The input data and labels.
mnist = tf.keras.datasets.mnist

start = time.time() ## 시작 시간 저장

(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, x_test) = (x_train / 255.0, x_test / 255.0)

# Add a channels dimension.
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

def create_train_dataset():
   print("==============================Processing Training DataSet==============================\n\n")
   train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(1, drop_remainder=True)
   train_ds = train_ds.map(lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))
   return train_ds.repeat()
###end of def:

def create_test_dataset():
   print("==============================Processing Test  DataSet==============================\n\n")
   test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(10000).batch(1, drop_remainder=True)
   test_ds = test_ds.map(lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))
   return test_ds.repeat()
###end of def:

# Create the model using the IPU-specific Sequential class instead of the
# standard tf.keras.Sequential class
def create_model():
   if len(ipus) > 0:
      model = ipu.keras.Sequential([
         keras.layers.Flatten(),
         keras.layers.Dense(128, activation='relu'),
         keras.layers.Dense(256, activation='relu'),
         keras.layers.Dense(128, activation='relu'),
         keras.layers.Dense(10, activation='softmax')])
    
      model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), 
                  optimizer = tf.keras.optimizers.Adam(), 
      #           experimental_steps_per_execution = 50, 
                  metrics=['sparse_categorical_accuracy'])
      return model   
   else: 
      model = tf.keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')])
    
      model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), 
                  optimizer = tf.keras.optimizers.Adam(), 
      #           experimental_steps_per_execution = 50, 
                  metrics=['sparse_categorical_accuracy'])
      return model
   ###end of if
###end of def:


def main():
   with strategy.scope():
      # Get the training dataset.
      print("==============================Getting Training DataSet==============================\n\n")
      ds1 = create_train_dataset()

      print("==============================Getting Test DataSet==============================\n\n")
      ds2 = create_test_dataset()

      # Create an instance of the model.
      print("==============================Building Model & Compile ==============================\n\n")
      model = create_model()
      
      print("==============================Model Training ==============================\n\n")
      model.fit(ds1, steps_per_epoch=2000, epochs=10)

      print("\n\n==============================Checking the result==============================\n\n")
      loss, accuracy = model.evaluate(ds2, steps=1000)

      print("Validation loss: {}".format(loss))

      print("Validation accuracy: {}%".format(100.0 * accuracy))

      print("\n\n==============================Finished Training by....==============================")
   ###end of with:
###end of def:

if __name__ == '__main__':
   main()
###end of if

print("Running Time :", round(time.time() - start, 2),"(Sec.)")  ## 현재시각 - 시작시간 = 실행 시간

#
###end of codes
#