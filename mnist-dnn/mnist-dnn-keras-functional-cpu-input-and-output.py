import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import time, os

start = time.time() # 시작 시간 저장

# CPU분산전략 정의
strategy = tf.distribute.get_strategy()

# The input data and labels.
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, x_test) = (x_train / 255.0, x_test / 255.0)

# Add a channels dimension.
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

def create_train_dataset():
    print("==============================Processing Training DataSet==============================\n\n")
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(32, drop_remainder=True)
    train_ds = train_ds.map(lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))
    return train_ds.repeat()
###end of def:

def create_test_dataset():
    print("==============================Processing Test  DataSet==============================\n\n")
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(10000).batch(32, drop_remainder=True)
    test_ds = test_ds.map(lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))
    return test_ds.repeat()
###end of def:

def create_model():
    
    inputs = tf.keras.Input(shape = (28, 28))
    
    flatten_layer = keras.layers.Lambda(lambda ipt: K.reshape(ipt, (-1, 28 * 28)))
    flatten_inputs = flatten_layer(inputs)

    x = keras.layers.Flatten()(flatten_inputs) 
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)

    outputs = keras.layers.Dense(10, activation='softmax')(x)

    # Defined the model.
    model = tf.keras.Model(inputs, outputs, name="dnn")
    return model
###end of def:

def main():
    # Get the training dataset.
    print("==============================Getting Training DataSet==============================\n\n")
    ds1 = create_train_dataset()
    print("==============================Getting Test DataSet==============================\n\n")
    ds2 = create_test_dataset()

    with strategy.scope():
        # Create an instance of the model.
        print("==============================Building Model==============================\n\n")
        model = create_model()

        model.summary()

        print("==============================Building Compile==============================\n\n")
        model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer = tf.keras.optimizers.Adam(),
                      metrics=['sparse_categorical_accuracy'])

        print("==============================Model Training ==============================\n\n")
        model.fit(ds1, steps_per_epoch=20, epochs=50)

        print("\n\n==============================Checking the result==============================\n\n")
        (loss, accuracy) = model.evaluate(ds2, steps=1000)
        print("Validation loss: {}".format(loss))
        print("Validation accuracy: {}%".format(100.0 * accuracy))
        print("\n\n==============================Job Done==============================")
    ###end of with:
###end of def:

if __name__ == '__main__':
    main()
###end of if

print("Total Execution Time :", time.time() - start,"(Sec)")  # 현재시각 - 시작시간 = 실행 시간

#
### end of Codes...
#


