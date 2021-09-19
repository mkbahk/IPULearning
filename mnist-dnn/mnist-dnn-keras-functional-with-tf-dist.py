# Module Import
import tensorflow as tf
from tensorflow import keras

# Load MNIST DataSet
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# one-hot enconding 수행
# 5 --> 0 0 0 0 0 1 0 0 0 0
# 1 --> 0 1 0 0 0 0 0 0 0 0
y_train = keras.utils.to_categorical(y=y_train, num_classes = 10)
y_test = keras.utils.to_categorical(y=y_test, num_classes=10)

# Reshaping DataSet
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
print(y_train.shape, y_test.shape)
print(x_train.shape, x_test.shape)

strategy = tf.distribute.get_strategy() ##만약 GPU나 cpu가 있으면 코어숫자만큼 분산전략수행
with strategy.scope():
   #모델생성
   model = keras.Sequential([
        keras.layers.Dense(32, activation='sigmoid', input_shape=(28*28,)),
        keras.layers.Dense(32, activation='sigmoid'),
        keras.layers.Dense(10, activation='sigmoid')])

   #모델컴파일
   model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1), loss="categorical_crossentropy", metrics=['accuracy'])
   model.summary()

   #모델훈련
   model.fit(x=x_train, y=y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

   #모델 평가
   model.evaluate(x_test, y_test)
###end of with:

#
###end of codes
#
