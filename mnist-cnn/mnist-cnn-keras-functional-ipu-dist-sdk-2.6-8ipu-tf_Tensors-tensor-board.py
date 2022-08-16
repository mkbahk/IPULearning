"""
Keras MNIST 예제를 기반으로 작성되었습니다.
https://keras.io/examples/vision/mnist_convnet/
작성자: 박문기(mkbahk@m e g a zone.com)
"""
# 파이썬 모듈 임포트
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import time
import os
from tensorflow.python import ipu


# 모델/데이타 하이퍼파라메터
num_classes = 10
input_shape = (28, 28, 1)


def data_fn():
    # 데이타를 훈련과 검증용으로 분리
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # [0, 1] 범위로 이미지들 조정
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # 이미지들이 (28, 28, 1) 형태로 변형하기
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # 클래스 벡터를 이진 클래스 메트릭스들로 변환
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return tf.convert_to_tensor(x_train), tf.convert_to_tensor(y_train), tf.convert_to_tensor(x_test), tf.convert_to_tensor(y_test)
#def

def functional_model_fn():
    # 단순한 합성곱 네트워크 만들기
    input_layer = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    output_layer = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(input_layer, output_layer)
#def

def train_model(model):
    # 훈련용 하이퍼파라메터
    batch_size = 600
    epochs = 10000
 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='/tmp/tblogs',
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq='epoch',
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None
    ) #tensorboard --logdir="/tmp/tblogs --bind-all"
    # 데이타 얻기
    x_train, y_train, x_test, y_test = data_fn()

    # 훈련을 위해서 모델을 컴파일
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], steps_per_execution=12)

    # 모델을 훈련하기
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard_callback])

    # 훈련된 모델을 평가하기
    eval_out = model.evaluate(x=x_test, y=y_test, batch_size=100)
    print("Evaluation Loss: %f Evaluation Accuracy: %f" % tuple(eval_out))
#def

if __name__ == '__main__':
    # IPU System 설정
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 8
    config.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        # funcaitonal model 훈련하기
        print("\n\nTraining a Function MNIST Model.")
        start = time.time()
        train_model(functional_model_fn())
        print("Running Time :", round(time.time() - start, 2),"(Sec.)")
    #with
#if

""" 
end of codes 
"""
