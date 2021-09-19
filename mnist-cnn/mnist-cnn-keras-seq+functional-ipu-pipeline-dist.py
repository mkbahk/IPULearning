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

    return x_train, y_train, x_test, y_test
### end of def:

def sequential_model_fn():
    # 단순한 합성곱 네트워크 만들기
    stages = [
        # Pipeline stage 0.
        [
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
        ],
        # Pipeline stage 1.
        [
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ]
    ]
    return ipu.keras.SequentialPipelineModel(stages, gradient_accumulation_count=8)
### end of def:

def functional_model_fn():
    # 단순한 합성곱 네트워크 만들기
    # 두개의 파이프라인 스테이지로 샤딩합니다.
    input_layer = keras.Input(shape=input_shape)

    with ipu.keras.PipelineStage(0):
        x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    with ipu.keras.PipelineStage(1):
        x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    with ipu.keras.PipelineStage(2):
        x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    with ipu.keras.PipelineStage(3):
        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        output_layer = layers.Dense(num_classes, activation='softmax')(x)
    ### end of with:

    return ipu.keras.PipelineModel(input_layer, output_layer, gradient_accumulation_count=8)
### end of def:

def train_model(model):
    # 훈련용 하이퍼파라메터
    batch_size = 128
    epochs = 100

    # 데이타 얻기
    x_train, y_train, x_test, y_test = data_fn()

    # 훈련을 위해서 모델을 컴파일
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 모델을 훈련하기
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)

    # 훈련된 모델을 평가하기
    eval_out = model.evaluate(x=x_test, y=y_test, batch_size=batch_size)
    print("Evaluation Loss: %f Evaluation Accuracy: %f" % tuple(eval_out))
### end of def:

if __name__ == '__main__':
    # IPU System 설정
    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.auto_select_ipus(cfg, 4)
    #cfg = ipu.utils.select_ipus(config, indices=[8])
    #cfg = ipu.utils.select_ipus(config, indices=[0, 1, 2, 3])
    ipu.utils.configure_ipu_system(cfg)

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
        #sequential model 훈련하기
        print("\n\nTraining a Sequential MNIST Model.")
        start = time.time()
        train_model(sequential_model_fn())
        print("Running Time :", round(time.time() - start, 2),"(Sec.)")
        
        #funcaitonal model 훈련하기
        print("\n\nTraining a Function MNIST Model.")
        start = time.time()
        train_model(functional_model_fn())
        print("Running Time :", round(time.time() - start, 2),"(Sec.)")
    ### end of with:
### end of if:

""" 
end of codes 
"""


