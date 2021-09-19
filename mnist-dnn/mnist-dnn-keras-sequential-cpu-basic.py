# Module Import
from tensorflow import keras
import time

# Load MNIST DataSet from Internet somewhere
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# INPU을 행렬방식으로 입력하기 위해 one-hot enconding 수행
# 5 --> 0 0 0 0 0 1 0 0 0 0
# 1 --> 0 1 0 0 0 0 0 0 0 0
y_train = keras.utils.to_categorical(y=y_train, num_classes = 10)
y_test = keras.utils.to_categorical(y=y_test, num_classes=10)

# Reshaping DataSet
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
print("Y Value & DataType:", y_train.shape, y_test.shape)
print("X Values & DataType", x_train.shape, x_test.shape)

start = time.time() # 시작 시간 저장

# 모델 구조(계층) 생성
model = keras.Sequential()
model.add(keras.layers.Dense(32, activation="sigmoid", input_shape=(28*28,)))
model.add(keras.layers.Dense(32, activation="sigmoid"))
model.add(keras.layers.Dense(10, activation="sigmoid"))

# 모델 컴파일
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1), loss="categorical_crossentropy", metrics=['accuracy'])

# 모델 구조 보여주기
model.summary()

# 모델 훈련
model.fit(x=x_train, y=y_train, batch_size=128, epochs=50, validation_data=(x_test, y_test))

# 모델 평가
model.evaluate(x_test, y_test )

print("Running Time :", round(time.time() - start, 2),"(Sec.)")  # 현재시각 - 시작시간 = 실행 시간

# 모델 저장
# 학습된 모델을 파일형태로 저장->load해 재-학습이나 추론에서 사용하게 함.
print("Save Model, name mnist_basic.h5...")
model.save("mnist_basic.h5")

print("Job Finished....")

#
###end of codes...
#
