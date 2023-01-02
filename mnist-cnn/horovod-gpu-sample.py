import tensorflow as tf
import horovod.tensorflow as hvd

# Initialize Horovod
hvd.init()


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create a model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
opt = tf.keras.optimizers.Adam(0.001 * hvd.size())
opt = hvd.DistributedOptimizer(opt)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'], experimental_run_tf_function=False)


# Train the model
model.fit(x_train, y_train, steps_per_epoch=500 // hvd.size(), epochs=50, batch_size=64, verbose=1 if hvd.rank() == 0 else 0)

# Evaluate the model
_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy:', accuracy)
