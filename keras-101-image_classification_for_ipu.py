import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python import ipu
import os

num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()
        ###end of try:
        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)
        ###end of if:
    ###end of for:
###end of for:

print("Deleted %d images" % num_skipped)
image_size = (180, 180)
batch_size = 2

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    # validation_split=0.2,
    # subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages/validation_set",
    # validation_split=0.2,
    # subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomContrast((0.1, 0.2)),
    ]
)

train_ds = train_ds.unbatch().repeat().batch(batch_size=batch_size, drop_remainder=True)
train_ds = train_ds.prefetch(buffer_size=32)

val_ds = val_ds.unbatch().repeat().batch(batch_size=batch_size, drop_remainder=True)
val_ds = val_ds.prefetch(buffer_size=32)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    with ipu.keras.PipelineStage(0):
        x = data_augmentation(inputs)
        x = layers.experimental.preprocessing.Rescaling(1.0/255)(x)
        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        previous_block_activation = x  # Set aside residual
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    with ipu.keras.PipelineStage(1):
    # Project residual
        residual = layers.Conv2D(128, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(256, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(256, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    # Project residual
        residual = layers.Conv2D(256, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(512, 3, padding="same")(x)
    with ipu.keras.PipelineStage(2):
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(512, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        # Project residual
        residual = layers.Conv2D(512, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(728, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    with ipu.keras.PipelineStage(3):
        x = layers.SeparableConv2D(728, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    # Project residual
        residual = layers.Conv2D(728, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        if num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = num_classes
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("./batch_2_epoch_50/save_at_{epoch}.h5"),
]

def main():
    config = ipu.config.IPUConfig()
    config.auto_select_ipus = 4
    config.configure_ipu_system()
    strategy = ipu.ipu_strategy.IPUStrategy()

    with strategy.scope():
        model = make_model(input_shape=image_size + (3,), num_classes=2)
        model.print_pipeline_stage_assignment_summary()
        model.set_pipelining_options(gradient_accumulation_steps_per_replica=8)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
            steps_per_execution=2880,
            # steps_per_execution은 gradient_accumulation_steps_per_replica * num of replicas 의 배수이어야 함
            # steps_per_epoch은 (= data size // (batch_size * num of replicas)) 보다 작아야 함
        )

        model.fit(
            train_ds,
            epochs=epochs,
            callbacks=callbacks,                  
            validation_data=val_ds,
            steps_per_epoch=11520,
            # steps_per_epoch은 (= steps_per_execution * num of replicas) 
        )
        ###end of with:
    ###end of if:
###end of def:
if __name__ == "__main__":
    main()