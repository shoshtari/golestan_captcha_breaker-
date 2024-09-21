import os

import tensorflow as tf
import tensorflow_datasets as tfds

# from tensorflow.keras.optimizers import Adam

tf.get_logger().setLevel("ERROR")

BATCH_SIZE = 64
(ds_test, ds_train), ds_info = tfds.load(
    "emnist",
    split=["test", "train"],
    with_info=True,
    shuffle_files=True,
    as_supervised=True,
)


def normalize_image(image, label):
    image = tf.image.rot90(image, k=3)
    image = tf.image.flip_left_right(image)
    return tf.cast(image, tf.float32) / 255.0, label


def improve_ds(ds, is_train=False):
    ds = ds.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        ds = ds.cache().shuffle(1000)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds


ds_train = improve_ds(ds_train, is_train=True)
ds_test = improve_ds(ds_test)


model = tf.keras.models.Sequential(
    [
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(16, 5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Conv2D(32, 3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Conv2D(64, 3, strides=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Conv2D(128, 3, strides=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dense(92),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dense(62, activation="softmax"),
    ]
)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

checkpoint_dir = "./checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "model.weights.h5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, save_freq="epoch"
)
if os.path.exists(checkpoint_path):
    try:
        model.load_weights(checkpoint_path)
    except ValueError:
        os.remove(checkpoint_path)


model.fit(ds_train, validation_data=ds_test, epochs=50, callbacks=[checkpoint_callback])
model.evaluate(ds_test)
