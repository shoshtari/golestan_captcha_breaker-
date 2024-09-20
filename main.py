import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.optimizers import Adam

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
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, strides=2, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, 3, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(62, activation="softmax"),
    ]
)
model.compile(
    optimizer=Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, decay=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.fit(ds_train, validation_data=ds_test, epochs=50)
model.evaluate(ds_test)
