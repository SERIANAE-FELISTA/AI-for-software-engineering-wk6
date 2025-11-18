import tensorflow as tf
from tf.keras import layers, models  
import tensorflow_datasets as tfds

# Check TF version
print(f"TensorFlow version: {tf.__version__}")

try:
    # Load dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'tf_flowers',
        split=['train[:80%]', 'train[80%:]'],
        with_info=True,
        as_supervised=True,
    )

    IMG_SIZE = 128
    BATCH_SIZE = 32

    # Preprocessing
    def preprocess(image, label):
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0
        return image, label

    ds_train = ds_train.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Build model
    model = models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(ds_info.features['label'].num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    history = model.fit(ds_train, validation_data=ds_test, epochs=5)

    # Evaluate
    loss, acc = model.evaluate(ds_test)
    print(f"Test Accuracy: {acc:.4f}")

except Exception as e:
    print(f"Error occurred: {e}")
    # Add more debugging: print(tf.config.list_physical_devices()
