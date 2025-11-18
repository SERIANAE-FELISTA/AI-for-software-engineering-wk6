import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 128

interpreter = tf.lite.Interpreter(model_path="recycle_classifier.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(image_path):
    img = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    result = interpreter.get_tensor(output_details[0]['index'])
    print("Prediction (0=non recyclable, 1=recyclable):", result)

predict("sample.jpg")
