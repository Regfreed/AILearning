from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.resnet import preprocess_input
import numpy as np


filename = "banana.jpg"

base_model = keras.applications.ResNet50(
    include_top = True,
    weights = "imagenet"
)

original = load_img(filename, target_size = (224, 224)) 
input_arr = img_to_array(original)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = base_model.predict(preprocess_input(input_arr))
P = decode_predictions(predictions)

print(P)