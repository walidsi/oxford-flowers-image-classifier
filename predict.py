import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import json

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# TODO: Create the process_image function
def process_image(image: np.ndarray) -> np.ndarray:
    image_tensor = tf.convert_to_tensor(image)
    image_resized = tf.image.resize(image_tensor, (224, 224))
    image_resized /= 255

    return image_resized.numpy()


# TODO: Create the predict function
def predict(image_path, model, top_k=5):
    im = Image.open(image_path)
    image_arr = np.asarray(im)

    processed_image = process_image(image_arr)

    processed_image = np.expand_dims(processed_image, 0)

    ps = model.predict(processed_image)

    #print(ps[0])

    top_k_indices = np.argsort(ps[0])[-top_k:]
    top_k_indices = top_k_indices[::-1]
    #print(top_k_indices)

    top_ps = ps[0][top_k_indices]

    # classes are zero based, add 1
    top_classes = []
    for index in top_k_indices:
        top_classes.append(str(int(index) + 1))

    #print(top_classes)

    return top_ps, top_classes


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('file_path', type=str)
    parser.add_argument('saved_model', type=str)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_names', type=str, default='label_map.json')
    # Parse the argument
    args = parser.parse_args()

    model = tf.keras.models.load_model(
        args.saved_model,
        custom_objects={'KerasLayer': hub.KerasLayer},
        compile=False)

    ps, classes = predict(args.file_path, model)

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)

    classes_names_list = []
    for key in classes:
        classes_names_list.append(class_names[key])

    print(classes_names_list)
    print(ps)
