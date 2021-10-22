import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19

import gc

def preprocess_image(image_path):
    width, height = keras.preprocessing.image.load_img(image_path).size
    img_nrows = 224
    img_ncols = 224

    # Util function to open, resize and format pictures into appropriate tensors
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

model = vgg19.VGG19(include_top=False)

# image = preprocess_image("/Users/divy/Desktop/Divy/Image Clasification(Biswa)/Startly/data/Beige_Soft_Lifestyle/erik-mclean-WBMjuGpbrCQ-unsplash.jpeg")

# x = model.predict(image)
# x = x[0]



def create_vgg_vectors(folder, to_folder):
    errors = []
    os.chdir("/Users/divy/Desktop/Divy/Image Clasification(Biswa)/Startly")
    if not os.path.exists(to_folder):
        os.mkdir(to_folder)
    for f in os.listdir(folder):
        if f == ".DS_Store" or f == "Icon\r":
            continue
        to_f = os.path.join(to_folder, f)
        f = os.path.join(folder, f)
        if not os.path.exists(to_f):
            os.mkdir(to_f)
        for e in os.listdir(f):
            if e == ".DS_Store" or e == "Icon\r":
                continue
            try:
                to_e = f"{os.path.join(to_f, e).split('.')[0]}.npy"
                if os.path.exists(to_e):
                    continue
                e = os.path.join(f, e)
                vector = model.predict(preprocess_image(e))[0]
                np.save(to_e, vector)
                del vector
                gc.collect()
            # if not os.path.exists(to_e):
            #     os.mkdir(to_e)
            # else:
            #     continue
            # try:
            #     style_reference_image = preprocess_image(e)
            #     features = feature_extractor(style_reference_image)
            #     for layer_name in style_layers:
            #         file = os.path.join(to_e, layer_name)
            #         if os.path.exists(file):
            #             continue
            #         layer_features = features[layer_name]
            #         G = gram_matrix(layer_features[0])
            #         file = open(file, "xb")
            #         pickle.dump(G, file)
            #         file.close()
            except Exception as ex:
                errors.append([ex, e])

    return errors

print(create_vgg_vectors("data", "feature_vectors_notop"))