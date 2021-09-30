import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19


def preprocess_image(image_path):
    width, height = keras.preprocessing.image.load_img(image_path).size
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)

    # Util function to open, resize and format pictures into appropriate tensors
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


# Build a VGG19 model loaded with pre-trained ImageNet weights
model = vgg19.VGG19(weights="imagenet", include_top=False)

# Get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1',
                'block5_pool']

style_output_dict = {}
for s in style_layers:
    style_output_dict[s] = outputs_dict[s]

feature_extractor = keras.Model(inputs=model.inputs, outputs=style_output_dict)

# GRAM matrix extractor
folder = "data"
to_folder = "data_G"
errors = []
if not os.path.exists(to_folder):
    os.mkdir(to_folder)
for f in os.listdir(folder):
    if f == ".DS_Store":
        continue
    to_f = os.path.join(to_folder, f)
    f = os.path.join(folder, f)
    if not os.path.exists(to_f):
        os.mkdir(to_f)
    l = len(os.listdir(f))
    for e in os.listdir(f):
        if e == ".DS_Store":
            continue
        to_e = os.path.join(to_f, e).split(".")[0]
        e = os.path.join(f, e)
        if not os.path.exists(to_e):
            os.mkdir(to_e)
        print(l)
        l -= 1
        try:
            style_reference_image = preprocess_image(e)
            features = feature_extractor(style_reference_image)
            # input_tensor = tf.concat(
            #     [style_reference_image, style_reference_image], axis=0
            # )
            for layer_name in style_layers:
                file = os.path.join(to_e, layer_name)
                if os.path.exists(file):
                    continue
                layer_features = features[layer_name]
                G = gram_matrix(layer_features[0])
                file = open(file, "xb")
                pickle.dump(G, file)
                file.close()
        except Exception as ex:
            errors.append([ex, e])
