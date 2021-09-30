import glob
import os
import pickle
import shutil

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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


def image_style(image_path):
    image_name = os.path.basename(image_path).split(".")[0]
    path = os.path.dirname(os.path.dirname(image_path))
    folders = glob.glob(path + f"/**/{image_name}.jpeg", recursive=True)
    img_style = []
    for e in folders:
        img_style_e = os.path.basename(os.path.dirname(e))
        if img_style != "Premium_Black":
            img_style = list(set(img_style).union(set(os.path.basename(img_style_e).split("_"))))
        else:
            img_style.append(img_style_e)
    return img_style


def style_loss(style, combination, img_nrows, img_ncols):
    # S = gram_matrix(style)
    # C = gram_matrix(combination)
    S = style
    C = combination
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


image_path = None
while image_path != "exit":
    image_path = input()
    image_name = os.path.basename(image_path).split(".")[0]
    image_styles = image_style(image_path)
    orignal_layers = {}
    width, height = keras.preprocessing.image.load_img(image_path).size
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)
    style_reference_image = preprocess_image(image_path)
    features = feature_extractor(style_reference_image)
    for layer_name in style_layers:
        layer_features = features[layer_name]
        orignal_layers[layer_name] = gram_matrix(layer_features[0])

    gmatrix_folder = "data_G"
    comp_dict = {}
    for f in os.listdir(gmatrix_folder):
        if f == ".DS_Store":
            continue
        f = os.path.join(gmatrix_folder, f)
        for fo in os.listdir(f):
            if fo == ".DS_Store" or fo == image_name:
                continue
            fo = os.path.join(f, fo)
            for fn in os.listdir(fo):
                if fn == ".DS_Store":
                    continue
                fn = os.path.join(fo, fn)
                infile = open(fn, 'rb')
                compare = pickle.load(infile, encoding='bytes')
                layer_name = os.path.basename(fn)
                try:
                    comp_dict[layer_name].append(
                        [fn, style_loss(orignal_layers[layer_name], compare, img_nrows, img_ncols).numpy()])
                    # comp_dict[layer_name][1].append(style_loss(orignal_layers[layer_name], compare, img_nrows, img_ncols).numpy())
                except KeyError:
                    comp_dict[layer_name] = [
                        [fn, style_loss(orignal_layers[layer_name], compare, img_nrows, img_ncols).numpy()]]
    # average5
    average = {}
    for k in comp_dict.keys():
        if k == "block5_pool":
            continue
        for e in comp_dict[k]:
            try:
                average[e[0]] += e[1]
            except KeyError:
                average[e[0]] = e[1]

    comp_dict["average5"] = []
    for k in average.keys():
        comp_dict["average5"].append([k, average[k] / 5])

    # average6
    average = {}
    for k in comp_dict.keys():
        for e in comp_dict[k]:
            try:
                average[e[0]] += e[1]
            except KeyError:
                average[e[0]] = e[1]

    comp_dict["average6"] = []
    for k in average.keys():
        comp_dict["average6"].append([k, average[k] / 6])

    best_files = {}
    for k in comp_dict.keys():
        comp_dict[k] = sorted(comp_dict[k], key=lambda x: x[1])
        bf = []
        i = 0
        while i < 10:
            bf.append(comp_dict[k][i][0])
            i += 1
        best_files[k] = bf
    result_folder = "results"
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    result_folder = os.path.join(result_folder, image_name)
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    copy_name = "img"
    for sty in image_styles:
        copy_name += "_" + sty
    shutil.copy(image_path, result_folder)
    os.rename(os.path.join(result_folder, f"{image_name}.jpeg"), os.path.join(result_folder, f"{copy_name}.jpeg"))

    best_files_keys = list(best_files.keys())

    ##showing Results
    i = 0

    while i < len(best_files_keys):
        rows = 3
        columns = 4
        fig = plt.figure(figsize=(15, 8))
        fig.suptitle(best_files_keys[i])
        j = 1
        for e in best_files[best_files_keys[i]]:
            img_path = f"{os.path.dirname(e.replace('data_G', 'data'))}.jpeg"
            try:
                img = mpimg.imread(img_path)
                fig.add_subplot(rows, columns, j)
                plot_img_styles = image_style(img_path)

                plt_title = f"{j}"
                for e in image_styles:
                    for f in plot_img_styles:
                        if e == f:
                            plt_title = plt_title + "_" + e

                plt.imshow(img)
                plt.axis('off')
                plt.title(plt_title)
                # plt.show()
                j += 1
            except Exception as e:
                print(e)
                break
        plt.savefig(f"{os.path.join(result_folder, best_files_keys[i])}.jpeg")
        plt.close(fig)
        i += 1
