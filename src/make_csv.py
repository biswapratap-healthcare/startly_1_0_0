import os
import random

os.chdir("/Users/divy/Desktop/Divy/Image Clasification(Biswa)/Startly")

csv_folder = "csvs"

feature_vectors = "feature_vectors_notop"

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1',
                'block5_pool',
                'average5',
                'average6']

csv = ["file_path"] + style_layers
csv = ",".join(csv) + "\n"

for e in os.listdir(feature_vectors):
    if e == ".DS_Store" or e == "Icon\r":
        continue
    e = os.path.join(feature_vectors, e)
    for f in os.listdir(e):
        if f == ".DS_Store" or f == "Icon\r":
            continue
        f = os.path.join(e, f)
        temp_list = [f]
        for s in style_layers:
            value = int(random.gauss(0, 5 / 3))
            if value > 5:
                value = 5
            elif value < -5:
                value = -5
            temp_list.append(str(value))
        #     temp_list.append(str(random.uniform(-5.0,5.0)))

        csv += ",".join(temp_list) + "\n"

file = open(os.path.join(csv_folder,"random_data_notop.csv"),"x",encoding='UTF8', newline='')

file.write(csv)
file.close()
