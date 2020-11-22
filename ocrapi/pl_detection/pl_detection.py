import functions_pl_detection as fct

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
dir_path = os.path.dirname(os.path.realpath(__file__))

def pl_detection(image):
  wpod_net = fct.load_model(os.path.join(dir_path, 'pl_detection'))
  image = fct.color2gray(image)
  imgPL = fct.get_plate(image, wpod_net, Dmax=608, Dmin=256)
  return imgPL[0]

if __name__ == '__main__':
    path_folder = os.path.join(dir_path, "..", "..", "license_plates_detection", "Photos")
    images = [f for f in os.listdir(path_folder)]
    for image in images:
        input = mpimg.imread(os.path.join(path_folder, image))
        output = pl_detection(input)
        plt.imshow(input)
        plt.show()
        plt.imshow(output)
        plt.show()
