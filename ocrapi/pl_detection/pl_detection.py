from . import functions_pl_detection as fct

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
dir_path = os.path.dirname(os.path.realpath(__file__))

def main(path):
	image = mpimg.imread(path)
	wpod_net = fct.load_model(os.path.join(dir_path, 'pl_detection'))
	image = fct.color2gray(image)
	imgPL = fct.get_plate(image, wpod_net, Dmax=608, Dmin=256)
	return imgPL[0]
