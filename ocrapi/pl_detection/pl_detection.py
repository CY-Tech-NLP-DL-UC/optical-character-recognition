from . import functions_pl_detection as fct

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
dir_path = os.path.dirname(os.path.realpath(__file__))


debug = True

def main(path):
	image = mpimg.imread(path)
	wpod_net = fct.load_model(os.path.join(dir_path, 'pl_detection'))
	if debug : print("model ok")
	image = fct.color2gray(image)
	if debug : print("image gris ok")
	imgPL = fct.get_plate(image, wpod_net, Dmax=608, Dmin=256)
	if debug :
		print("get_plate ok")
		print("#"*20, "FINI")
	return imgPL[0]

if __name__=="__main__":
	main(os.path.join(dir_path,"..","..","license_plates_detection", "Photos", "image_test.jpg"))