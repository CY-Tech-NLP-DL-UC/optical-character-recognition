#from TranslateHandwriting import translateHandwriting
from . import TranslateHandwriting as th
#from lettersDetection import lettersDetection, informaticLetter
from . import lettersDetection as ld
import tensorflow.compat.v1 as tf
import cv2
import numpy as np

def main(image_path):
    # exemple avec une image de test
    print("TYPE OF IMAGE PATH : ", type(image_path))
    print("PATH OF MY IMAGE : ", image_path)
    image = cv2.imread(image_path)
    image = np.array(image, dtype=np.uint8)
    words_tuple, phrases = ld.lettersDetection(image)
    words = [i[0] for i in words_tuple]

    letter = []
    for word in words :
        result = th.translateHandwriting(word)
        letter.append(result)
        tf.reset_default_graph()
    result = ld.informaticLetter(letter, words_tuple, phrases)
    print(result)
    return result


if __name__ == '__main__':
	main()
