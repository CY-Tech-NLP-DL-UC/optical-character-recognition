from TranslateHandwriting import translateHandwriting
from lettersDetection import lettersDetection
import tensorflow.compat.v1 as tf
import cv2
import numpy as np

def main():
    # exemple avec une image de test
    image = cv2.imread('../data/letters.png')
    image = np.array(image, dtype=np.uint8)
    words = lettersDetection(image)

    letter = []
    for word in words :
        result = translateHandwriting(word)
        letter.append(result)
        tf.reset_default_graph()

    # words are stored in "letter"


if __name__ == '__main__':
	main()
