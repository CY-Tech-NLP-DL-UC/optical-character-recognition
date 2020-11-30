from TranslateHandwriting import translateHandwriting
from lettersDetection import lettersDetection, informaticLetter
import tensorflow.compat.v1 as tf
import cv2
import numpy as np

def main():
    # exemple avec une image de test
    image = cv2.imread('../data/letters.png')
    image = np.array(image, dtype=np.uint8)
    words_tuple, phrases = lettersDetection(image)
    words = [i[0] for i in words_tuple]

    letter = []
    for word in words :
        result = translateHandwriting(word)
        letter.append(result)
        tf.reset_default_graph()
    print(informaticLetter(letter, words_tuple, phrases))


if __name__ == '__main__':
	main()
