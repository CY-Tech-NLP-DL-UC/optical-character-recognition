from TranslateHandwriting import translateHandwriting
import cv2

def main():
    # exemple avec une image de test
    image = cv2.imread('../data/test.png', cv2.IMREAD_GRAYSCALE)
    result = translateHandwriting(image)

if __name__ == '__main__':
	main()
