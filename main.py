import cv2 as cv
import numpy as np

def main():
    img = cv.imread('img/1.bmp', cv.COLOR_BGR2HSV)
    if img is None:
        print("Error: Image not found.")
        return
    cv.imshow('Original Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    blurred = cv.GaussianBlur(img, (0,0), 2)
    cv.imshow('Blurred Image', blurred)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()