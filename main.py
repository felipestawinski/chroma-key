import cv2 as cv
import numpy as np

def main():
    # Load the image with default color mode (BGR)
    img = cv.imread('img/1.bmp')
    if img is None:
        print("Error: Image not found.")
        return
    
    background = cv.imread('img/back.jpg')
    if background is None:
        print("Error: Image not found.")
        return
    
    # Show the original image
    cv.imshow('Original Image', img)
    cv.waitKey(0)
    
    # Convert BGR to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Try a wider range for green detection
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # Create a mask for green color
    mask = cv.inRange(hsv, lower_green, upper_green)
    cv.imshow('Green Mask', mask)
    cv.waitKey(0)
    
    mask_inv = cv.bitwise_not(mask)
    cv.imshow("Inverse green mask", mask_inv)
    cv.waitKey(0)

    mask_float = mask_inv.astype(np.float32) / 255.0
    blurred_mask = cv.GaussianBlur(mask_float, (0,0), 10)
    cv.imshow("Blurred Inverse Mask", blurred_mask)
    cv.waitKey(0)

    blurred_mask_3d = cv.merge([blurred_mask, blurred_mask, blurred_mask])
    foreground = img.astype(np.float32) * blurred_mask_3d
    foreground = foreground.astype(np.uint8)
    cv.imshow("Foreground", foreground)
    cv.waitKey(0)
    
    # Resize background to match foreground dimensions
    h, w = img.shape[:2]
    background_resized = cv.resize(background, (w, h))
    cv.imshow("Resized Background", background_resized)
    cv.waitKey(0)
    
    # Create inverse mask for background
    background_mask_3d = cv.merge([1.0 - blurred_mask, 1.0 - blurred_mask, 1.0 - blurred_mask])
    background_part = background_resized.astype(np.float32) * background_mask_3d
    background_part = background_part.astype(np.uint8)
    cv.imshow("Background Part", background_part)
    cv.waitKey(0)
    
    # Combine foreground and background
    result = cv.add(foreground, background_part)
    cv.imshow("Final Result", result)
    cv.waitKey(0)

    cv.destroyAllWindows()

    

if __name__ == "__main__":
    main()