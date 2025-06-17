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
    
    # set range for green detection
    lower_green = np.array([45, 50, 50])
    upper_green = np.array([75, 255, 255])

    # Create a mask for green color
    mask = cv.inRange(hsv, lower_green, upper_green)
    cv.imshow('Green Mask', mask)
    cv.waitKey(0)
    
    mask_inv = cv.bitwise_not(mask)
    cv.imshow("Inverse green mask", mask_inv)
    cv.waitKey(0)

    mask_float = mask_inv.astype(np.float32) / 255.0
    blurred_mask = cv.GaussianBlur(mask_float, (15,15), 0)
    #blurred_mask = np.clip(blurred_mask, 0.0, 1.0)
    cv.imshow("Blurred Inverse Mask", blurred_mask)
    cv.waitKey(0)

    # h, w = blurred_mask.shape
    # for i in range(h):
    #     for j in range(w):
    #         print(f"Blurred mask value at ({i}, {j}): {blurred_mask[i, j]}")
    
    
    blurred_mask_3d = cv.merge([blurred_mask, blurred_mask, blurred_mask])
    foreground = img.astype(np.float32) * blurred_mask_3d
    foreground = foreground.astype(np.uint8)
    cv.imshow("Foreground", foreground)
    cv.waitKey(0)
    
    # aqui
    foreground_limpo = foreground.copy()
    b, g, r = cv.split(foreground_limpo)
    condicao_despill = (g > r) & (g > b) & (b > 0)
    novo_g = ((r.astype(float) + b.astype(float)) / 2).astype(np.uint8)
    
    # Aplica a correção apenas nos pixels que satisfazem a condição
    g[condicao_despill] = novo_g[condicao_despill]
    foreground_limpo = cv.merge([b, g, r])
    
    cv.imshow("Foreground Cleaned", foreground_limpo)
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
    result = cv.add(foreground_limpo, background_part)
    cv.imshow("Final Result", result)
    cv.waitKey(0)

    cv.destroyAllWindows()

    

if __name__ == "__main__":
    main()