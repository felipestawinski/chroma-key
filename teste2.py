import cv2
import numpy as np

def chroma_key(fg_img_path, bg_img_path, output_path=None):
    # Load images
    foreground = cv2.imread(fg_img_path)
    background = cv2.imread(bg_img_path)

    # Resize background to match foreground
    bg_resized = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))

    # Convert foreground image to float for processing
    fg_float = foreground.astype(np.float32) / 255.0

    # Convert to HSV for better color masking
    hsv = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV)

    # Define the green color range for chroma keying
    # Lower and upper green can be broad to accommodate different images
    lower_green = np.array([35, 80, 40])    # hue, sat, val
    upper_green = np.array([85, 255, 255])

    # Create initial green mask
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # --- Step 1: Soft/Continuous Mask based on green "amount" ---

    # Extract hue, saturation, value channels
    h, s, v = cv2.split(hsv)

    # Calculate green-ness as a continuous mask:
    # 1. Calculate how close each pixel's hue is to the green hue (60)
    hue_dist = np.abs(h.astype(np.float32) - 60)
    hue_dist = np.minimum(hue_dist, 180 - hue_dist)  # Hue is circular

    # 2. Normalize: 0 (green) to 1 (not green)
    # Hue tolerance: 35 to 85 is green-ish. Let's map 35 to 0, 85 to 1.
    hue_mask = np.clip((hue_dist - 0) / (25), 0, 1) # 25 for softness

    # 3. Higher saturation => more chroma, so use as multiplier
    sat_mask = s.astype(np.float32) / 255.0

    # 4. "Green probability" = inverse of hue_mask, times saturation
    green_prob = (1.0 - hue_mask) * sat_mask

    # 5. Make a continuous alpha: 1 = foreground (not green), 0 = background (fully green)
    alpha = 1.0 - green_prob

    # Optional: sharpen the transition at high green-probability, but keep softness at edges
    alpha = np.clip(alpha * 1.2 - 0.1, 0, 1)  # adjust values for softness

    # --- Step 2: Clean mask for smoother edges ---

    # Slight blur for anti-aliasing/soft boundary (optional but recommended)
    alpha = cv2.GaussianBlur(alpha, (7,7), 0)

    # Make alpha 3-channel
    alpha_3c = np.dstack([alpha]*3)

    # --- Step 3: Compose final image ---
    bg_float = bg_resized.astype(np.float32) / 255.0

    # Composite: output = fg * alpha + bg * (1 - alpha)
    out = fg_float * alpha_3c + bg_float * (1 - alpha_3c)
    out = (out * 255).astype(np.uint8)

    # Optionally save result
    if output_path:
        cv2.imwrite(output_path, out)

    return out, alpha

# Example usage
if __name__ == "__main__":
    fg_path = "img/1.bmp"  # foreground with green background
    bg_path = "img/back.jpg"       # new scene
    output_path = "img/output_scene.jpg"

    result, alpha = chroma_key(fg_path, bg_path, output_path)

    # Show the result
    cv2.imshow("Composited Scene", result)
    cv2.imshow("Alpha Mask", (alpha*255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
