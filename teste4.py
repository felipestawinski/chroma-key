import cv2 as cv
import numpy as np
import argparse
import os

def create_parser():
    parser = argparse.ArgumentParser(description='Chroma Key Processing')
    parser.add_argument('--input', type=str, default='img/1.bmp', help='Path to input image with green background')
    parser.add_argument('--background', type=str, default='img/back.jpg', help='Path to background image')
    parser.add_argument('--output', type=str, help='Path to save output image')
    parser.add_argument('--show', action='store_true', help='Show intermediate results')
    return parser

def calculate_green_dominance(img):
    """Calculate how dominant green is compared to other channels in each pixel"""
    b, g, r = cv.split(img)
    
    # Green dominance over red and blue
    dominance_r = np.clip((g.astype(np.float32) - r.astype(np.float32)) / 255.0, 0, 1)
    dominance_b = np.clip((g.astype(np.float32) - b.astype(np.float32)) / 255.0, 0, 1)
    
    # Combine dominance scores (pixel is green if it's more green than both red and blue)
    green_score = dominance_r * dominance_b
    
    return green_score

def create_soft_mask(img, hsv_lower=(45, 30, 30), hsv_upper=(85, 255, 255), 
                     expansion_kernel_size=5, blur_amount=15):
    """Create a soft mask that captures the transition regions between foreground and background"""
    
    # HSV approach for primary segmentation
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_green = np.array(hsv_lower)
    upper_green = np.array(hsv_upper)
    
    # Create base mask
    mask = cv.inRange(hsv, lower_green, upper_green)
    
    # Calculate green dominance as a continuous value
    green_dominance = calculate_green_dominance(img)
    
    # Combine binary mask with green dominance for transition handling
    combined_mask = np.maximum(mask / 255.0, green_dominance * 0.8) 
    
    # Ensure the mask isn't too aggressive (preserve semi-transparent areas)
    kernel = np.ones((expansion_kernel_size, expansion_kernel_size), np.uint8)
    dilated_mask = cv.dilate(mask, kernel)
    transition_zone = dilated_mask - mask
    
    # Add more granularity to transition areas
    combined_mask = np.where(
        transition_zone > 0,
        np.minimum(combined_mask * 1.2, 1.0),  # Enhance transition areas slightly
        combined_mask
    )
    
    # Invert the mask (1 for foreground, 0 for background/green)
    inverted_mask = 1.0 - combined_mask
    
    # Smooth the transitions with a large Gaussian blur
    blurred_mask = cv.GaussianBlur(inverted_mask, (blur_amount, blur_amount), 0)
    
    # Ensure values stay in valid range
    blurred_mask = np.clip(blurred_mask, 0.0, 1.0)
    
    return blurred_mask

def apply_despill(img, mask, strength=0.5):
    """Remove green color spill from edges"""
    # Split the channels
    b, g, r = cv.split(img.astype(np.float32))
    
    # Use red and blue to estimate what green should be
    natural_green = (r + b) / 2.0
    
    # Detect where green is unnaturally high
    green_excess = np.maximum(g - natural_green, 0)
    
    # Scale the effect by the inverse mask (more effect near green edges)
    # Make sure mask has same shape as the color channels
    mask_resized = cv.resize(mask, (g.shape[1], g.shape[0]))
    inv_mask = 1.0 - mask_resized
    
    correction = green_excess * inv_mask * strength
    
    # Apply the correction
    g_corrected = g - correction
    
    # Ensure all channels have the same shape and type
    b = b.astype(np.float32)
    g_corrected = g_corrected.astype(np.float32)
    r = r.astype(np.float32)
    
    # Merge channels back
    despilled = cv.merge([b, g_corrected, r])
    
    return despilled

def enhance_foreground(img, strength=0.1):
    """Enhance the foreground to make it pop against the new background"""
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b_channel = cv.split(lab)
    
    # Enhance lightness slightly
    l = np.clip(l * (1 + strength), 0, 255).astype(np.uint8)
    
    # Enhance color slightly
    a = np.clip(a * (1 + strength), 0, 255).astype(np.uint8)
    b_channel = np.clip(b_channel * (1 + strength), 0, 255).astype(np.uint8)
    
    enhanced_lab = cv.merge([l, a, b_channel])
    enhanced_img = cv.cvtColor(enhanced_lab, cv.COLOR_LAB2BGR)
    
    return enhanced_img

def match_histogram(source, reference):
    """Match histogram of source image to reference image"""
    # Convert to LAB color space
    source_lab = cv.cvtColor(source, cv.COLOR_BGR2LAB)
    reference_lab = cv.cvtColor(reference, cv.COLOR_BGR2LAB)
    
    # Split channels
    source_l, source_a, source_b = cv.split(source_lab)
    reference_l, reference_a, reference_b = cv.split(reference_lab)
    
    # Only match the L channel (brightness)
    source_hist = cv.calcHist([source_l], [0], None, [256], [0, 256])
    reference_hist = cv.calcHist([reference_l], [0], None, [256], [0, 256])
    
    # Calculate CDFs
    source_cdf = source_hist.cumsum() / source_hist.sum()
    reference_cdf = reference_hist.cumsum() / reference_hist.sum()
    
    # Create lookup table
    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        j = 255
        while j > 0 and source_cdf[i] <= reference_cdf[j]:
            j -= 1
        # Ensure j is never negative
        lookup_table[i] = max(0, j)
    
    # Apply lookup table
    matched_l = cv.LUT(source_l, lookup_table)
    
    # Ensure all channels have the same type
    matched_l = matched_l.astype(source_a.dtype)
    
    # Merge channels and convert back
    matched_lab = cv.merge([matched_l, source_a, source_b])
    matched_bgr = cv.cvtColor(matched_lab, cv.COLOR_LAB2BGR)
    
    return matched_bgr

def process_chroma_key(input_path, background_path, show_steps=False):
    """Process an image with chroma key effect"""
    # Load images
    img = cv.imread(input_path)
    if img is None:
        print(f"Error: Could not load image from {input_path}")
        return None
        
    background = cv.imread(background_path)
    if background is None:
        print(f"Error: Could not load background from {background_path}")
        return None
    
    # Step 1: Create a soft mask
    mask = create_soft_mask(img)
    mask_3d = cv.merge([mask, mask, mask])
    
    if show_steps:
        cv.imshow("Original Image", img)
        cv.imshow("Soft Mask", mask)
        cv.waitKey(0)
    
    # Step 2: Apply green spill reduction
    despilled = apply_despill(img, mask)
    
    if show_steps:
        cv.imshow("After Despill", despilled.astype(np.uint8))
        cv.waitKey(0)
    
    # Step 3: Extract foreground
    foreground = despilled * mask_3d
    
    if show_steps:
        cv.imshow("Extracted Foreground", foreground.astype(np.uint8))
        cv.waitKey(0)
    
    # Step 4: Enhance foreground slightly
    enhanced_foreground = enhance_foreground(foreground.astype(np.uint8))
    
    if show_steps:
        cv.imshow("Enhanced Foreground", enhanced_foreground)
        cv.waitKey(0)
    
    # Step 5: Resize background to match foreground
    h, w = img.shape[:2]
    background_resized = cv.resize(background, (w, h))
    
    # Optional: Match histogram between foreground and background for better blending
    matched_foreground = match_histogram(enhanced_foreground, background_resized)
    
    if show_steps:
        cv.imshow("Resized Background", background_resized)
        cv.imshow("Histogram Matched Foreground", matched_foreground)
        cv.waitKey(0)
    
    # Step 6: Apply background through inverse mask
    background_mask_3d = 1.0 - mask_3d
    background_part = background_resized.astype(np.float32) * background_mask_3d
    
    if show_steps:
        cv.imshow("Background Part", background_part.astype(np.uint8))
        cv.waitKey(0)
    
    # Step 7: Combine foreground and background
    result = cv.add(matched_foreground.astype(np.float32) * mask_3d, background_part)
    
    return result.astype(np.uint8)

def main():
    # parser = create_parser()
    # args = parser.parse_args()

    result = process_chroma_key('img/1.bmp', 'img/back.jpg', show_steps=False)

    if result is not None:
        # Show result
        cv.imshow("Final Result", result)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        # Save result if output path is specified
        # if args.output:
        #     cv.imwrite(args.output, result)
        #     print(f"Result saved to {args.output}")

if __name__ == "__main__":
    main()