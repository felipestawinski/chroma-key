import cv2
import numpy as np


def compute_continuous_green_mask(fg_bgr: np.ndarray) -> np.ndarray:
    """
    Compute a continuous mask for green-screen removal based on green dominance.

    For each pixel:
      mask_raw = clip((G - (R + B) / 2) / (G + epsilon), 0, 1)
    Then smooth the mask to feather edges.
    """
    # Convert to float for computation
    fg = fg_bgr.astype(np.float32)
    B, G, R = cv2.split(fg)

    # Compute raw mask: degree to which green dominates over red and blue
    mask_raw = (G - (R + B) / 2) / (G + 1e-6)
    mask_raw = np.clip(mask_raw, 0.0, 1.0)

    # Feather edges by Gaussian blur
    # Kernel size chosen to balance smoothing and detail preservation
    mask_blurred = cv2.GaussianBlur(mask_raw, (21, 21), 0)
    return mask_blurred


def chroma_key(fg_bgr: np.ndarray, bg_bgr: np.ndarray) -> np.ndarray:
    """
    Composite foreground over background using a continuous mask.

    Args:
        fg_bgr: Foreground image with green background (H x W x 3)
        bg_bgr: Background replacement image (H x W x 3)

    Returns:
        Composite image (H x W x 3) as uint8
    """
    # Resize background to match foreground dimensions
    h, w = fg_bgr.shape[:2]
    bg_resized = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_AREA)

    # Compute continuous mask
    mask = compute_continuous_green_mask(fg_bgr)
    mask = mask[..., np.newaxis]  # expand dims for broadcasting

    # Composite: fg * (1-mask) + bg * mask
    composite = fg_bgr.astype(np.float32) * (1.0 - mask) + bg_resized.astype(np.float32) * mask
    return composite.astype(np.uint8)


if __name__ == '__main__':

    # Load images
    fg = cv2.imread('img/1.bmp', cv2.IMREAD_COLOR)
    bg = cv2.imread('img/back.jpg', cv2.IMREAD_COLOR)
    if fg is None:
        raise FileNotFoundError(f"Could not load foreground image")
    if bg is None:
        raise FileNotFoundError(f"Could not load background image")

    # Perform chroma key compositing
    result = chroma_key(fg, bg)
    cv2.imshow("Composited Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save and display
    # cv2.imwrite(args.output, result)
    # print(f"Saved composited image to {args.output}")
