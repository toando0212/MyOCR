from PIL import Image
import numpy as np
import cv2
import logging
import os
from doctr.io import DocumentFile

logger = logging.getLogger(__name__)

def correct_perspective(pil_img: Image.Image) -> Image.Image:
    """
    Corrects perspective distortion in an image of a document.
    Finds the largest quadrilateral in the image and warps it to a top-down view.
    """
    try:
        img = np.array(pil_img.convert("RGB"))
        orig = img.copy()
        
        proc_height = 500.0
        if img.shape[0] <= proc_height:
             ratio = 1.0
             proc_img = img.copy()
        else:
             ratio = img.shape[0] / proc_height
             proc_img = cv2.resize(img, (int(img.shape[1] / ratio), int(proc_height)))

        gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        screenCnt = None
        img_area = proc_img.shape[0] * proc_img.shape[1]

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                if cv2.contourArea(approx) > img_area * 0.20:
                    screenCnt = approx
                    break
                else:
                    logger.info("Found a 4-point contour, but its area is too small. Skipping it.")
        
        if screenCnt is None:
            logger.warning("Could not find a suitable document contour. Skipping perspective correction.")
            return pil_img
        
        logger.info("Found document contour, applying perspective correction.")

        pts = screenCnt.reshape(4, 2) * ratio
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        if maxWidth <= 0 or maxHeight <= 0:
            logger.warning("Calculated invalid dimensions for warped image. Skipping perspective correction.")
            return pil_img

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
        return Image.fromarray(warped)
    except Exception as e:
        logger.error(f"Error during perspective correction: {e}", exc_info=True)
        return pil_img

def deskew_image(pil_img: Image.Image) -> Image.Image:
    """
    Deskews an image using the Projection Profile Method.
    """
    try:
        img_for_deskew = np.array(pil_img.convert("L"))
        thresh = cv2.adaptiveThreshold(img_for_deskew, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 15, 5)

        max_score = -1.0
        best_angle = 0.0

        for angle in np.arange(-15, 15, 0.5):
            (h, w) = thresh.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            hist = np.sum(rotated, axis=1, dtype=np.float32) / 255.0
            score = np.sum((hist[1:] - hist[:-1]) ** 2)
            
            if score > max_score:
                max_score = score
                best_angle = angle
        
        logger.info(f"Detected best skew angle: {best_angle:.2f} degrees")

        if abs(best_angle) < 0.5:
            logger.info("Skew angle is insignificant, skipping rotation.")
            return pil_img

        (h, w) = img_for_deskew.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated = cv2.warpAffine(np.array(pil_img), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rotated)
    except Exception as e:
        logger.error(f"Error during image deskewing: {e}", exc_info=True)
        return pil_img

def remove_horizontal_lines(pil_img):
    """
    Removes horizontal lines from an image.
    """
    try:
        img = np.array(pil_img)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        inverted_img = 255 - img if img.mean() > 127 else img
        
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detected_lines = cv2.morphologyEx(inverted_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        img_no_lines = cv2.subtract(inverted_img, detected_lines)
        
        final_img = 255 - img_no_lines if img.mean() > 127 else img_no_lines
        
        return Image.fromarray(final_img)
    except Exception as e:
        logger.error(f"Error in horizontal line removal: {str(e)}", exc_info=True)
        return pil_img

def segment_image_into_lines(pil_img, det_model):
    """
    Segments an image into lines of text using a provided DocTR detection model.
    """
    temp_path = None
    try:
        # Using a temporary file can be more stable for some library backends
        temp_path = "temp_input_segmentation.png"
        pil_img.convert("RGB").save(temp_path)
        
        doc = DocumentFile.from_images(temp_path)
        result = det_model(doc)
        
        # Ensure result is not empty and has the expected structure
        if not result or 'words' not in result[0] or result[0]['words'].shape[0] == 0:
            return []

        page = result[0]
        img_width, img_height = pil_img.size
        
        # Extract word boxes with absolute coordinates
        words = [
            (int(w[0] * img_width), int(w[1] * img_height), int(w[2] * img_width), int(w[3] * img_height))
            for w in page['words'][:, :-1]
        ]
        
        # Sort words top-to-bottom, then left-to-right
        words.sort(key=lambda w: (w[1], w[0]))

        if not words:
            return []

        # Merge sorted words into lines
        lines = []
        current_line = [words[0]]
        for box in words[1:]:
            last_box = current_line[-1]
            last_box_y_center = (last_box[1] + last_box[3]) / 2
            current_box_y_center = (box[1] + box[3]) / 2
            last_box_height = last_box[3] - last_box[1]

            # Merge if vertically aligned
            if abs(current_box_y_center - last_box_y_center) < last_box_height * 0.7:
                current_line.append(box)
            else:
                # Finalize the previous line
                min_x = min(b[0] for b in current_line)
                min_y = min(b[1] for b in current_line)
                max_x = max(b[2] for b in current_line)
                max_y = max(b[3] for b in current_line)
                lines.append((min_x, min_y, max_x, max_y))
                # Start a new line
                current_line = [box]
        
        # Add the last line
        if current_line:
            min_x = min(b[0] for b in current_line)
            min_y = min(b[1] for b in current_line)
            max_x = max(b[2] for b in current_line)
            max_y = max(b[3] for b in current_line)
            lines.append((min_x, min_y, max_x, max_y))

        return lines
    except Exception as e:
        logger.error(f"Error in line segmentation: {e}", exc_info=True)
        return []
    finally:
        # Clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path) 