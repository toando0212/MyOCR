from PIL import Image
import numpy as np
import cv2
import logging
import os
import math
from doctr.io import DocumentFile

logger = logging.getLogger(__name__)

def find_best_document_contour(contours, img):
    """
    Finds the best contour that likely represents a document by checking
    geometry and brightness.
    """
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    img_area = img.shape[0] * img.shape[1]
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            contour_area = cv2.contourArea(approx)
            # Check if contour is reasonably large but not the whole image
            if 0.1 < (contour_area / img_area) < 0.95 and cv2.isContourConvex(approx):
                # Check if the contour is mostly white, as a document should be
                mask = np.zeros(img.shape[:2], dtype="uint8")
                cv2.drawContours(mask, [approx], -1, 255, -1)
                
                # We need to handle potential divide by zero if mask area is 0
                mask_area = cv2.countNonZero(mask)
                if mask_area == 0:
                    continue

                # Use RGB image for mean calculation
                mean_val = cv2.mean(img, mask=mask)[:3]
                mean_brightness = np.mean(mean_val)
                
                # A document should be bright (higher values)
                if mean_brightness > 128: 
                    return approx
    
    return None

def correct_perspective(pil_img):
    """
    Corrects perspective on challenging images, especially open books with curves and shadows.
    This version uses aggressive thresholding and morphological operations to isolate the document
    and then finds the minimum area rectangle to handle non-flat surfaces.
    """
    try:
        img = np.array(pil_img.convert("RGB"))
        original_image = img.copy()
        
        # 1. Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Using a bilateral filter is better at preserving edges while removing noise.
        # This helps separate the page from the background.
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive thresholding is key for handling the uneven lighting and shadows.
        thresh = cv2.adaptiveThreshold(blurred, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 21, 5)

        # 2. Morphological Operations
        # Use a large kernel to close gaps and merge the book pages into a single blob.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Erode and dilate to remove any small noise that survived the closing.
        closed = cv2.erode(closed, None, iterations=2)
        closed = cv2.dilate(closed, None, iterations=2)

        # 3. Find Contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.error("No contours found after aggressive preprocessing. Cannot correct perspective.")
            return pil_img

        # Try to find a 4-point document contour first
        doc_contour = find_best_document_contour(contours, original_image)
        
        # 4. Get Bounding Box and Warp
        if doc_contour is not None:
            # Found a good 4-point contour, use it
            logger.info("Found a 4-point document contour, using it for perspective correction.")
            pts = doc_contour.reshape(4, 2).astype("float32")
            
            # Order the points for the transform
            ordered_pts = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            ordered_pts[0] = pts[np.argmin(s)]
            ordered_pts[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            ordered_pts[1] = pts[np.argmin(diff)]
            ordered_pts[3] = pts[np.argmax(diff)]
        
        else:
            # Fallback to the largest contour and minAreaRect
            logger.warning("Could not find a clear 4-point document contour. Falling back to minAreaRect.")
            c = max(contours, key=cv2.contourArea)

            if cv2.contourArea(c) < (img.shape[0] * img.shape[1] * 0.1):
                 logger.warning("Largest contour is too small. Returning original image.")
                 return pil_img
                 
            # This is the most important step for handling the curved book.
            # It finds the tightest 4-point bounding box around the largest contour.
            rect = cv2.minAreaRect(c)
            
            # Get the 4 corners of the bounding box
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Order the points for the transform
            pts = box.astype("float32")
            ordered_pts = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            ordered_pts[0] = pts[np.argmin(s)]
            ordered_pts[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            ordered_pts[1] = pts[np.argmin(diff)]
            ordered_pts[3] = pts[np.argmax(diff)]
        
        (tl, tr, br, bl) = ordered_pts
        
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        if maxWidth == 0 or maxHeight == 0:
            logger.error("Calculated warped dimensions are zero. Aborting.")
            return pil_img

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
            
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(original_image, M, (maxWidth, maxHeight))
        
        logger.info("Perspective corrected successfully.")
        return Image.fromarray(warped)

    except Exception as e:
        logger.error(f"Critical error in perspective correction: {str(e)}", exc_info=True)
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
    try:
        # Process image in-memory to avoid filesystem permission issues
        doc = DocumentFile.from_images([np.array(pil_img.convert("RGB"))])
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