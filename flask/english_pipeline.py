from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import numpy as np
import os
import cv2
import torch
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import logging
import sys
import math

# Set up logging to force output to console
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

logger.info("Starting the Unified OCR pipeline application...")

try:
    # Load a unified OCR predictor from doctr
    # Using a lightweight detection model and the requested recognition model.
    logger.info("Loading DocTR OCR predictor with det_arch='db_mobilenet_v3_large' and reco_arch='crnn_mobilenet_v3_small'...")
    ocr_model = ocr_predictor(det_arch='db_mobilenet_v3_large', reco_arch='crnn_mobilenet_v3_small', pretrained=True)
    logger.info("DocTR OCR predictor loaded.")

except Exception as e:
    logger.error(f"Error during model initialization: {str(e)}", exc_info=True)
    raise

def correct_perspective(pil_img: Image.Image) -> Image.Image:
    """
    Corrects perspective distortion in an image of a document.
    Finds the largest quadrilateral in the image and warps it to a top-down view.
    This version includes a check to ensure the detected contour is large enough to be the document.
    """
    try:
        img = np.array(pil_img.convert("RGB"))
        orig = img.copy()
        
        # Downscale for faster processing, preserving aspect ratio
        proc_height = 500.0
        if img.shape[0] <= proc_height:
             ratio = 1.0
             proc_img = img.copy()
        else:
             ratio = img.shape[0] / proc_height
             proc_img = cv2.resize(img, (int(img.shape[1] / ratio), int(proc_height)))

        # Edge detection
        gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        screenCnt = None
        img_area = proc_img.shape[0] * proc_img.shape[1]

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                if cv2.contourArea(approx) > img_area * 0.20: # Must be at least 20% of the image
                    screenCnt = approx
                    break
                else:
                    logger.info(f"Found a 4-point contour, but its area is too small. Skipping it.")
        
        if screenCnt is None:
            logger.warning("Could not find a suitable document contour. Skipping perspective correction.")
            return pil_img
        
        logger.info("Found document contour, applying perspective correction.")

        # Order the points for the perspective transform
        pts = screenCnt.reshape(4, 2) * ratio
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left
        rect[2] = pts[np.argmax(s)] # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right
        rect[3] = pts[np.argmax(diff)] # Bottom-left
        
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
        logger.info("Perspective correction applied successfully.")
        return Image.fromarray(warped)
    except Exception as e:
        logger.error(f"Error during perspective correction: {e}", exc_info=True)
        return pil_img

def deskew_image(pil_img: Image.Image) -> Image.Image:
    """
    Deskews an image using the Projection Profile Method with adaptive thresholding.
    """
    try:
        img_for_deskew = np.array(pil_img.convert("L"))
        
        # Adaptive thresholding is better for images with varying lighting.
        # It creates a binary image where text is white on a black background.
        thresh = cv2.adaptiveThreshold(img_for_deskew, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 15, 5)

        max_score = -1.0
        best_angle = 0.0

        # Test a range of angles
        for angle in np.arange(-15, 15, 0.5):
            (h, w) = thresh.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Rotate the binary image using nearest-neighbor to avoid interpolation artifacts
            rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Calculate horizontal projection profile
            hist = np.sum(rotated, axis=1, dtype=np.float32) / 255.0
            
            # Score is the variance of the projection profile
            score = np.sum((hist[1:] - hist[:-1]) ** 2)
            
            if score > max_score:
                max_score = score
                best_angle = angle
        
        logger.info(f"Detected best skew angle: {best_angle:.2f} degrees")

        # Use a stricter threshold to avoid over-correcting already straight images
        if abs(best_angle) < 0.5:
            logger.info("Skew angle is insignificant, skipping rotation.")
            return pil_img

        # Rotate the original color image by the best angle for a high-quality result
        (h, w) = img_for_deskew.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated = cv2.warpAffine(np.array(pil_img), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        logger.info("Deskew correction applied.")
        return Image.fromarray(rotated)
    except Exception as e:
        logger.error(f"Error during image deskewing: {e}", exc_info=True)
        return pil_img

def remove_horizontal_lines(pil_img):
    try:
        img = np.array(pil_img)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if img.mean() > 127:
            img = 255 - img
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detected_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        img_no_lines = cv2.subtract(img, detected_lines)
        img_no_lines = 255 - img_no_lines
        return Image.fromarray(img_no_lines)
    except Exception as e:
        logger.error(f"Error in horizontal line removal: {str(e)}", exc_info=True)
        return pil_img

@torch.no_grad()
def ocr_pipeline(pil_img):
    logger.info("OCR pipeline started.")
    if pil_img is None:
        # Return empty values for all outputs
        return "Please upload an image.", None, None, None, None

    try:
        # logger.info("Step 1: Correcting perspective...")
        # corrected_img = correct_perspective(pil_img)
        logger.info("Step 1: Perspective correction DISABLED. Passing original image to next step.")
        corrected_img = pil_img # Keep variable for consistent return signature, but it's the original image.

        logger.info("Step 2: Deskewing image...")
        deskewed_img = deskew_image(corrected_img)
        
        logger.info("Step 3: Removing horizontal lines...")
        processed_img_for_detection = remove_horizontal_lines(deskewed_img)
        
        logger.info("Step 4: Running full OCR with DocTR...")
        img_array = np.array(processed_img_for_detection.convert("RGB"))
        result = ocr_model([img_array])
        
        if not result.pages or not result.pages[0].blocks:
            logger.warning("No text detected by DocTR.")
            return "No text detected.", corrected_img, deskewed_img, processed_img_for_detection, deskewed_img.copy().convert("RGB")

        recognized_results = []
        page = result.pages[0]
        page_height, page_width = page.dimensions

        for block in page.blocks:
            for line in block.lines:
                # Reconstruct the line from words
                line_text = " ".join([word.value for word in line.words])
                if not line_text: continue
                
                # Get line coordinates
                x_min, y_min = line.geometry[0]
                x_max, y_max = line.geometry[1]
                line_box = (int(x_min * page_width), int(y_min * page_height), int(x_max * page_width), int(y_max * page_height))
                
                recognized_results.append({'box': line_box, 'text': line_text})
                logger.info(f"Recognized Line: {line_text}")

        # Create visualization on the deskewed image
        vis_img = deskewed_img.copy().convert("RGB")
        draw = ImageDraw.Draw(vis_img)
        
        for res in recognized_results:
            box, text = res['box'], res['text']
            color = "blue"  # Single color for all recognized text
            
            draw.rectangle(box, outline=color, width=2)
            # Display only the recognized text, no classification label needed
            display_text = text
            text_position = (box[0], box[1] - 15 if box[1] > 15 else box[1])
            
            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                font = ImageFont.load_default()

            draw.text(text_position, display_text, fill=color, font=font)
        
        out_text = "\n".join([r['text'] for r in recognized_results])
        logger.info("OCR pipeline completed successfully.")
        
        return out_text, corrected_img, deskewed_img, processed_img_for_detection, vis_img
        
    except Exception as e:
        error_msg = f"Error in OCR pipeline: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # Return original image in the last slot on error, and Nones for the rest
        return error_msg, None, None, None, pil_img

# Create Gradio interface
demo = gr.Interface(
    fn=ocr_pipeline,
    inputs=gr.Image(type="pil", label="Upload Document Image"),
    outputs=[
        gr.Textbox(label="Recognized Text"),
        gr.Image(type="pil", label="1. Perspective Corrected"),
        gr.Image(type="pil", label="2. Deskewed"),
        gr.Image(type="pil", label="3. Preprocessed (Lines Removed)"),
        gr.Image(type="pil", label="4. Final Result (Boxes on Deskewed Image)")
    ],
    title="English OCR Pipeline: Preprocessing Visualization",
    description="Shows the output of each preprocessing step to diagnose failures. 1. Perspective Correction -> 2. Rotational Deskew -> 3. Line Removal -> 4. OCR Result."
)

if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    demo.launch(debug=True, server_name="0.0.0.0", server_port=7863)
    logger.info("Gradio interface running at http://0.0.0.0:7863") 