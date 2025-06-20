from PIL import Image, ImageDraw
import gradio as gr
import numpy as np
import subprocess
import json
import os
import cv2
import torch
from doctr.models import detection_predictor
from doctr.io import DocumentFile
from torchvision.transforms import Compose
import logging
import sys
import math
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# Set up logging to force output to console
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

logger.info("Starting the OCR application...")

try:
    # Load VietOCR model
    logger.info("Loading VietOCR model (vgg_seq2seq)...")
    # Configure VietOCR
    config = Cfg.load_config_from_name('vgg_seq2seq')
    # config = Cfg.load_config_from_name('vgg_transformer')
    # Set device to CPU
    config['device'] = 'cpu'
    config['predictor']['beamsearch'] = False
    # Initialize the predictor
    vietocr_predictor = Predictor(config)
    logger.info("VietOCR model loaded successfully!")
except Exception as e:
    logger.error(f"Error during model initialization: {str(e)}")
    raise

def doctr_segment_lines(pil_img):
    logger.info("Starting line segmentation...")
    try:
        # Save temp image to file
        temp_path = "temp_input_doctr.png"
        pil_img.convert("RGB").save(temp_path)
        
        logger.info("Loading image with DocTR...")
        doc = DocumentFile.from_images(temp_path)
        
        logger.info("Initializing DocTR detection model...")
        det_model = detection_predictor(arch="db_mobilenet_v3_large", pretrained=True)
        
        logger.info("Running line detection...")
        result = det_model(doc)
        
        logger.debug(f"Raw detection result from DocTR: {result}")
        
        if not result:
            logger.warning("DocTR returned an empty result list.")
            return []
            
        page = result[0]
        if 'words' not in page or page['words'].shape[0] == 0:
            logger.warning("DocTR detected 0 words on the page.")
            return []

        # Get absolute word boxes
        img_width, img_height = pil_img.size
        words = [
            (
                int(w[0] * img_width),
                int(w[1] * img_height),
                int(w[2] * img_width),
                int(w[3] * img_height)
            )
            for w in page['words'][:, :-1]
        ]

        # Sort words by their vertical position, then horizontal
        words.sort(key=lambda w: (w[1], w[0]))

        lines = []
        if not words:
            return []

        # Heuristic-based line reconstruction
        current_line = [words[0]]
        for box in words[1:]:
            last_box = current_line[-1]
            last_box_y_center = (last_box[1] + last_box[3]) / 2
            current_box_y_center = (box[1] + box[3]) / 2
            last_box_height = last_box[3] - last_box[1]

            if abs(current_box_y_center - last_box_y_center) < last_box_height * 0.7:
                current_line.append(box)
            else:
                min_x = min(b[0] for b in current_line)
                min_y = min(b[1] for b in current_line)
                max_x = max(b[2] for b in current_line)
                max_y = max(b[3] for b in current_line)
                lines.append((min_x, min_y, max_x, max_y))
                current_line = [box]
        
        if current_line:
            min_x = min(b[0] for b in current_line)
            min_y = min(b[1] for b in current_line)
            max_x = max(b[2] for b in current_line)
            max_y = max(b[3] for b in current_line)
            lines.append((min_x, min_y, max_x, max_y))

        line_boxes = lines
        logger.info(f"Reconstructed {len(line_boxes)} lines from {len(words)} words.")
        
        try:
            os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file: {str(e)}")
            
        return line_boxes
    except Exception as e:
        logger.error(f"Error in line segmentation: {str(e)}", exc_info=True)
        raise

@torch.no_grad()
def recognize_lines_with_vietocr(pil_img, line_boxes):
    logger.info("Starting text recognition with VietOCR...")
    try:
        recognized_lines = []
        logger.info(f"Line boxes: {line_boxes}")
        for i, box in enumerate(line_boxes):
            x_min, y_min, x_max, y_max = box
            crop = pil_img.crop((x_min, y_min, x_max, y_max))
            
            if np.array(crop).std() < 10:
                logger.debug(f"Skipping line {i+1} due to low contrast")
                continue
                
            logger.debug(f"Processing line {i+1} with box: {box}")
            
            # Recognize text using VietOCR
            text = vietocr_predictor.predict(crop)
            
            if len(text) < 2:
                logger.debug(f"Skipping line {i+1} due to short text: '{text}'")
                continue
                
            recognized_lines.append(text)
            logger.debug(f"Recognized line {i+1}: {text}")
            
        logger.info(f"Successfully recognized {len(recognized_lines)} lines")
        return recognized_lines
    except Exception as e:
        logger.error(f"Error in text recognition: {str(e)}")
        raise

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
        logger.error(f"Error in horizontal line removal: {str(e)}")
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
        return Image.fromarray(warped)
    except Exception as e:
        logger.error(f"Error during perspective correction: {e}", exc_info=True)
        return pil_img

def deskew_image(pil_img: Image.Image) -> Image.Image:
    """Detects the skew angle of the text in the image and rotates it to be straight."""
    try:
        img = np.array(pil_img.convert("L"))
        img_inverted = cv2.bitwise_not(img)
        thresh = cv2.threshold(img_inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None:
            logger.warning("Deskew: No lines detected by Hough Transform, skipping rotation.")
            return pil_img

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)

        median_angle = np.median(angles)
        logger.info(f"Detected skew angle via Hough Transform: {median_angle:.2f} degrees")

        if abs(median_angle) < 1:
            logger.info("Skew angle is insignificant, skipping rotation.")
            return pil_img

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(np.array(pil_img), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rotated)
    except Exception as e:
        logger.error(f"Error during image deskewing: {e}", exc_info=True)
        return pil_img

def ocr_pipeline(pil_img):
    logger.info("OCR pipeline started for a new image.")
    try:
        if pil_img is None:
            logger.warning("Input image is None, stopping pipeline.")
            return "Please upload an image.", None, None, None
            
        logger.info("Correcting perspective...")
        corrected_img = correct_perspective(pil_img)

        logger.info("Deskewing image...")
        deskewed_img = deskew_image(corrected_img)
        
        logger.info("Removing horizontal lines...")
        processed_img_for_detection = remove_horizontal_lines(deskewed_img)
        
        logger.info("Detecting lines...")
        line_boxes = doctr_segment_lines(processed_img_for_detection)
        
        if not line_boxes:
            logger.warning("No lines were detected by DocTR.")
            return "No lines detected.", processed_img_for_detection, deskewed_img.copy().convert("RGB"), None

        logger.info("Recognizing text with VietOCR...")
        recognized_lines = recognize_lines_with_vietocr(deskewed_img, line_boxes)
        
        logger.info("Creating visualization...")
        line_img = deskewed_img.copy().convert("RGB")
        draw_line = ImageDraw.Draw(line_img)
        
        for i, box in enumerate(line_boxes):
            draw_line.rectangle([box[0], box[1], box[2], box[3]], outline="orange", width=2)
            if i < len(recognized_lines):
                text_position = (box[0], box[1] - 15 if box[1] > 15 else box[1])
                draw_line.text(text_position, recognized_lines[i], fill="orange")
        
        mask = np.zeros((deskewed_img.height, deskewed_img.width), dtype=np.uint8)
        for box in line_boxes:
            y_min, y_max = int(box[1]), int(box[3])
            x_min, x_max = int(box[0]), int(box[2])
            mask[y_min:y_max, x_min:x_max] = 255
            
        out_text = "\n".join([f"Line {i+1}: {t}" for i, t in enumerate(recognized_lines)])
        logger.info("OCR pipeline completed successfully.")
        
        return out_text, processed_img_for_detection, line_img, Image.fromarray(mask)
    except Exception as e:
        error_msg = f"Error in OCR pipeline: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg, None, pil_img, None

# Create Gradio interface
demo = gr.Interface(
    fn=ocr_pipeline,
    inputs=gr.Image(type="pil", label="Upload Document Image"),
    outputs=[
        gr.Textbox(label="Recognized Lines"),
        gr.Image(type="pil", label="Preprocessed Image (for Detection)"),
        gr.Image(type="pil", label="Line Bounding Boxes (DocTR)"),
        gr.Image(type="pil", label="Line Mask (DocTR)")
    ],
    title="Unified Line-level OCR with DocTR Line Segmentation and VietOCR Recognition",
    description="Detects lines using DocTR, recognizes each line with VietOCR, and visualizes detected line bounding boxes and mask."
)

if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    demo.launch(debug=True, server_name="0.0.0.0", server_port=7862)
    logger.info("Gradio interface running at http://0.0.0.0:7861") 