from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import numpy as np
import os
import cv2
import torch
from doctr.models import detection_predictor
from doctr.io import DocumentFile
import logging
import sys
import math
from joblib import load
import pandas as pd

# Set up logging to force output to console
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

logger.info("Starting the English OCR pipeline application...")

try:
    # Load TrOCR models
    logger.info("Loading TrOCR printed model (microsoft/trocr-small-printed)...")
    printed_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
    printed_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
    printed_model.to("cpu")
    logger.info("Printed model loaded.")

    logger.info("Loading TrOCR handwritten model (microsoft/trocr-small-handwritten)...")
    handwritten_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
    handwritten_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
    handwritten_model.to("cpu")
    logger.info("Handwritten model loaded.")

    # Load classifier
    logger.info("Loading text classifier model...")
    CLASSIFIER_PATH = 'flask/full_model.joblib'
    classifier_model = load(CLASSIFIER_PATH)
    logger.info("Classifier model loaded.")

except Exception as e:
    logger.error(f"Error during model initialization: {str(e)}", exc_info=True)
    raise

def extract_features(img):
    """Extracts features for the classifier from a given image region."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rows, cols = img.shape
    if cols == 0 or rows == 0:
        return [0, 0, 0, 0, 0]

    arr = [rows, cols, rows / cols]
    
    _, bwMask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    myavg = 0
    for xx in range(cols):
        mycnt = np.sum(bwMask[:, xx] == 0)
        myavg += (mycnt * 1.0) / rows
    myavg /= cols
    arr.append(myavg)
    
    change = 0
    for xx in range(rows):
        row_data = bwMask[xx, :]
        mycnt = np.sum(row_data[:-1] != row_data[1:])
        change += (mycnt * 1.0) / cols
    change /= rows
    arr.append(change)
    
    return arr

def classify_region(roi_img):
    """Classifies an image region as 'Handwritten' or 'Printed'."""
    try:
        features = extract_features(roi_img)
        pred = classifier_model.predict([features])[0]
        return str(pred)
    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        return "Unknown"

def doctr_segment_lines(pil_img):
    logger.info("Starting line segmentation with DocTR...")
    try:
        temp_path = "temp_input_doctr.png"
        pil_img.convert("RGB").save(temp_path)
        
        doc = DocumentFile.from_images(temp_path)
        
        det_model = detection_predictor(arch="db_mobilenet_v3_large", pretrained=True)
        
        result = det_model(doc)
        
        if not result or 'words' not in result[0] or result[0]['words'].shape[0] == 0:
            logger.warning("DocTR detected 0 words.")
            os.remove(temp_path)
            return []

        page = result[0]
        img_width, img_height = pil_img.size
        words = [
            (int(w[0] * img_width), int(w[1] * img_height), int(w[2] * img_width), int(w[3] * img_height))
            for w in page['words'][:, :-1]
        ]
        words.sort(key=lambda w: (w[1], w[0]))

        lines = []
        if not words:
            os.remove(temp_path)
            return []

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

        logger.info(f"Reconstructed {len(lines)} lines from {len(words)} words.")
        os.remove(temp_path)
        return lines
    except Exception as e:
        logger.error(f"Error in line segmentation: {str(e)}", exc_info=True)
        return []

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

def deskew_image(pil_img: Image.Image) -> Image.Image:
    try:
        img = np.array(pil_img.convert("L"))
        img_inverted = cv2.bitwise_not(img)
        thresh = cv2.threshold(img_inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None:
            logger.warning("Deskew: No lines detected, skipping rotation.")
            return pil_img

        angles = [math.degrees(math.atan2(y2 - y1, x2 - x1)) for line in lines for x1, y1, x2, y2 in line]
        median_angle = np.median(angles)
        
        logger.info(f"Detected skew angle: {median_angle:.2f} degrees")

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

@torch.no_grad()
def ocr_pipeline(pil_img):
    logger.info("OCR pipeline started.")
    if pil_img is None:
        return "Please upload an image.", None, None

    try:
        logger.info("Deskewing image...")
        deskewed_img = deskew_image(pil_img)
        
        logger.info("Removing horizontal lines...")
        processed_img_for_detection = remove_horizontal_lines(deskewed_img)
        
        logger.info("Detecting lines...")
        line_boxes = doctr_segment_lines(processed_img_for_detection)
        
        if not line_boxes:
            return "No lines detected.", processed_img_for_detection, deskewed_img.copy().convert("RGB")

        recognized_results = []
        for i, box in enumerate(line_boxes):
            x_min, y_min, x_max, y_max = box
            crop = processed_img_for_detection.crop((x_min, y_min, x_max, y_max))
            
            if np.array(crop).std() < 10:
                continue

            # Classify line
            line_label = classify_region(np.array(crop.convert("RGB")))
            logger.info(f"Line {i+1} classified as: {line_label}")

            # Recognize text
            if 'handwritten' in line_label.lower():
                processor = handwritten_processor
                model = handwritten_model
            else: # Default to printed
                processor = printed_processor
                model = printed_model
            
            pixel_values = processor(images=crop.convert("RGB"), return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            if len(text) < 2:
                continue

            recognized_results.append({'box': box, 'label': line_label, 'text': text})
            logger.info(f"Recognized Line {i+1} ({line_label}): {text}")

        # Create visualization
        line_img = deskewed_img.copy().convert("RGB")
        draw = ImageDraw.Draw(line_img)
        
        for res in recognized_results:
            box, label, text = res['box'], res['label'], res['text']
            # Determine color based on label
            color = "green" if 'handwritten' in label.lower() else "blue" if 'printed' in label.lower() else "orange"
            
            draw.rectangle(box, outline=color, width=2)
            
            # Format text to display
            display_text = f"[{label}] {text}"
            text_position = (box[0], box[1] - 15 if box[1] > 15 else box[1])
            
            try:
                # Use a default font
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                font = ImageFont.load_default()

            draw.text(text_position, display_text, fill=color, font=font)
        
        out_text = "\n".join([f"Line {i+1} ({r['label']}): {r['text']}" for i, r in enumerate(recognized_results)])
        logger.info("OCR pipeline completed successfully.")
        
        return out_text, processed_img_for_detection, line_img
        
    except Exception as e:
        error_msg = f"Error in OCR pipeline: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg, None, pil_img

# Create Gradio interface
demo = gr.Interface(
    fn=ocr_pipeline,
    inputs=gr.Image(type="pil", label="Upload Document Image"),
    outputs=[
        gr.Textbox(label="Recognized Lines (with Type)"),
        gr.Image(type="pil", label="Preprocessed Image (for Detection)"),
        gr.Image(type="pil", label="Classified & Recognized Lines")
    ],
    title="English OCR Pipeline: Line Detection, Classification, and Recognition",
    description="Detects lines, classifies each as Handwritten or Printed, then uses the appropriate TrOCR model for recognition."
)

if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    demo.launch(debug=True, server_name="0.0.0.0", server_port=7863)
    logger.info("Gradio interface running at http://0.0.0.0:7863") 