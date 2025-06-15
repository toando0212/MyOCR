from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from optimum.intel.openvino import OVModelForVision2Seq
# from optimum.intel.openvino import OVModelForVisionEncoderDecoder
from PIL import Image, ImageDraw
import gradio as gr
import numpy as np
import subprocess
import json
import os
import cv2
import torch
# from doctr.models.detection.predictor import DetectionPredictor
from doctr.models import detection
from doctr.io import DocumentFile
from doctr.models import detection_predictor

# from doctr.transforms import Resize, Normalize, ToTensor
from torchvision.transforms import Compose
import logging
import sys
import math

# Set up logging to force output to console
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# If you want to use the Intel GPU (iGPU) with PyTorch, make sure "dml" backend
# is available or install Intel Extension for PyTorch. For now we stick to CPU.
# device = "cuda" if torch.cuda.is_available() else "cpu"
# logger.info(f"Using device: {device}")

logger.info("Starting the OCR application...")

try:
    # Load TrOCR processor and model
    logger.info("Loading TrOCR processor...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten", use_fast=False)
    
    logger.info("Loading TrOCR model (PyTorch)…")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
    model.eval()
    # Send model to CPU explicitly to avoid accidental GPU OOM.
    model.to("cpu")
    logger.info("TrOCR model loaded successfully!")
except Exception as e:
    logger.error(f"Error during model initialization: {str(e)}")
    raise

def doctr_segment_lines(pil_img):
    logger.info("Starting line segmentation...")
    try:
        # Lưu ảnh tạm ra file
        temp_path = "temp_input_doctr.png"
        pil_img.convert("RGB").save(temp_path)
        
        logger.info("Loading image with DocTR...")
        doc = DocumentFile.from_images(temp_path)
        
        logger.info("Initializing DocTR detection model...")
        det_model = detection_predictor(arch="db_mobilenet_v3_large", pretrained=True)
        
        logger.info("Running line detection...")
        result = det_model(doc)
        
        logger.debug(f"Raw detection result from DocTR: {result}")
        
        # The detection predictor returns a list of pages, each with a dict of words.
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
            # Get the y-center of the last box in the current line
            last_box = current_line[-1]
            last_box_y_center = (last_box[1] + last_box[3]) / 2
            
            # Get the y-center of the current box
            current_box_y_center = (box[1] + box[3]) / 2
            
            # Get the height of the last box
            last_box_height = last_box[3] - last_box[1]

            # If the vertical distance between centers is less than a threshold (e.g., 70% of the last box's height),
            # consider them to be on the same line. This is a robust heuristic for typical text.
            if abs(current_box_y_center - last_box_y_center) < last_box_height * 0.7:
                current_line.append(box)
            else:
                # Finalize the previous line by creating an encompassing bounding box
                min_x = min(b[0] for b in current_line)
                min_y = min(b[1] for b in current_line)
                max_x = max(b[2] for b in current_line)
                max_y = max(b[3] for b in current_line)
                lines.append((min_x, min_y, max_x, max_y))
                # Start a new line
                current_line = [box]
        
        # Add the last processed line
        if current_line:
            min_x = min(b[0] for b in current_line)
            min_y = min(b[1] for b in current_line)
            max_x = max(b[2] for b in current_line)
            max_y = max(b[3] for b in current_line)
            lines.append((min_x, min_y, max_x, max_y))

        line_boxes = lines
        logger.info(f"Reconstructed {len(line_boxes)} lines from {len(words)} words.")
        
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file: {str(e)}")
            
        return line_boxes
    except Exception as e:
        logger.error(f"Error in line segmentation: {str(e)}", exc_info=True)
        raise

@torch.no_grad()
def recognize_lines_with_trocr(pil_img, line_boxes):
    logger.info("Starting text recognition...")
    try:
        recognized_lines = []
        logger.info(f"Line boxes: {line_boxes}")  # Log line boxes
        for i, box in enumerate(line_boxes):
            x_min, y_min, x_max, y_max = box
            crop = pil_img.crop((x_min, y_min, x_max, y_max))
            
            # Skip if the crop is too dark or empty
            if np.array(crop).std() < 10:
                logger.debug(f"Skipping line {i+1} due to low contrast")
                continue
                
            logger.debug(f"Processing line {i+1} with box: {box}")
            # Process image and prepare inputs
            # Ensure the crop is converted to RGB before processing, as TrOCR expects 3 channels
            pixel_values = processor(images=crop.convert("RGB"), return_tensors="pt").pixel_values
            logger.info(f"Pixel values shape: {pixel_values.shape}")  # Log shape

            # Generate token ids with greedy decoding
            generated_ids = model.generate(pixel_values)
            logger.info(f"Generated IDs: {generated_ids}")  # Log generated IDs

            # Decode ids to string
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
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
        # Invert if needed (text should be black)
        if img.mean() > 127:
            img = 255 - img
        # Morphological operation to remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detected_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        # Subtract lines from image
        img_no_lines = cv2.subtract(img, detected_lines)
        # Invert back if needed
        img_no_lines = 255 - img_no_lines
        # Return a grayscale image, not a 1-bit image, as it's safer for processing
        return Image.fromarray(img_no_lines)
    except Exception as e:
        logger.error(f"Error in horizontal line removal: {str(e)}")
        raise

def deskew_image(pil_img: Image.Image) -> Image.Image:
    """Detects the skew angle of the text in the image and rotates it to be straight."""
    try:
        # Convert PIL image to OpenCV format, to grayscale
        img = np.array(pil_img.convert("L"))

        # Invert and threshold the image to get a binary image of the text
        img_inverted = cv2.bitwise_not(img)
        thresh = cv2.threshold(img_inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Use Hough Line Transform to detect lines in the image, which is more robust
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None:
            logger.warning("Deskew: No lines detected by Hough Transform, skipping rotation.")
            return pil_img

        # Calculate the angle for each detected line
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle in degrees
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)

        # Get the median angle, which is more robust to outliers
        median_angle = np.median(angles)
        
        logger.info(f"Detected skew angle via Hough Transform: {median_angle:.2f} degrees")

        # Only rotate if the angle is significant to avoid small errors on straight images
        if abs(median_angle) < 1:
            logger.info("Skew angle is insignificant, skipping rotation.")
            return pil_img

        # Rotate the image to deskew it
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        # Use the original PIL image for rotation to preserve color/quality
        rotated = cv2.warpAffine(np.array(pil_img), M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return Image.fromarray(rotated)
    except Exception as e:
        logger.error(f"Error during image deskewing: {e}", exc_info=True)
        # If deskewing fails, return the original image
        return pil_img

def ocr_pipeline(pil_img):
    logger.info("OCR pipeline started for a new image.")
    try:
        if pil_img is None:
            logger.warning("Input image is None, stopping pipeline.")
            return "Please upload an image.", None, None, None
            
        # 1. Deskew the image to straighten the text
        logger.info("Deskewing image...")
        deskewed_img = deskew_image(pil_img)
        
        # 2. Remove horizontal lines for better line detection
        logger.info("Removing horizontal lines...")
        processed_img_for_detection = remove_horizontal_lines(deskewed_img)
        
        # 3. Detect lines using the processed image
        logger.info("Detecting lines...")
        line_boxes = doctr_segment_lines(processed_img_for_detection)
        
        if not line_boxes:
            logger.warning("No lines were detected by DocTR.")
            return "No lines detected.", processed_img_for_detection, deskewed_img.copy().convert("RGB"), None

        # 4. Recognize text using the cleaned image for better accuracy
        logger.info("Recognizing text...")
        recognized_lines = recognize_lines_with_trocr(processed_img_for_detection, line_boxes)
        
        # 5. Create visualization on a copy of the deskewed image
        logger.info("Creating visualization...")
        line_img = deskewed_img.copy().convert("RGB")
        draw_line = ImageDraw.Draw(line_img)
        
        for i, box in enumerate(line_boxes):
            draw_line.rectangle([box[0], box[1], box[2], box[3]], outline="orange", width=2)
            # Draw text only if a corresponding recognized line exists
            if i < len(recognized_lines):
                # Simple positioning for the label text above the box
                text_position = (box[0], box[1] - 15 if box[1] > 15 else box[1])
                draw_line.text(text_position, recognized_lines[i], fill="orange")
        
        # Create mask using dimensions from the deskewed image
        mask = np.zeros((deskewed_img.height, deskewed_img.width), dtype=np.uint8)
        for box in line_boxes:
            # Ensure box coordinates are integers for numpy slicing
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
    title="Unified Line-level OCR with DocTR Line Segmentation and TrOCR Recognition",
    description="Detects lines using DocTR, recognizes each line with TrOCR, and visualizes detected line bounding boxes and mask."
)

if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    demo.launch(debug=True, server_name="0.0.0.0", server_port=7861)
    logger.info("Gradio interface running at http://0.0.0.0:7861")

