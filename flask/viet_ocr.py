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
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from pyvi import ViTokenizer
from pipeline_utils import correct_perspective, deskew_image, remove_horizontal_lines

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
    # Load VietOCR model for recognition
    logger.info("Loading VietOCR model (vgg_seq2seq)...")
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cpu' # Force CPU for compatibility
    config['predictor']['beamsearch'] = False
    vietocr_predictor = Predictor(config)
    logger.info("VietOCR model loaded successfully!")

    # Load Doctr predictor for line detection only
    # The recognition architecture is specified but won't be used because we call with detect_only=True
    logger.info("Initializing DocTR predictor for line detection...")
    doctr_predictor = ocr_predictor(det_arch='fast_small', pretrained=True)
    logger.info("DocTR predictor loaded successfully!")

except Exception as e:
    logger.error(f"Error during model initialization: {str(e)}", exc_info=True)
    raise

def merge_boxes_to_lines(boxes: list, page_dims: tuple, tolerance_ratio: float = 0.7) -> list:
    """
    Merges bounding boxes into lines of text, which is crucial for handwriting.
    
    Args:
        boxes: A list of bounding boxes, each defined as (xmin, ymin, xmax, ymax)
               in relative coordinates (0 to 1).
        page_dims: A tuple (height, width) of the page.
        tolerance_ratio: The vertical tolerance for merging boxes into the same line,
                         as a ratio of the box height.
    
    Returns:
        A list of merged bounding boxes representing text lines in absolute coordinates.
    """
    if not boxes:
        return []

    page_height, page_width = page_dims
    # Convert boxes to absolute coordinates and include their height for sorting
    abs_boxes = [
        (int(b[0] * page_width), int(b[1] * page_height), int(b[2] * page_width), int(b[3] * page_height))
        for b in boxes
    ]

    # Sort boxes primarily by their vertical position, then horizontal
    sorted_boxes = sorted(abs_boxes, key=lambda x: (x[1], x[0]))

    merged_lines = []
    if not sorted_boxes:
        return []

    current_line = list(sorted_boxes[0])

    for i in range(1, len(sorted_boxes)):
        box = sorted_boxes[i]
        
        current_line_height = current_line[3] - current_line[1]
        current_line_y_center = current_line[1] + current_line_height / 2
        
        box_height = box[3] - box[1]
        box_y_center = box[1] + box_height / 2

        vertical_tolerance = min(current_line_height, box_height) * tolerance_ratio

        # Check if the box y-center is within the vertical tolerance of the current line's y-center
        if abs(box_y_center - current_line_y_center) <= vertical_tolerance:
            # Merge box into current line by expanding the line's bounding box
            current_line[0] = min(current_line[0], box[0])
            current_line[1] = min(current_line[1], box[1])
            current_line[2] = max(current_line[2], box[2])
            current_line[3] = max(current_line[3], box[3])
        else:
            merged_lines.append(tuple(current_line))
            current_line = list(box)

    merged_lines.append(tuple(current_line))
    
    return merged_lines

def post_process_text(text: str) -> str:
    """
    Correctly segments Vietnamese words using pyvi.
    """
    logger.info("Starting post-processing (Vietnamese word segmentation)...")
    try:
        # Tokenize the text to add underscores between compound words
        tokenized_text = ViTokenizer.tokenize(text)
        # Replace underscores with spaces for better readability
        readable_text = tokenized_text.replace('_', ' ')
        logger.info("Post-processing completed.")
        return readable_text
    except Exception as e:
        logger.error(f"Error during post-processing: {e}", exc_info=True)
        return text # Return original text on error

@torch.no_grad()
def ocr_pipeline(pil_img):
    logger.info("OCR pipeline started.")
    if pil_img is None:
        # Return empty values for all outputs
        return "Please upload an image.", "", None, None, None, None

    try:
        # Step 1: Preprocessing
        logger.info("Step 1: Correcting perspective...")
        corrected_img = correct_perspective(pil_img)

        logger.info("Step 2: Deskewing image...")
        deskewed_img = deskew_image(corrected_img)
        
        logger.info("Step 3: Removing horizontal lines...")
        processed_img_for_detection = remove_horizontal_lines(deskewed_img)
        
        # Step 4: Line Detection using DocTR
        logger.info("Step 4: Running DocTR to detect lines (recognition results will be replaced)...")
        img_array = np.array(processed_img_for_detection.convert("RGB"))
        
        # Run full OCR with DocTR. We will use its detection results (bounding boxes)
        # and then use VietOCR for the actual text recognition. This avoids the 'detect_only' error.
        result = doctr_predictor([img_array])
        
        if not result.pages or not result.pages[0].blocks:
            logger.warning("No text boxes detected by DocTR.")
            # Return intermediate images for debugging
            return "No text boxes detected.", "", corrected_img, deskewed_img, processed_img_for_detection, deskewed_img.copy().convert("RGB")

        # Step 4a: Extract all detected boxes from DocTR
        logger.info("Step 4a: Extracting individual boxes from DocTR result...")
        page = result.pages[0]
        page_dims = page.dimensions # (height, width)
        all_boxes = []
        for block in page.blocks:
            for line in block.lines:
                all_boxes.append((line.geometry[0][0], line.geometry[0][1], line.geometry[1][0], line.geometry[1][1]))

        # Step 4b: Merge boxes into lines
        logger.info("Step 4b: Merging detected boxes into lines for handwriting...")
        merged_line_boxes = merge_boxes_to_lines(all_boxes, page_dims)
        if not merged_line_boxes:
            logger.warning("Box merging resulted in no lines. OCR will be empty.")
            return "Box merging failed to create any lines.", "", corrected_img, deskewed_img, processed_img_for_detection, deskewed_img.copy().convert("RGB")

        # Step 5: Text Recognition using VietOCR on MERGED lines
        logger.info("Step 5: Recognizing text with VietOCR on merged lines...")
        recognized_results = []
        
        for line_box in merged_line_boxes:
            # Crop the line from the clean, deskewed image for best recognition results
            line_crop = deskewed_img.crop(line_box)
            
            # Recognize text using VietOCR
            if line_crop.width > 0 and line_crop.height > 0:
                try:
                    line_text = vietocr_predictor.predict(line_crop)
                except Exception as recog_e:
                    logger.warning(f"VietOCR failed to recognize a line crop: {recog_e}")
                    line_text = "" # Assign empty string on failure
            else:
                line_text = ""

            if not line_text:
                logger.debug("Skipping line due to empty recognition result.")
                continue
            
            recognized_results.append({
                'box': line_box, 
                'text': line_text, 
            })
            logger.info(f"Recognized Line: {line_text}")
        
        # Step 6: Create visualization on the deskewed image
        logger.info("Step 6: Creating visualization...")
        vis_img = deskewed_img.copy().convert("RGB")
        draw = ImageDraw.Draw(vis_img)
        
        for res in recognized_results:
            box, text = res['box'], res['text']
            draw.rectangle(box, outline="blue", width=2)
            
            # Position text above the bounding box
            text_position = (box[0], box[1] - 15 if box[1] > 15 else box[1])
            
            try:
                # Use a common font if available
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                # Fallback to default font
                font = ImageFont.load_default()
                
            draw.text(text_position, text, fill="blue", font=font)
        
        out_text = "\n".join([r['text'] for r in recognized_results])
        logger.info("OCR pipeline completed successfully.")
        
        # Step 7: Post-processing
        post_processed_text = post_process_text(out_text)
        
        return out_text, post_processed_text, corrected_img, deskewed_img, processed_img_for_detection, vis_img

    except Exception as e:
        error_msg = f"Error in OCR pipeline: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # Return original image in the last slot on error, and Nones for the rest
        return error_msg, "", None, None, None, pil_img

# Create Gradio interface
demo = gr.Interface(
    fn=ocr_pipeline,
    inputs=gr.Image(type="pil", label="Upload Document Image"),
    outputs=[
        gr.Textbox(label="1. Raw Recognized Text"),
        gr.Textbox(label="2. Post-Processed Text (Word Segmented)"),
        gr.Image(type="pil", label="3. Perspective Corrected"),
        gr.Image(type="pil", label="4. Deskewed"),
        gr.Image(type="pil", label="5. Preprocessed (Lines Removed)"),
        gr.Image(type="pil", label="6. Final Result (Boxes on Deskewed Image)")
    ],
    title="Vietnamese OCR Pipeline: Preprocessing, Recognition, and Post-processing",
    description="Shows the output of each step. 1. Perspective Correction -> 2. Rotational Deskew -> 3. Line Removal -> 4. Line Detection (DocTR) -> 4b. Box Merging -> 5. Recognition (VietOCR) -> 6. Post-processing (Word Segmentation)."
)

if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    demo.launch(debug=True, server_name="0.0.0.0", server_port=7862)
    logger.info("Gradio interface running at http://0.0.0.0:7862") 