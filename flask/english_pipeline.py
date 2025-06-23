from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import numpy as np
import os
import cv2
import torch
from doctr.models import detection_predictor, recognition_predictor, ocr_predictor
from doctr.io import DocumentFile
import logging
import sys
import math
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

logger.info("Starting the English OCR pipeline application...")

try:
    # Use the high-level OCR predictor for detection. It's more robust and provides structured output.
    # logger.info("Loading DocTR full OCR predictor ('db_mobilenet_v3_large', 'crnn_mobilenet_v3_large')...")
    # We use the full predictor to get structured line geometry, then we'll re-run recognition on merged lines.
    doctr_predictor = ocr_predictor(
        det_arch='fast_small', 
        reco_arch='vitstr_small', # Using a more standard recognition model for general purpose
        pretrained=True
    )

    # Move model to GPU if available
    if torch.cuda.is_available():
        logger.info("Moving models to CUDA device...")
        doctr_predictor.to(torch.device('cuda'))
    
    logger.info("DocTR models loaded successfully.")

except Exception as e:
    logger.error(f"Error during model initialization: {str(e)}", exc_info=True)
    raise

def merge_boxes_to_lines(boxes: list, page_dims: tuple, tolerance_ratio: float = 0.7) -> list:
    """
    Merges geometric bounding boxes into lines of text.
    Almost identical to the function in the VietOCR pipeline.
    
    Args:
        boxes: List of boxes, each as ((xmin, ymin), (xmax, ymax)) in relative coordinates.
        page_dims: Tuple (height, width) of the page.
    
    Returns:
        List of merged bounding boxes representing text lines in absolute coordinates.
    """
    if not boxes:
        return []

    page_height, page_width = page_dims
    # Convert boxes to absolute coordinates (xmin, ymin, xmax, ymax)
    abs_boxes = [
        (int(b[0] * page_width), int(b[1] * page_height), int(b[2] * page_width), int(b[3] * page_height))
        for b in boxes
    ]

    # Sort boxes top-to-bottom, then left-to-right
    sorted_boxes = sorted(abs_boxes, key=lambda x: (x[1], x[0]))

    if not sorted_boxes: return []

    merged_lines = []
    current_line = list(sorted_boxes[0])

    for i in range(1, len(sorted_boxes)):
        box = sorted_boxes[i]
        
        current_line_height = current_line[3] - current_line[1]
        current_line_y_center = current_line[1] + current_line_height / 2
        
        box_height = box[3] - box[1]
        box_y_center = box[1] + box_height / 2

        vertical_tolerance = min(current_line_height, box_height) * tolerance_ratio if current_line_height > 0 and box_height > 0 else 0

        if abs(box_y_center - current_line_y_center) <= vertical_tolerance:
            # Merge box into current line
            current_line[0] = min(current_line[0], box[0])
            current_line[1] = min(current_line[1], box[1])
            current_line[2] = max(current_line[2], box[2])
            current_line[3] = max(current_line[3], box[3])
        else:
            merged_lines.append(tuple(current_line))
            current_line = list(box)

    merged_lines.append(tuple(current_line))
    return merged_lines

@torch.no_grad()
def ocr_pipeline(pil_img):
    logger.info("OCR pipeline started.")
    if pil_img is None:
        return "Please upload an image.", None, None, None, None

    try:
        # Step 1-3: Preprocessing (remains the same)
        logger.info("Steps 1-3: Correcting perspective, deskewing, and removing lines...")
        corrected_img = correct_perspective(pil_img)
        deskewed_img = deskew_image(corrected_img)
        processed_img_for_detection = remove_horizontal_lines(deskewed_img)
        
        # Step 4: Run detection using the FULL ocr_predictor to get structured output
        logger.info("Step 4: Running DocTR to get initial line geometry...")
        # The predictor expects a list of numpy arrays, ensure it's RGB
        img_array = np.array(processed_img_for_detection.convert("RGB"))
        result = doctr_predictor([img_array])

        if not result.pages or not result.pages[0].blocks:
            logger.warning("No text blocks detected by DocTR.")
            return "No text blocks detected.", corrected_img, deskewed_img, processed_img_for_detection, deskewed_img.copy().convert("RGB")

        # The pipeline is now simplified. We directly use the results from the ocr_predictor.
        # No more manual box merging or second-pass recognition. This is the "plug-and-play" approach.

        # Step 5: Extract text and visualization data directly from the result
        logger.info("Step 5: Extracting text and geometries from DocTR result...")
        out_text = result.render()
        
        # Step 6: Create visualization
        vis_img = deskewed_img.copy().convert("RGB")
        draw = ImageDraw.Draw(vis_img)
        page_dims = vis_img.size # (width, height)
        
        # Draw boxes for each line from the predictor's output
        for block in result.pages[0].blocks:
            for line in block.lines:
                # Get the line's text content
                line_text = " ".join([word.value for word in line.words])
                
                # Get the line's geometry (relative coordinates)
                xmin, ymin = line.geometry[0]
                xmax, ymax = line.geometry[1]
                
                # Convert to absolute pixel coordinates
                box = [int(xmin * page_dims[0]), int(ymin * page_dims[1]), int(xmax * page_dims[0]), int(ymax * page_dims[1])]
                
                draw.rectangle(box, outline="blue", width=2)
                try:
                    font = ImageFont.truetype("arial.ttf", 15)
                except IOError:
                    font = ImageFont.load_default()
                # Position text slightly above the top-left corner of the box
                draw.text((box[0], box[1] - 15), line_text, fill="blue", font=font)
        
        logger.info("OCR pipeline completed successfully.")
        
        return out_text, corrected_img, deskewed_img, processed_img_for_detection, vis_img
        
    except Exception as e:
        error_msg = f"Error in OCR pipeline: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg, None, None, None, pil_img

# Gradio interface remains largely the same
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
    title="English OCR Pipeline (REFACTORED for Handwriting)",
    description="Shows the output of each step. NEW: 1. Detect Only -> 2. Merge Boxes -> 3. Recognize Line-by-Line. This should improve handwriting recognition."
)

if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    demo.launch(debug=True, server_name="0.0.0.0", server_port=7863)
    logger.info("Gradio interface running at http://0.0.0.0:7863") 