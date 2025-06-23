# -*- coding: utf-8 -*-
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
logger = logging.getLogger("be4_post_detect")

logger.info("Starting the 'BEFORE MERGE' OCR pipeline application...")

try:
    # Load VietOCR model for recognition
    logger.info("Loading VietOCR model (vgg_seq2seq)...")
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cpu' # Force CPU for compatibility
    config['predictor']['beamsearch'] = False
    vietocr_predictor = Predictor(config)
    logger.info("VietOCR model loaded successfully!")

    # Load Doctr predictor for line detection only
    logger.info("Initializing DocTR predictor for line detection...")
    doctr_predictor = ocr_predictor(det_arch='fast_small', pretrained=True)
    logger.info("DocTR predictor loaded successfully!")

except Exception as e:
    logger.error(f"Error during model initialization: {str(e)}", exc_info=True)
    raise

def post_process_text(text: str) -> str:
    """
    Correctly segments Vietnamese words using pyvi.
    """
    logger.info("Starting post-processing (Vietnamese word segmentation)...")
    try:
        tokenized_text = ViTokenizer.tokenize(text)
        readable_text = tokenized_text.replace('_', ' ')
        logger.info("Post-processing completed.")
        return readable_text
    except Exception as e:
        logger.error(f"Error during post-processing: {e}", exc_info=True)
        return text

@torch.no_grad()
def ocr_pipeline_before_merge(pil_img):
    logger.info("OCR pipeline (BEFORE MERGE) started.")
    if pil_img is None:
        return "Please upload an image.", "", None, None, None, None

    try:
        # Step 1-3: Preprocessing (Same as the main pipeline)
        logger.info("Step 1-3: Correcting perspective, deskewing, and removing lines...")
        corrected_img = correct_perspective(pil_img)
        deskewed_img = deskew_image(corrected_img)
        processed_img_for_detection = remove_horizontal_lines(deskewed_img)
        
        # Step 4: Line Detection using DocTR
        logger.info("Step 4: Running DocTR to detect raw lines...")
        img_array = np.array(processed_img_for_detection.convert("RGB"))
        result = doctr_predictor([img_array])
        
        if not result.pages or not result.pages[0].blocks:
            logger.warning("No text boxes detected by DocTR.")
            return "No text boxes detected.", "", corrected_img, deskewed_img, processed_img_for_detection, deskewed_img.copy().convert("RGB")

        # Step 4a: Extract all detected boxes and convert to ABSOLUTE coordinates
        # THIS IS THE KEY DIFFERENCE: We DO NOT MERGE the boxes.
        logger.info("Step 4a: Using raw detected boxes from DocTR (NO MERGING)...")
        page = result.pages[0]
        page_dims = page.dimensions # (height, width)
        page_height, page_width = page_dims
        
        unmerged_line_boxes = []
        for block in page.blocks:
            for line in block.lines:
                # Convert relative coordinates to absolute
                xmin, ymin = int(line.geometry[0][0] * page_width), int(line.geometry[0][1] * page_height)
                xmax, ymax = int(line.geometry[1][0] * page_width), int(line.geometry[1][1] * page_height)
                unmerged_line_boxes.append((xmin, ymin, xmax, ymax))

        if not unmerged_line_boxes:
            logger.warning("DocTR did not detect any lines. OCR will be empty.")
            return "DocTR detected no lines.", "", corrected_img, deskewed_img, processed_img_for_detection, deskewed_img.copy().convert("RGB")

        # Step 5: Text Recognition using VietOCR on UNMERGED lines
        logger.info("Step 5: Recognizing text with VietOCR on raw (unmerged) lines...")
        recognized_results = []
        
        for line_box in unmerged_line_boxes:
            line_crop = deskewed_img.crop(line_box)
            if line_crop.width > 0 and line_crop.height > 0:
                try:
                    line_text = vietocr_predictor.predict(line_crop)
                except Exception as recog_e:
                    logger.warning(f"VietOCR failed to recognize a line crop: {recog_e}")
                    line_text = ""
            else:
                line_text = ""
            if not line_text: continue
            recognized_results.append({'box': line_box, 'text': line_text})
            logger.info(f"Recognized (Raw) Line: {line_text}")
        
        # Step 6: Create visualization
        logger.info("Step 6: Creating visualization...")
        vis_img = deskewed_img.copy().convert("RGB")
        draw = ImageDraw.Draw(vis_img)
        
        for res in recognized_results:
            box, text = res['box'], res['text']
            draw.rectangle(box, outline="red", width=2) # Use RED for unmerged boxes
            text_position = (box[0], box[1] - 15)
            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                font = ImageFont.load_default()
            draw.text(text_position, text, fill="red", font=font)
        
        out_text = "\n".join([r['text'] for r in recognized_results])
        logger.info("OCR pipeline (BEFORE MERGE) completed successfully.")
        
        # Step 7: Post-processing
        post_processed_text = post_process_text(out_text)
        
        return out_text, post_processed_text, corrected_img, deskewed_img, processed_img_for_detection, vis_img

    except Exception as e:
        error_msg = f"Error in OCR pipeline: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg, "", None, None, None, pil_img

# Create Gradio interface for the "Before Merge" pipeline
demo = gr.Interface(
    fn=ocr_pipeline_before_merge,
    inputs=gr.Image(type="pil", label="Upload Document Image"),
    outputs=[
        gr.Textbox(label="1. Raw Recognized Text"),
        gr.Textbox(label="2. Post-Processed Text (Word Segmented)"),
        gr.Image(type="pil", label="3. Perspective Corrected"),
        gr.Image(type="pil", label="4. Deskewed"),
        gr.Image(type="pil", label="5. Preprocessed (Lines Removed)"),
        gr.Image(type="pil", label="6. Final Result (RAW Unmerged Boxes in RED)")
    ],
    title="[FOR REPORT] Vietnamese OCR - BEFORE BBox Merging",
    description="This pipeline visualizes the raw output of the DocTR line detector **WITHOUT** the custom handwriting box merging step. This is used for comparison and reporting."
)

if __name__ == "__main__":
    logger.info("Starting Gradio interface for the 'BEFORE MERGE' pipeline...")
    demo.launch(server_port=7865)
    logger.info("Gradio interface running.") 