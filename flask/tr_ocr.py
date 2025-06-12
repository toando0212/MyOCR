from openvino.runtime import Core
from transformers import TrOCRProcessor
from optimum.intel.openvino import OVModelForVision2Seq
from PIL import Image, ImageDraw
import cv2
import gradio as gr
import numpy as np
import os
from craft_text_detector import Craft

# Load OpenVINO core and TrOCR model/processor globally
core = Core()
print("Available devices:", core.available_devices)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten", use_fast=True)
model = OVModelForVision2Seq.from_pretrained("microsoft/trocr-small-handwritten", export=True, device="GPU")

# Load CRAFT model globally (reuse for all requests)
craft_detector = Craft(output_dir=None, crop_type='word', cuda=True)

def detect_words_with_craft(pil_img):
    # Save to temp file for CRAFT
    temp_path = "_temp_craft_input.png"
    pil_img.save(temp_path)
    prediction = craft_detector.detect_text(temp_path)
    boxes = prediction["boxes"]
    word_boxes = []
    for box in boxes:
        x_min = int(min(pt[0] for pt in box))
        y_min = int(min(pt[1] for pt in box))
        x_max = int(max(pt[0] for pt in box))
        y_max = int(max(pt[1] for pt in box))
        word_boxes.append([x_min, y_min, x_max, y_max])
    return word_boxes

def group_words_into_lines(word_boxes, y_threshold=50):
    filtered_boxes = [b for b in word_boxes if (b[2] - b[0] > 25 and b[3] - b[1] > 15)]
    if not filtered_boxes:
        return []
    filtered_boxes.sort(key=lambda b: b[1])
    lines = []
    current_line = []
    for box in filtered_boxes:
        if not current_line:
            current_line.append(box)
        else:
            current_y = np.mean([(b[1] + b[3]) // 2 for b in current_line])
            box_y = (box[1] + box[3]) // 2
            if abs(box_y - current_y) < y_threshold:
                current_line.append(box)
            else:
                lines.append(sorted(current_line, key=lambda b: b[0]))
                current_line = [box]
    if current_line:
        lines.append(sorted(current_line, key=lambda b: b[0]))
    return lines

def recognize_lines_with_trocr(pil_img, lines):
    recognized_lines = []
    line_boxes = []
    for line in lines:
        x_min = min(b[0] for b in line)
        y_min = min(b[1] for b in line)
        x_max = max(b[2] for b in line)
        y_max = max(b[3] for b in line)
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(pil_img.width, x_max + padding)
        y_max = min(pil_img.height, y_max + padding)
        crop = pil_img.crop((x_min, y_min, x_max, y_max))
        if np.array(crop).std() < 10:
            continue
        pixel_values = processor(images=crop, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if len(text) < 2:
            continue
        recognized_lines.append(text)
        line_boxes.append((x_min, y_min, x_max, y_max))
    return recognized_lines, line_boxes

def visualize_boxes(pil_img, word_boxes, line_boxes):
    word_img = pil_img.copy()
    draw_word = ImageDraw.Draw(word_img)
    for box in word_boxes:
        draw_word.rectangle([box[0], box[1], box[2], box[3]], outline="blue", width=2)
    line_img = pil_img.copy()
    draw_line = ImageDraw.Draw(line_img)
    for box in line_boxes:
        draw_line.rectangle([box[0], box[1], box[2], box[3]], outline="orange", width=2)
    return word_img, line_img

def ocr_pipeline(pil_img):
    word_boxes = detect_words_with_craft(pil_img)
    lines = group_words_into_lines(word_boxes)
    recognized_lines, line_boxes = recognize_lines_with_trocr(pil_img, lines)
    word_img, line_img = visualize_boxes(pil_img, word_boxes, line_boxes)
    out_text = "\n".join([f"Line {i+1}: {t}" for i, t in enumerate(recognized_lines)])
    return out_text, line_img, word_img

demo = gr.Interface(
    fn=ocr_pipeline,
    inputs=gr.Image(type="pil", label="Upload Document Image"),
    outputs=[
        gr.Textbox(label="Recognized Lines"),
        gr.Image(type="pil", label="Line Bounding Boxes"),
        gr.Image(type="pil", label="Word Bounding Boxes (CRAFT)")
    ],
    title="Unified Line-level OCR with CRAFT Word Detection and TrOCR Recognition",
    description="Detects words using CRAFT, groups them into lines, recognizes each line with TrOCR, and visualizes detected line and word bounding boxes."
)

if __name__ == "__main__":
    demo.launch()