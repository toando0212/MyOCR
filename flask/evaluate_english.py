import os
import pandas as pd
from PIL import Image
import time
import logging
import sys
import json

# Ensure the script's directory is in the python path to allow importing local modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Suppress verbose INFO logging from underlying libraries during model loading
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('doctr').setLevel(logging.WARNING)

# Now, import the correct pipeline.
from english_pipeline import ocr_pipeline

# Setup basic logging for the evaluation script itself
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_levenshtein(s1, s2):
    """Calculates the Levenshtein distance between two sequences."""
    if len(s1) < len(s2):
        return calculate_levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_cer(ground_truth, prediction):
    """Calculates the Character Error Rate (CER) after normalizing strings."""
    # Normalize by making lowercase and removing extra whitespace
    gt_norm = ' '.join(ground_truth.lower().split())
    pred_norm = ' '.join(prediction.lower().split())
    
    distance = calculate_levenshtein(gt_norm, pred_norm)
    if len(gt_norm) == 0:
        return 1.0 if len(pred_norm) > 0 else 0.0
    return distance / len(gt_norm)

def calculate_wer(ground_truth, prediction):
    """Calculates the Word Error Rate (WER) after normalizing strings."""
    # Normalize by making lowercase and splitting into words
    gt_words = ground_truth.lower().split()
    pred_words = prediction.lower().split()

    distance = calculate_levenshtein(gt_words, pred_words)
    if len(gt_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0
    return distance / len(gt_words)

def load_ground_truth_from_json(json_file_path):
    """Loads text data from FUNSD JSON annotation file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Extract text from the 'form' field
        texts = []
        for item in data.get('form', []):
            text = item.get('text', '').strip()
            if text:
                texts.append(text)
        return ' '.join(texts)
    except Exception as e:
        logging.error(f"Error loading JSON file {json_file_path}: {e}")
        return ""

def evaluate_pipeline():
    """
    Evaluates the english_pipeline on the first 50 images of the FUNSD dataset.
    """
    # --- Configuration ---
    IMAGE_DIR = 'D:/MyOCR/testing_data_FUNSD/images'
    ANNOTATION_DIR = 'D:/MyOCR/testing_data_FUNSD/annotations'
    NUM_IMAGES_TO_TEST = 20

    # --- Load Ground Truth Data ---
    logging.info(f"Loading ground truth data from {ANNOTATION_DIR}")
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')])[:NUM_IMAGES_TO_TEST]
    results = []

    logging.info(f"Starting evaluation on the first {NUM_IMAGES_TO_TEST} images...")

    for filename in image_files:
        image_path = os.path.join(IMAGE_DIR, filename)
        json_filename = filename.replace('.png', '.json')
        json_file_path = os.path.join(ANNOTATION_DIR, json_filename)

        if not os.path.exists(image_path):
            logging.warning(f"Image file not found: {image_path}. Skipping.")
            continue

        if not os.path.exists(json_file_path):
            logging.warning(f"Annotation file not found: {json_file_path}. Skipping.")
            continue

        ground_truth = load_ground_truth_from_json(json_file_path)

        try:
            # --- Run OCR Pipeline ---
            image = Image.open(image_path)
            
            start_time = time.time()
            # The english_pipeline returns 5 values
            predicted_text, _, _, _, _ = ocr_pipeline(image)
            end_time = time.time()
            
            processing_time = end_time - start_time

            # The output format is "[LABEL] text", so we parse it accordingly.
            predicted_lines = []
            if predicted_text and isinstance(predicted_text, str):
                for line in predicted_text.split('\n'):
                    if ']' in line:
                        predicted_lines.append(line.split(']', 1)[-1].strip())
            clean_prediction = " ".join(predicted_lines)

            # --- Calculate Metrics ---
            if not isinstance(ground_truth, str):
                ground_truth = "" # Handle potential non-string data
            
            # --- For transparency, log the strings being compared ---
            logging.info(f"Comparing strings for {filename}:")
            logging.info(f"  - GROUND TRUTH: '{ground_truth}'")
            logging.info(f"  - PREDICTION:   '{clean_prediction}'")

            cer = calculate_cer(ground_truth, clean_prediction)
            wer = calculate_wer(ground_truth, clean_prediction)

            results.append({
                'filename': filename,
                'ground_truth': ground_truth,
                'prediction': clean_prediction,
                'cer': cer,
                'wer': wer,
                'time': processing_time
            })
            
            logging.info(f"Processed {filename} | CER: {cer:.4f}, WER: {wer:.4f}\n")

        except Exception as e:
            logging.error(f"An error occurred while processing {filename}: {e}", exc_info=True)

    # --- Aggregate and Display Results ---
    if not results:
        logging.error("Evaluation could not be completed for any image.")
        return

    results_df = pd.DataFrame(results)
    
    avg_cer = results_df['cer'].mean()
    avg_wer = results_df['wer'].mean()
    avg_time = results_df['time'].mean()

    print("\n" + "="*50)
    print("      EVALUATION SUMMARY (english_pipeline on FUNSD)")
    print("="*50)
    print(f"Images Evaluated:      {len(results_df)}")
    print(f"Average CER:           {avg_cer:.4f}")
    print(f"Average WER:           {avg_wer:.4f}")
    print(f"Average Processing Time: {avg_time:.2f} seconds/image")
    print("="*50 + "\n")

    print("Detailed Results:")
    print(results_df[['filename', 'cer', 'wer', 'time']].to_string(index=False))

if __name__ == "__main__":
    evaluate_pipeline() 