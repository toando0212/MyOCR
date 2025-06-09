import gradio as gr
import requests
import numpy as np
import cv2

# Set your Flask backend URL here
FLASK_URL = "http://192.168.1.132:5000/classify"  # Change to your server's IP if needed

def classify_via_flask(image, language):
	# Convert PIL image to bytes for upload
	import io
	from PIL import Image
	pil_img = Image.fromarray(image.astype('uint8'), 'RGB') if isinstance(image, np.ndarray) else image
	buf = io.BytesIO()
	pil_img.save(buf, format='PNG')
	buf.seek(0)
	files = {'image': ('image.png', buf, 'image/png')}
	data = {'language': language}
	response = requests.post(FLASK_URL, files=files, data=data)
	if response.status_code != 200:
		return image, "Error", "Error", None
	data = response.json()
	img_np = np.array(pil_img).copy()
	ocr_table = []
	for block in data['results']:
		x, y, w, h = block['box']
		label = block['label']
		idx = block['block_index']
		text = block.get('text', '')
		color = (0, 255, 0) if 'Handwritten' in label else (255, 0, 0)
		cv2.rectangle(img_np, (x, y), (x+w, y+h), color, 2)
		cv2.putText(img_np, f"{idx}:{label.split('_')[0]}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
		ocr_table.append([idx, label, text])
	label_text = "\n".join([f"Block {row[0]}: {row[1]}" for row in ocr_table])
	# For download
	from PIL import Image as PILImage
	pil_annotated = PILImage.fromarray(img_np)
	import tempfile
	temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
	pil_annotated.save(temp_file.name)
	return img_np, label_text, ocr_table, temp_file.name

with gr.Blocks() as demo:
	gr.Markdown("# Prescription Text Block Classifier (Flask Backend)")
	with gr.Row():
		with gr.Column():
			img_input = gr.Image(type="numpy", label="Upload Prescription Image")
			lang_input = gr.Radio(choices=["eng", "vie"], value="eng", label="OCR Language")
			btn = gr.Button("Classify Blocks")
		with gr.Column():
			img_out = gr.Image(type="numpy", label="Annotated Image")
			txt_out = gr.Textbox(label="Block Index → Label Mapping")
			ocr_table = gr.Dataframe(headers=["Block Index", "Label", "OCR Text"], label="OCR Results")
			download_btn = gr.File(label="Download Annotated Image")
	btn.click(classify_via_flask, inputs=[img_input, lang_input], outputs=[img_out, txt_out, ocr_table, download_btn])

if __name__ == "__main__":
	demo.launch()
