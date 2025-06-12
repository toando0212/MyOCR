# Load processor and model (first time will download and export to OpenVINO IR)
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# model = OVModelForVision2Seq.from_pretrained("microsoft/trocr-base-handwritten", export=True, device="GPU")

# # Load your image (replace 'your_image.png' with your file)
# image = Image.open("/uploads/test2_handwritten_scanned.jpg").convert("RGB")

# # Preprocess
# pixel_values = processor(images=image, return_tensors="pt").pixel_values

# # Run inference
# generated_ids = model.generate(pixel_values)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print("Recognized text:", generated_text)