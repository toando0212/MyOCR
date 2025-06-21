from doctr.models import detection, recognition

# Liệt kê các mô hình phát hiện văn bản
detection_models = [name for name in dir(detection) if not name.startswith("_")]
print("Các mô hình phát hiện văn bản có sẵn:", detection_models)

# Liệt kê các mô hình nhận dạng văn bản
recognition_models = [name for name in dir(recognition) if not name.startswith("_")]
print("Các mô hình nhận dạng văn bản có sẵn:", recognition_models)