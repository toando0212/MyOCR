from doctr.models import detection, recognition

# Liệt kê các mô hình phát hiện văn bản
detection_models = detection.__all__
print("Các mô hình phát hiện văn bản có sẵn:", detection_models)

# Liệt kê các mô hình nhận dạng văn bản
recognition_models = recognition.__all__
print("Các mô hình nhận dạng văn bản có sẵn:", recognition_models)