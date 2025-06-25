import torch
from doctr.models import fast_small, vitstr_small
from doctr.models.utils import export_model_to_onnx


def export_detection_model():
    print("Exporting fast_small (detection) model to ONNX...")
    batch_size = 1
    input_shape = (3, 512, 512)  # docTR detection models expect 3xHxW
    model = fast_small(pretrained=True, exportable=True)
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    model_path = export_model_to_onnx(
        model,
        model_name="fast_small.onnx",
        dummy_input=dummy_input
    )
    print(f"Detection model exported to: {model_path}")


def export_recognition_model():
    print("Exporting vitstr_small (recognition) model to ONNX...")
    batch_size = 1
    input_shape = (3, 32, 128)  # docTR recognition models expect 3x32x128
    model = vitstr_small(pretrained=True, exportable=True)
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    model_path = export_model_to_onnx(
        model,
        model_name="vitstr_small.onnx",
        dummy_input=dummy_input
    )
    print(f"Recognition model exported to: {model_path}")


def main():
    export_detection_model()
    export_recognition_model()


if __name__ == "__main__":
    main() 