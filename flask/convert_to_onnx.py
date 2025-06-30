import torch
from doctr.models import fast_small, vitstr_small, db_mobilenet_v3_large, crnn_mobilenet_v3_large, linknet_resnet18
from doctr.models.utils import export_model_to_onnx
from doctr.datasets import VOCABS

# VOCAB CHUẨN TỪ NOTEBOOK TRAINING
VOCAB = "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!\"#$%&''()*+,-./:;<=>?@[\\]^_`{|}~"


def export_detection_model():
    print("Exporting db_mobilenet_v3_large (detection) model to ONNX...")
    batch_size = 1
    input_shape = (3, 1024, 1024)  # docTR detection models expect 3xHxW
    model = db_mobilenet_v3_large(pretrained=True, exportable=True)
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    model_path = export_model_to_onnx(
        model,
        model_name="db_mobilenet_v3_large",
        dummy_input=dummy_input
    )
    print(f"Detection model exported to: {model_path}")


def export_recognition_model():
    print("Exporting vitstr_small (recognition) model to ONNX...")
    batch_size = 1
    input_shape = (3, 32, 128)  # docTR recognition models expect 3x32x128
    # model = vitstr_small(pretrained=True, exportable=True)
    model = crnn_mobilenet_v3_large(pretrained=True, exportable=True)
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    model_path = export_model_to_onnx(
        model,
        model_name="crnn_mobilenet_v3_large",
        dummy_input=dummy_input
    )
    print(f"Recognition model exported to: {model_path}")


def export_custom_crnn_checkpoint_to_onnx():
    print("Exporting custom CRNN checkpoint to ONNX...")
    batch_size = 1
    input_shape = (3, 32, 128)
    
    # 1. Khởi tạo model với ĐÚNG VOCAB đã train
    model = crnn_mobilenet_v3_large(pretrained=False, exportable=True, vocab=VOCAB)
    
    # 2. Load checkpoint
    checkpoint = torch.load("best_checkpoint_(2).pth", map_location="cpu")
    
    # Lấy đúng state_dict từ bên trong checkpoint
    # Lỗi cho thấy key đúng là 'model_state_dict'
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint: # Fallback cho trường hợp key khác
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict) # Bây giờ có thể dùng strict=True
    
    # 3. Export
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    model_path = export_model_to_onnx(
        model,
        model_name="best_checkpoint_printed3", # Tên file output
        dummy_input=dummy_input
    )
    print(f"Custom CRNN model exported to: {model_path}")


def export_finetuned_english_recog_to_onnx():
    print("Exporting finetuned English recognition model (with French vocab) to ONNX...")
    batch_size = 1
    input_shape = (3, 32, 128)

    # VOCAB tiếng Pháp từ doctr
    # french_vocab = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ"

    # 1. Khởi tạo model với vocab tiếng Pháp
    model = crnn_mobilenet_v3_large(pretrained=False, exportable=True, vocab=VOCABS["french"])

    # 2. Load checkpoint đã finetune tiếng Anh
    checkpoint = torch.load("crnn_eng_hand_print.pth", map_location="cpu")
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    # 3. Export
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    model_path = export_model_to_onnx(
        model,
        model_name="crnn_eng_hand_print",
        dummy_input=dummy_input
    )
    print(f"Finetuned English recognition model (French vocab) exported to: {model_path}")


def main():
    # export_detection_model()
    # export_recognition_model()
    export_custom_crnn_checkpoint_to_onnx()
    # export_finetuned_english_recog_to_onnx()
    # export_custom_fast_onnx()


if __name__ == "__main__":
    main() 