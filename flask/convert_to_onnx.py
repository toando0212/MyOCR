import torch
from doctr.models import fast_small, vitstr_small, db_mobilenet_v3_large, crnn_mobilenet_v3_large, linknet_resnet18
from doctr.models.utils import export_model_to_onnx
from doctr.datasets import VOCABS
import torch.nn as nn
import torch.nn.functional as F

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
    base_model = crnn_mobilenet_v3_large(pretrained=False, exportable=True, vocab=VOCAB)
    model = CRNNWithDomainHead(base_model)
    checkpoint = torch.load("best_checkpoint_epoch153_grl.pth", map_location="cpu")
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    # Export only the logits output for ONNX (if you want both, adjust accordingly)
    torch.onnx.export(
        model, dummy_input, "best_checkpoint_epoch153_grl.onnx",
        input_names=["input"], output_names=["output"],
        opset_version=11
    )
    print("Custom CRNN model exported to: best_checkpoint_printed3.onnx")


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


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class CRNNWithDomainHead(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.feat_extractor = base_model.feat_extractor
        self.decoder = base_model.decoder
        self.linear = base_model.linear
        self.domain_grl = GradientReversal(0.0)
        with torch.no_grad():
            dummy_input = torch.zeros(1, *base_model.cfg['input_shape'])
            feat_out = self.feat_extractor(dummy_input)
            feat_channels = feat_out.shape[1]
        self.domain_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feat_channels, 2)
        )
        self.vocab = base_model.vocab
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}

    def forward(self, x):
        features = self.feat_extractor(x)
        reversed_feat = self.domain_grl(features)
        domain_logits = self.domain_head(reversed_feat)
        b, c, h, w = features.shape
        features_seq = features.reshape(b, c * h, w)
        features_seq = torch.transpose(features_seq, 1, 2)
        decoded_features, _ = self.decoder(features_seq)
        ocr_logits = self.linear(decoded_features)
        return {"domain_logits": domain_logits, "logits": ocr_logits}


def main():
    # export_detection_model()
    # export_recognition_model()
    export_custom_crnn_checkpoint_to_onnx()
    # export_finetuned_english_recog_to_onnx()
    # export_custom_fast_onnx()


if __name__ == "__main__":
    main()