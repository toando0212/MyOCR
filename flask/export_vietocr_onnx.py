import torch
from vietocr.model.vocab import Vocab
from vietocr.model.config import get_config
from vietocr.model.architecture import CNNLSTM
import onnx

# ---- CẤU HÌNH ----
config = get_config('vgg_seq2seq')
vocab = Vocab(config['vocab'])

# Load mô hình
model = CNNLSTM(config)
model.load_state_dict(torch.load('weights/seq2seq.pth', map_location='cpu'))
model.eval()

# ---- INPUT DUMMY ----
dummy_input = torch.randn(1, 3, 32, 128)

# ---- EXPORT SANG ONNX ----
torch.onnx.export(
    model,
    dummy_input,
    "vietocr_vgg_seq2seq.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        'input': {0: 'batch_size', 3: 'width'},
        'logits': {0: 'batch_size', 1: 'seq_len'}
    },
    export_params=True,
    opset_version=11,
    do_constant_folding=True
)

print("✅ Convert thành công: vietocr_vgg_seq2seq.onnx")
