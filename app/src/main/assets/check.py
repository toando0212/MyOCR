import onnxruntime as ort
import numpy as np

img = torch.rand(1, 3, 32, 475)  # batch_size=1, 3 channels, height=32, width=475
img = img.to(config['device'])
# CNN
src = model.cnn(img)
print("src (output CNN):", src.shape)

# Encoder
encoder_outputs, hidden = model.transformer.encoder(src)
print("encoder_outputs:", encoder_outputs.shape)
print("hidden:", hidden.shape)

# tgt: thông thường là (seq_len, batch) hoặc (batch, seq_len)
tgt = torch.LongTensor([[1]])  # batch_size=1, seq_len=1
print("tgt:", tgt.shape)

# Decoder
out = model.transformer.decoder(tgt, hidden, encoder_outputs)
print("decoder output:", [o.shape if hasattr(o, 'shape') else type(o) for o in (out if isinstance(out, (tuple, list)) else [out])])