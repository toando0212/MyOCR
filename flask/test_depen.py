import os
import torch # Import torch for saving model weights
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

config = Cfg.load_config_from_name('vgg_seq2seq')
# config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = 'cpu' # Force CPU usage
config['cnn']['pretrained'] = False # Ensure CNN backbone is not looking for pre-trained weights tied to training data

# Load the model using Predictor
predictor = Predictor(config)
model = predictor.model

total_params = sum(p.numel() for p in model.parameters())
print(f"Tổng số tham số: {total_params}")

# Get model weight size
temp_weights_path = "temp_model_weights.pth"
torch.save(model.state_dict(), temp_weights_path)

weight_size_bytes = os.path.getsize(temp_weights_path)
weight_size_mb = weight_size_bytes / (1024 * 1024)

print(f"Kích thước trọng số mô hình: {weight_size_mb:.2f} MB")

os.remove(temp_weights_path) # Clean up the temporary file