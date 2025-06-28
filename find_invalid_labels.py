import json

LABEL_FILE = 'train_labels_filtered_with_png.json'
INVALID_CHAR = 'ƒ'

with open(LABEL_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Nếu là dict: key là tên ảnh, value là nhãn
if isinstance(data, dict):
    invalid = [(k, v) for k, v in data.items() if INVALID_CHAR in v]
# Nếu là list: mỗi phần tử là [tên ảnh, nhãn]
elif isinstance(data, list):
    invalid = [(item[0], item[1]) for item in data if INVALID_CHAR in item[1]]
else:
    print('Unsupported JSON structure!')
    exit(1)

if not invalid:
    print('No labels contain the invalid character.')
else:
    print(f'Found {len(invalid)} labels containing "{INVALID_CHAR}":')
    for img, label in invalid:
        print(f'Image: {img} | Label: {label}') 