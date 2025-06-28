import csv
import json

# Đọc file CSV
labels = {}
with open('InkData_word.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Giả sử cột tên là 'image' và 'label'
        labels[row['id']] = row['label']

# Ghi ra file json theo format doctr
with open('train_labels_doctr.json', 'w', encoding='utf-8') as f:
    json.dump(labels, f, ensure_ascii=False, indent=2) 
