import os

os.environ['TESSDATA_PREFIX'] = os.path.join(os.path.dirname(__file__), '..', 'Tesseract-OCR')
print("TESSDATA_PREFIX:", os.environ['TESSDATA_PREFIX'])
print("eng.traineddata exists:", os.path.exists(os.path.join(os.environ['TESSDATA_PREFIX'], 'tessdata', 'eng.traineddata')))
