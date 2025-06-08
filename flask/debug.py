import os
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'D:\MyOCR\Tesseract-OCR\tesseract.exe'
# import pytesseract
print(pytesseract.get_tesseract_version())


# # You can even comment out the TESSDATA_PREFIX line for this test
# os.environ['TESSDATA_PREFIX'] = r'D:\MyOCR\Tesseract-OCR'

# print("TESSDATA_PREFIX:", os.environ['TESSDATA_PREFIX'])
# print("eng.traineddata exists:", os.path.exists(os.path.join(os.environ['TESSDATA_PREFIX'], 'tessdata', 'vie.traineddata')))

# print(pytesseract.pytesseract.tesseract_cmd)

# print("Tesseract-OCR contents:", os.listdir(os.environ['TESSDATA_PREFIX']))
# print("tessdata contents:", os.listdir(os.path.join(os.environ['TESSDATA_PREFIX'], 'tessdata')))


# img = Image.new('RGB', (100, 30), color = (255, 255, 255))
# print(pytesseract.image_to_string(
#     img, 
#     lang='eng', 
#     config='--tessdata-dir "D:/MyOCR/Tesseract-OCR/tessdata"'
# ))
