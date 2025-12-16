import os
from PIL import Image
import pytesseract
from tqdm import tqdm

# change to your path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def scan_image(path):
    image = Image.open(path)
    txt = pytesseract.image_to_string(image)
    ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    return txt, ocr_result