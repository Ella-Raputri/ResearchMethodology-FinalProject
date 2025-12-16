from ultralytics import YOLO
import os
from PIL import Image
import glob
import json
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
from .ocr import scan_image

caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

translator = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-id")
translator_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-id")

def generate_caption(img_url, img_obj):
    ymin, ymax, xmin, xmax = img_obj["y_min"], img_obj["y_max"], img_obj["x_min"], img_obj["x_max"]
    raw_image = Image.open(img_url).convert('RGB')
    crop = raw_image.crop((xmin, ymin, xmax, ymax))

    text = "an image of"
    inputs = caption_processor(crop, text, return_tensors="pt")

    out = caption_model.generate(**inputs)
    caption_en = caption_processor.decode(out[0], skip_special_tokens=True)

    translated = translator.generate(**translator_tokenizer(caption_en, return_tensors="pt", padding=True))
    caption_id = translator_tokenizer.decode(translated[0], skip_special_tokens=True)
    # print(caption_id, caption_en, sep='\n')
    return caption_id

def get_detected_result(image_path, label_path):
    detected = {i : list() for i in range(0, 3)}
    img = Image.open(image_path)
    img_width, img_height = img.size

    with open(label_path, 'r') as f:
        lines = f.readlines()

    # print(f"Image: {image_name}")
    for line in lines:
        # print(line, end='')
        label, x_center, y_center, w, h, confidence = map(float, line.strip().split())

        x_min = int((x_center - w / 2) * img_width)
        y_min = int((y_center - h / 2) * img_height)
        x_max = int((x_center + w / 2) * img_width)
        y_max = int((y_center + h / 2) * img_height)

        obj = dict()
        obj['x_min'] = x_min
        obj['x_max'] = x_max
        obj['y_min'] = y_min
        obj['y_max'] = y_max
        obj['conf'] = confidence
        detected[label].append(obj)
    
    return detected

def intersect_rect(image_obj, x_min2, x_max2, y_min2, y_max2):
    tolerance = 2
    x_min1, x_max1 = image_obj['x_min']+tolerance, image_obj['x_max']+tolerance
    y_min1, y_max1 = image_obj['y_min']+tolerance, image_obj['y_max']+tolerance

    return not ((x_max1 <= x_min2 or x_max2 <= x_min1) or
                (y_max1 <= y_min2 or y_max2 <= y_min1))

def clean_text(ocr_result, detected, caption_pos):
    new_txt = ""
    max_idx = len(ocr_result['text'])

    # tdk append text dalam image yg kedetect ocr
    for i in range(max_idx):
        text = ocr_result['text'][i]
        if i in caption_pos: new_txt += text + " "
        
        is_exclude = False
        exclude_list = detected[1] + detected[2] #1:pagenumber, 2:picture

        for obj in exclude_list:
            # print('masyk')
            x_min2 = ocr_result['left'][i]
            x_max2 = ocr_result['left'][i]+ocr_result['width'][i]
            y_min2 = ocr_result['top'][i]
            y_max2 = ocr_result['top'][i]+ocr_result['height'][i]

            if(intersect_rect(obj, x_min2, x_max2, y_min2, y_max2)): 
                is_exclude = True
                break
        
        if not is_exclude and text != "": new_txt += text + " "
    
    return new_txt.strip()

def match_caption_img(detected):
    if len(detected[0]) != 0: #kalau ada caption
        return []
    
    caption_img = [] #{caption: no urut caption, img: no urut img}: dict
    img_used = {} #utk track img apakah udah dipakai atau belum

    # loop greedy matching caption-img
    for id, caption in enumerate(detected[0]):
        y_min_c, y_max_c = caption['y_min'], caption['y_max']
        min_val = 1e5
        ambil_idx = -1
        ambil_obj = None

        for idx, img in enumerate(detected[2]):
            y_min_img, y_max_img = img['y_min'], img['y_max']

            # caption di atas img
            diff_up = abs(y_min_img-y_max_c)
            # kalau caption di bawah img
            diff_down = abs(y_min_c - y_max_img)
            
            if(diff_up < diff_down): 
                if diff_up < min_val: 
                    min_val = diff_up
                    ambil_idx = idx
                    ambil_obj = img
            else: 
                if diff_down < min_val: 
                    min_val = diff_down
                    ambil_idx = idx
                    ambil_obj = img
        
        # print(id, min_val, ambil_idx)

        #caption lebih byk dari img
        if(ambil_idx in img_used):
            if(min_val < img_used[ambil_idx]['value']):
                prev_caption_idx = img_used[ambil_idx]['caption']
                caption_img[prev_caption_idx]['img'] = None
                caption_img[prev_caption_idx]['img_obj'] = None
        
        caption_img.append({"caption":id, "img":ambil_idx, "img_obj":ambil_obj}) 
        img_used[ambil_idx] = {"caption":id, "value":min_val}

    # img lbih byk dri caption
    for image_id, image in enumerate(detected[2]):
        if image_id not in img_used:
            caption_img.append({"caption":None, "img":image_id, "img_obj":image}) 

    # cleaning, utamain img
    caption_img = [record for record in caption_img if record["img"] is not None]
    return caption_img

def append_caption(record, ocr_data, excludes):
    caption_x = (record['img_obj']['x_min'] + record['img_obj']['x_max']) / 2
    caption_y = (record['img_obj']['y_min'] + record['img_obj']['y_max']) / 2
    caption_str = record['caption']

    min_dist = 1e5
    insert_index = -1

    for i in range(len(ocr_data['top'])):
        if(ocr_data['left'][i] > caption_x) or (ocr_data['top'][i] > caption_y): continue
        if i in excludes: continue
        
        # hitung manhattan
        manhattan = abs(ocr_data['top'][i]-caption_y) + abs(ocr_data['left'][i]-caption_x)
        if (manhattan <= min_dist):
            min_dist = manhattan
            insert_index = i+1

    excludes.append(insert_index)
    ocr_data['left'].insert(insert_index, record['img_obj']['x_max']+1)
    ocr_data['top'].insert(insert_index, record['img_obj']['y_max']+1)

    ocr_data['width'].insert(insert_index, abs(record['img_obj']['x_max']-record['img_obj']['x_min']))
    ocr_data['height'].insert(insert_index, abs(record['img_obj']['y_max']-record['img_obj']['y_min']))
    ocr_data['text'].insert(insert_index, caption_str)

    return ocr_data, excludes

def find_label_path(source_name):
    pattern = "detect_result/predict*/labels"
    label_dirs = sorted(glob.glob(pattern))  

    for label_dir in label_dirs:
        label_path = os.path.join(label_dir, f"{source_name}.txt")
        if os.path.exists(label_path):
            return label_path

    return None

def yolo_clean(source_file):
    model = YOLO("../yolo/models/kfold_result/kfold_training/fold_4/weights/best.pt")
    results = model.predict(
        source=source_file, save=True, save_txt=True, save_conf=True,
        project="detect_result",
    )

    source_name = source_file.split('\\')[-1].split('.')[0]
    print('src name ', source_name)
    
    txt, ocr_data = scan_image(source_file)
    label_path = find_label_path(source_name)

    if label_path is None:
        print(f"No image, caption, or header-footer in {source_file}")
        str = " ".join(ocr_data['text'])
        return str
        
    detected_objs = get_detected_result(source_file, label_path)
    matches = match_caption_img(detected_objs)
    excludes = []

    for record in matches:
        if record['img_obj'] is not None and not record['caption']:
            new_caption = generate_caption(source_file, record['img_obj'])
            # ubah caption-img (matches) ke dict 
            # add caption ke cleaned text
            record['caption'] = new_caption
            ocr_data, excludes = append_caption(record, ocr_data, excludes)

    cleaned = clean_text(ocr_data, detected_objs, excludes)
    return cleaned
        