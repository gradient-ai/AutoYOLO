##WORKING!!!!
##TO FIX:
## - youtube videos
##
##


## Deep learning packages

import torch
from omegaconf import OmegaConf
from torch.nn.functional import normalize
import torch
from transformers import pipeline
import torchvision
from ultralytics import YOLO



# Gradio & HF 
from gradio import processing_utils
import gradio as gr
from huggingface_hub import hf_hub_download



# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from transformers import BlipProcessor, BlipForConditionalGeneration
# segment anything
from segment_anything import build_sam, SamPredictor 
from segment_anything.utils.amg import remove_small_regions


# image libraries
import PIL
from PIL import Image, ImageDraw, ImageFont
import cv2 # returns one box
from exif import Image as eximg
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# helper libraries 
import inflect
from textblob import TextBlob
from gligen.task_grounded_generation import grounded_generation_box, load_ckpt, load_common_ckpt

# essentials 
import pandas as pd
import numpy as np
from io import BytesIO
from sys import platform
import requests
import time
import tqdm
import yaml
import json
from functools import partial
from collections import Counter
import math
import gc
import random
from typing import Optional
import sys
import PIL
import json
import argparse
import nltk
import os
import copy
import yaml
import subprocess
import re
import ast
import os
import warnings
from datetime import datetime
from collections import defaultdict
import subprocess

# Setup: We first need to load in our models and set some relevant variables 

sys.tracebacklimit = 0

hf_hub_download = partial(hf_hub_download, library_name="gligen_demo")

model="../datasets/dolly-v2-12b/dolly-v2-12b/"
generate_text = pipeline(model=model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
config_file = 'groundingdino/config/GroundingDINO_SwinT_OGC.py'
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swint_ogc.pth"
sam_checkpoint='sam_vit_h_4b8939.pth'
device="cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=device)


def make_yaml(name, *cat):
    subprocess.run(['mkdir', f'{name}'])
    lst = list(set([*cat]))
    for i in lst:
        if i == '':
            lst.remove(i)
    nc = len(lst)
    dict1 = {'names': lst,
             'nc': nc, 
             'test': f'test/images',
             'train': f'train/images',
             'val': f'valid/images'}
    with open(f'{name}/data.yaml', 'w') as file:
        documents = yaml.dump(dict1, file)
    return dict1
def make_str(i):
    return str(i)

class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        if x is None:
            return x
        if self.tool == "sketch" and self.source in ["upload", "webcam"] and type(x) != dict:
            decode_image = processing_utils.decode_base64_to_image(x)
            width, height = decode_image.size
            mask = np.zeros((height, width, 4), dtype=np.uint8)
            mask[..., -1] = 255
            mask = self.postprocess(mask)
            x = {'image': x, 'mask': mask}
        return super().preprocess(x)


class Blocks(gr.Blocks):

    def __init__(
        self,
        theme: str = "default",
        analytics_enabled: Optional[bool] = None,
        mode: str = "blocks",
        title: str = "Gradio",
        css: Optional[str] = None,
        **kwargs,
    ):

        self.extra_configs = {
            'thumbnail': kwargs.pop('thumbnail', ''),
            'url': kwargs.pop('url', 'https://gradio.app/'),
            'creator': kwargs.pop('creator', '@teamGradio'),
        }

        super(Blocks, self).__init__(theme, analytics_enabled, mode, title, css, **kwargs)
        warnings.filterwarnings("ignore")

    def get_config_file(self):
        config = super(Blocks, self).get_config_file()

        for k, v in self.extra_configs.items():
            config[k] = v
        
        return config


def draw_box(boxes=[], texts=[], img=None):
    if len(boxes) == 0 and img is None:
        return None

    if img is None:
        img = Image.new('RGB', (512, 512), (255, 255, 255))
    colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("DejaVuSansMono.ttf", size=18)
    for bid, box in enumerate(boxes):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=colors[bid % len(colors)], width=4)
        anno_text = texts[bid]
        draw.rectangle([box[0], box[3] - int(font.size * 1.2), box[0] + int((len(anno_text) + 0.8) * font.size * 0.6), box[3]], outline=colors[bid % len(colors)], fill=colors[bid % len(colors)], width=4)
        draw.text([box[0] + int(font.size * 0.2), box[3] - int(font.size*1.2)], anno_text, font=font, fill=(255,255,255))
    return img

def get_concat(ims):
    if len(ims) == 1:
        n_col = 1
    else:
        n_col = 2
    n_row = math.ceil(len(ims) / 2)
    dst = Image.new('RGB', (ims[0].width * n_col, ims[0].height * n_row), color="white")
    for i, im in enumerate(ims):
        row_id = i // n_col
        col_id = i % n_col
        dst.paste(im, (im.width * col_id, im.height * row_id))
    return dst


def auto_append_grounding(language_instruction, grounding_texts):
    for grounding_text in grounding_texts:
        if grounding_text not in language_instruction and grounding_text != 'auto':
            language_instruction += "; " + grounding_text
    return language_instruction


def slice_per(source, step):
    return [source[i::step] for i in range(step)]


## Manual labeler - automatically generates bounding box labels from drawings over images in Ultralytics YOLOv8 format
def generate(task, dir_name, split, grounding_texts, sketch_pad,
             alpha_sample, guidance_scale, batch_size,
             fix_seed, rand_seed, use_actual_mask, append_grounding, style_cond_image,
             state):
    
    # First, if there is no config YAML file for this dataset, we will generate one using the params
    if os.path.isdir(dir_name) == False:
        try:
            subprocess.run(['mkdir',f'{name}'])
        except:
            None
        subprocess.run(['mkdir',f'datasets/{dir_name}/'])
        subprocess.run(['mkdir',f'datasets/{dir_name}/test'])
        subprocess.run(['mkdir',f'datasets/{dir_name}/test/labels'])
        subprocess.run(['mkdir',f'datasets/{dir_name}/test/images'])
        subprocess.run(['mkdir',f'datasets/{dir_name}/valid'])
        subprocess.run(['mkdir',f'datasets/{dir_name}/valid/labels'])
        subprocess.run(['mkdir',f'datasets/{dir_name}/valid/images'])
        subprocess.run(['mkdir',f'datasets/{dir_name}/train'])
        subprocess.run(['mkdir',f'datasets/{dir_name}/train/labels'])
        subprocess.run(['mkdir',f'datasets/{dir_name}/train/images'])
        subprocess.run(['touch', f'datasets/{dir_name}/data.yaml'])
        # subprocess.run(['cp','-r',f'datasets/{name}/', 'datasets/datasets/'])


        make_yaml(dir_name, *grounding_texts.split(';'))
    num = len(os.listdir(f'datasets/{dir_name}/{split}/images'))
    image = state.get('original_image', sketch_pad['image']).copy()
    image = center_crop(image)
    image = Image.fromarray(image)
    size = image.size
    H, W = size[1], size[0]
    image.save(f'datasets/{dir_name}/{split}/images/{dir_name}-{num}.png')
    if 'boxes' not in state:
        state['boxes'] = []

    boxes = state['boxes']
    grounding_texts = [x.strip() for x in grounding_texts.split(';')]
    # assert len(boxes) == len(grounding_texts)
    if len(boxes) != len(grounding_texts):
        if len(boxes) < len(grounding_texts):
            raise ValueError("""The number of boxes should be equal to the number of grounding objects.
Number of boxes drawn: {}, number of grounding tokens: {}.
Please draw boxes accordingly on the sketch pad.""".format(len(boxes), len(grounding_texts)))
        grounding_texts = grounding_texts + [""] * (len(boxes) - len(grounding_texts))

    boxes2 = []
    for box in boxes:
        x0_ = (box[2]+box[0])/(W*2)
        y0_ = (box[3] + box[1])/(H*2)
        w_ = (box[2] - box[0])/W
        h_ = (box[3] - box[1])/H
        new_box = [x0_, y0_, abs(w_), abs(h_)]
        boxes2.append(new_box)
    grounding_instruction = {}

    grounding_instruction = defaultdict(list)
    for obj,box in zip(grounding_texts, boxes2):
        
        grounding_instruction[obj].append(box)
    g_i = dict(grounding_instruction)
    with open(f'{dir_name}/data.yaml', 'r') as file:
        confi = yaml.safe_load(file)
        with open(f'datasets/{dir_name}/{split}/labels/{dir_name}-{num}.txt', 'w') as f:
            for i in list(g_i.keys()):
                if len(g_i[i])>1:
                    for box in g_i[i]:
                        f.write(f'{confi["names"].index(i)} {" ".join(map(str, box))}')
                        f.write('\n')
                else:
                    f.write(f'{confi["names"].index(i)} {" ".join(map(str, g_i[i][0]))}')
                    f.write('\n')
            
    return image, g_i, state



def train(tr_name, epochs, model_type, batch_size):

    model_dict = {'YOLOv8n':'yolov8n', 'YOLOv8s':'yolov8s', 'YOLOv8m':'yolov8m', 'YOLOv8l':'yolov8l', 'YOLOv8x':'yolov8x'}
    model_type = model_dict[model_type]
    print(tr_name, epochs, model_type)
    model = YOLO(f"{model_type}.yaml")  # build a new model from scratch
    model = YOLO(f"{model_type}.pt")  # load a pretrained model (recommended for training)
    
    model.train(data=f"{tr_name}/data.yaml", epochs=epochs, verbose = True, batch = batch_size, patience = 100)
    
    #     yield pd.read_csv('runs/detect/train28.csv')
    metrics = model.val()  # evaluate model performance on the validation set
    #success = model.export(format="onnx")  # export the model to ONNX format
    return pd.DataFrame.from_dict([metrics.results_dict]), model.trainer.best


def infer(model_path, model_type, img, vid, url):
    print(f" the model path is {model_path}, \n  the model type is {model_type}, \n if there is image it is {img}, \n if there is video it is {vid}, \n from url is {url}")
    model_dict = {'YOLOv8n':'yolov8n', 'YOLOv8s':'yolov8s', 'YOLOv8m':'yolov8m', 'YOLOv8l':'yolov8l', 'YOLOv8x':'yolov8x'}
    model_type = model_dict[model_type]
    model = YOLO(f"{model_type}.yaml")  # build a new model from scratch
    model = YOLO(f"runs/detect/{model_path}/weights/best.pt")  #load your pretrained model (recommended for training)
    if url != '':
        results = model.predict(url)
        lst = []
        for i in range(len(results)):
             res_plotted = results[i].plot()
             lst.append((res_plotted))
        if len(lst) == 1:
            return cv2.cvtColor(lst[0], cv2.COLOR_BGR2RGB), None, results
        frameSize = PIL.Image.fromarray(lst[0]).size
        out = cv2.VideoWriter('output_video.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 120, frameSize)

        for i in lst:
            out.write(i)
        out.release()
        return None, 'output_video.mp4', results
    elif vid != None:
        results = model.predict(vid)
        lst = []
        for i in range(len(results)):
             res_plotted = results[i].plot()
             lst.append((res_plotted))
        frameSize = PIL.Image.fromarray(lst[0]).size
        out = cv2.VideoWriter('output_video.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)

        for i in lst:
            out.write(i)

        out.release()
        return None, 'output_video.mp4', results
    elif img[0][0][0] != None:
        img = Image.fromarray(img)    
        test = model.predict(img)
        test = cv2.cvtColor(test[0].plot(), cv2.COLOR_BGR2RGB)
        return test, None, test
    
def binarize(x):
    return (x != 0).astype('uint8') * 255

def sized_center_crop(img, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    return img[starty:starty+cropy, startx:startx+cropx]

def sized_center_fill(img, fill, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    img[starty:starty+cropy, startx:startx+cropx] = fill
    return img

def sized_center_mask(img, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    center_region = img[starty:starty+cropy, startx:startx+cropx].copy()
    img = (img * 0.2).astype('uint8')
    img[starty:starty+cropy, startx:startx+cropx] = center_region
    return img

def center_crop(img, HW=None, tgt_size=(512, 512)):
    if HW is None:
        H, W = img.shape[:2]
        HW = min(H, W)
    img = sized_center_crop(img, HW, HW)
    img = Image.fromarray(img)
    img = img.resize(tgt_size)
    return np.array(img)

def draw(task, input, grounding_texts, new_image_trigger, state):
    if type(input) == dict:
        image = input['image']
        mask = input['mask']
    else:
        mask = input

    if mask.ndim == 3:
        mask = mask[..., 0]

    image_scale = 1.0

    # resize trigger
    if task == "Grounded Inpainting":
        mask_cond = mask.sum() == 0
        # size_cond = mask.shape != (512, 512)
        if mask_cond and 'original_image' not in state:
            image = Image.fromarray(image)
            width, height = image.size
            scale = 600 / min(width, height)
            image = image.resize((int(width * scale), int(height * scale)))
            state['original_image'] = np.array(image).copy()
            image_scale = float(height / width)
            return [None, new_image_trigger + 1, image_scale, state]
        else:
            original_image = state['original_image']
            H, W = original_image.shape[:2]
            image_scale = float(H / W)

    mask = binarize(mask)
    if mask.shape != (512, 512):
        # assert False, "should not receive any non- 512x512 masks."
        if 'original_image' in state and state['original_image'].shape[:2] == mask.shape:
            mask = center_crop(mask, state['inpaint_hw'])
            image = center_crop(state['original_image'], state['inpaint_hw'])
        else:
            mask = np.zeros((512, 512), dtype=np.uint8)
    # mask = center_crop(mask)
    mask = binarize(mask)

    if type(mask) != np.ndarray:
        mask = np.array(mask)

    if mask.sum() == 0 and task != "Grounded Inpainting":
        state = {}

    if task != 'Grounded Inpainting':
        image = None
    else:
        image = Image.fromarray(image)

    if 'boxes' not in state:
        state['boxes'] = []

    if 'masks' not in state or len(state['masks']) == 0:
        state['masks'] = []
        last_mask = np.zeros_like(mask)
    else:
        last_mask = state['masks'][-1]

    if type(mask) == np.ndarray and mask.size > 1:
        diff_mask = mask - last_mask
    else:
        diff_mask = np.zeros([])

    if diff_mask.sum() > 0:
        x1x2 = np.where(diff_mask.max(0) != 0)[0]
        y1y2 = np.where(diff_mask.max(1) != 0)[0]
        y1, y2 = y1y2.min(), y1y2.max()
        x1, x2 = x1x2.min(), x1x2.max()

        if (x2 - x1 > 5) and (y2 - y1 > 5):
            state['masks'].append(mask.copy())
            state['boxes'].append((x1, y1, x2, y2))

    grounding_texts = [x.strip() for x in grounding_texts.split(';')]
    grounding_texts = [x for x in grounding_texts if len(x) > 0]
    if len(grounding_texts) < len(state['boxes']):
        grounding_texts += [f'Obj. {bid+1}' for bid in range(len(grounding_texts), len(state['boxes']))]

    box_image = draw_box(state['boxes'], grounding_texts, image)

    if box_image is not None and state.get('inpaint_hw', None):
        inpaint_hw = state['inpaint_hw']
        box_image_resize = np.array(box_image.resize((inpaint_hw, inpaint_hw)))
        original_image = state['original_image'].copy()
        box_image = sized_center_fill(original_image, box_image_resize, inpaint_hw, inpaint_hw)

    return [box_image, new_image_trigger, image_scale, state]

def clear(task, sketch_pad_trigger, batch_size, state, switch_task=False):
    if task != 'Grounded Inpainting':
        sketch_pad_trigger = sketch_pad_trigger + 1
    blank_samples = batch_size % 2 if batch_size > 1 else 0
    out_images = [gr.Image.update(value=None, visible=True) for i in range(batch_size)] \
                    + [gr.Image.update(value=None, visible=True) for _ in range(blank_samples)] \
                    + [gr.Image.update(value=None, visible=True) for _ in range(4 - batch_size - blank_samples)]
    state = {}
    return [None, sketch_pad_trigger, None, 1.0] + out_images + [state]

def Dropdown_list():
    new_options =  sorted(os.listdir('datasets/'))
    return gr.Dropdown.update(choices=new_options)
def Dropdown_list2():
    new_options =  sorted(os.listdir('runs/detect'))
    return gr.Dropdown.update(choices=new_options)
def get_model(file_name):
    return f'runs/detect/{file_name}/weights/best.pt'
def on_select(evt: gr.SelectData):  # SelectData is a subclass of EventData   
    return evt.value
def regurg(inp):
    return os.listdir(inp)
def regurg2(evt: gr.SelectData):  # SelectData is a subclass of EventData
    outy = os.listdir(f'datasets/{evt.value}/train/images/')
    lst = []
    for i in outy:
        lst.append(f'datasets/{evt.value}/train/images/'+i)
    return lst
def regurg3(evt: gr.SelectData):
    lst =[]
    for i in os.listdir(f'datasets/{evt.value}/valid/images/'):
        lst.append(f'datasets/{evt.value}/valid/images/'+i)
    return lst
def select_inp_type(choice):
    print(choice)
    if choice =='Video':
        return gr.Image.update(visible = True), gr.Video.update(visible = False), gr.Image.update(visible = True), gr.Video.update(visible = False)
    elif choice =='Image':
        return gr.Image.update(visible = False), gr.Video.update(visible = True), gr.Image.update(visible = False), gr.Video.update(visible = True)
def select_upload_types(choice):
    if choice == 'Single uploads':
        return gr.File.update(visible = True, interactive = True), gr.Dropdown.update(visible = True), gr.Button.update(visible=True)
    elif choice == 'Upload bulk':
        return gr.File.update(visible = False, interactive = True), gr.Dropdown.update(visible = False), gr.Button.update(visible=False)
def refresh_img_select(files):
    return [file.name for file in files]

def fix():
    return gr.Image.update(visible= False), gr.Image.update(visible=False)

def make_yaml_auto(name, lst):
    subprocess.run(['mkdir', f'{name}'])
    subprocess.run(['touch', f'{name}/data.yaml'])
    lst2 = []
    for i in lst[0].split(';'):
        if i != '':
            lst2.append(i)
    # print(lst)
    nc = len(lst2)
    dict1 = {'names': list(sorted(lst2)),
             'nc': nc, 
             'test': f'test/images',
             'train': f'train/images',
             'val': f'valid/images'}
    with open(f'{name}/data.yaml', 'w') as file:
        documents = yaml.dump(dict1, file)
    return dict1

def load_model_hf(model_config_path, repo_id, filename, device='cpu'):
    args = SLConfig.fromfile(model_config_path) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model    

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask
import torchvision.transforms as Tr
def load_image(image_path):
    # # load image
    # image_pil = Image.open(image_path).convert("RGB")  # load image
    image_pil = image_path

    transform = T.Compose(
        [
            T.ToTensor(),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    transform2 = Tr.ToPILImage()
    return transform2(image), image


def unique(list1):
    unique_list = pd.Series(list1).drop_duplicates().tolist()
    return unique_list

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases
        # explicit function to normalize array
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def save_mask_data(f
                   , mask_list, box_list, label_list):
    value = 0  # 0 for background
    output_dir = './outputs'
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    mask_img_path = os.path.join(output_dir, 'mask.jpg')
    plt.savefig(mask_img_path, bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    
    mask_json_path = os.path.join(output_dir, 'mask.json')
    with open(mask_json_path, 'w') as f:
        json.dump(json_data, f)

    return mask_img_path, mask_json_path

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def generate_caption(raw_image):
    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def generate_tags(caption, targets, split=',', max_tokens=100, model="../datasets/dolly-v2-12b/dolly-v2-12b"):
    targets = " ".join(targets)
    prompt = [
        {
            'role': 'system',
            'content': f'Extract the unique nouns in the caption. Do not include adverbs. Remove all the adjectives. No plural words. Replace these words with similar ones from this list: {targets}.' + \
                       f'List the nouns in singular form. Split them by "{split} ". ' + \
                       f'Caption: {caption}.'
        }
    ]
    reply = generate_text(prompt)
    reply = reply[0]
    tags = reply.strip("'[]'")
    tags_o = re.sub('\n', ', ', tags)
    is_noun = lambda pos: pos[:2] == 'NN'
    # do the nlp stuff
    tokenized = nltk.word_tokenize(tags_o)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
    nouns = unique(nouns)
    return ", ".join(nouns)


def check_caption(caption, pred_phrases, max_tokens=100, model="../datasets/dolly-v2-12b/dolly-v2-12b/"):
    object_list = [obj.split('(')[0] for obj in pred_phrases]
    object_num = []
    for obj in set(object_list):
        object_num.append(f'{object_list.count(obj)} {obj}')
    object_num = ', '.join(object_num)
    print(f"Correct object number: {object_num}")

    prompt = [
        {
            'role': 'system',
            'content': 'Revise the number in the caption if it is wrong. ' + \
                       f'Caption: {caption}. ' + \
                       f'True object number: {object_num}. ' + \
                       'Only give the revised caption: '
        }
    ]

    reply = generate_text(prompt)
    # sometimes return with "Caption: xxx, xxx, xxx"
    caption = str(reply).split(':')[-1].strip()
    return caption

def run_grounded_sam(image_path_, targets, name, split, box_threshold, text_threshold, iou_threshold, area_threshold):
    output_dir=f"datasets/{name}/{split}/"
    train_dir = f"datasets/{name}/train/"
    valid_dir = f"datasets/{name}/valid/"
    test_dir = f"datasets/{name}/test/"
    # output_dir=f"../datasets/"
    # make dir
    if output_dir not in os.listdir('datasets'):
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(train_dir+'/labels/', exist_ok=True)
        os.makedirs(train_dir+'/images/', exist_ok=True)
        os.makedirs(train_dir+'/labeled_imgs/', exist_ok=True)
        os.makedirs(train_dir+'/masks/', exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(test_dir+'/labels/', exist_ok=True)
        os.makedirs(test_dir+'/images/', exist_ok=True)
        os.makedirs(test_dir+'/labeled_imgs/', exist_ok=True)
        os.makedirs(test_dir+'/masks/', exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)
        os.makedirs(valid_dir+'/labels/', exist_ok=True)
        os.makedirs(valid_dir+'/images/', exist_ok=True)
        os.makedirs(valid_dir+'/labeled_imgs/', exist_ok=True)
        os.makedirs(valid_dir+'/masks/', exist_ok=True)
    saved = targets
    if type(targets.split(';')) == 'list':
        save_targs = targets.split(';')
        
    else:
        save_targs = [targets.rstrip(';')]
    save_name = name
    save_split = split
    if not os.path.isfile(f'{name}/data.yaml'):    
            make_yaml_auto(name, save_targs)
    count = len(os.listdir(f"datasets/{name}/{split}/images/"))
    # load image
    
    for impath in image_path_:
        # try:
        if 1+1 == 2:
            split = save_split
            name = save_name
            targets = save_targs
            image_path = Image.open(impath.name)

            image_pil, image = load_image(image_path.convert("RGB"))
            # load model
            model = load_model_hf(config_file, ckpt_repo_id, ckpt_filenmae)

            # visualize raw image

            caption = generate_caption(image_pil)
            # Currently ", " is better for detecting single tags
            # while ". " is a little worse in some case
            splity = ','
            with open(f"/notebooks/{name}/data.yaml", 'r') as stream:
                data_loaded = yaml.safe_load(stream)
            # targets = data_loaded['names']
            # tags = generate_tags(caption, targets, split=splity)
            tags = saved.replace(';', ', ')
            print(f'the tags are: {tags}')
            try:
            # run grounding dino model
                boxes_filt, scores, pred_phrases = get_grounding_output(
                    model, image, tags, box_threshold, text_threshold, device=device
                )
            except:
                continue
            # print('working test 1')
            size = image_pil.size

            # initialize SAM
            predictor = SamPredictor(sam)
            image = np.array(image_pil)
            # image = center_crop(image)
            # image = np.array(Image.open(image_path))
            # print('working test 2')
            predictor.set_image(image)
            # print('working test 3')

            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            if len(boxes_filt.size()) == 0:
                continue
                
            # Get boxes in YOLO format
            save_boxes = []
            for box in boxes_filt:
                x0_ = (box[2]+box[0])/(W*2)
                y0_ = (box[3] + box[1])/(H*2)
                w_ = (box[2] - box[0])/W
                h_ = (box[3] - box[1])/H
                new_box = [x0_, y0_, abs(w_), abs(h_)]
                
                save_boxes.append(new_box)
            save_boxes = torch.tensor(save_boxes)
            
            # use NMS to handle overlapped boxes
            # Try-except to catch any images that are corrupted
            try:
                print(f"Before NMS: {boxes_filt.shape[0]} boxes")
                nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
                boxes_filt = boxes_filt[nms_idx]
                pred_phrases = [pred_phrases[idx] for idx in nms_idx]
                print(f"After NMS: {boxes_filt.shape[0]} boxes")
                caption = check_caption(caption, pred_phrases)
                print(f"Revise caption with number: {caption}")
                # print(f'boxes filt = {boxes_filt}')
                # print(f'image is {image}')
            
                transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

                masks, _, _ = predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes,
                    multimask_output = False,
                )
                # area threshold: remove the mask when area < area_thresh (in pixels)
                new_masks = []
                for mask in masks:
                    # reshape to be used in remove_small_regions()
                    mask = mask.cpu().numpy().squeeze()
                    mask, _ = remove_small_regions(mask, area_threshold, mode="holes")
                    mask, _ = remove_small_regions(mask, area_threshold, mode="islands")
                    new_masks.append(torch.as_tensor(mask).unsqueeze(0))

                masks = torch.stack(new_masks, dim=0)
                # masks: [1, 1, 512, 512]
                assert sam_checkpoint, 'sam_checkpoint is not found!'
            except: 
                continue

            # draw output image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box, label in zip(boxes_filt, pred_phrases):
                show_box(box.numpy(), plt.gca(), label)
            plt.axis('off')
            image_path = os.path.join(output_dir, "grounding_dino_output.jpg")
            plt.savefig(image_path, bbox_inches="tight")
            image_result = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            os.remove(os.path.join(output_dir, "grounding_dino_output.jpg"))
            lst = []
            for i in pred_phrases:
                tmpz = i.split('(')[0]
                if len(tmpz.split(' '))>=2:
                    lst.append(tmpz.split(' ')[0])
                else:
                    lst.append(tmpz)
            print(f'the potential caption categories are {lst}')
            lst = list(set(lst))

            dict_tags = dict(zip(list(sorted(save_targs[0].split(';'))), range(len(save_targs[0].split(';')))))

            mask_img_path, _ = save_mask_data('./outputs', masks, boxes_filt, pred_phrases)

            temp_boxes = (np.asarray(save_boxes)).tolist()
            with open(f"/notebooks/datasets/{name}/{split}/labels/input-{count}.txt", 'w') as f:
                image_pil.save(os.path.join(output_dir, f"images/input-{count}.jpg"))
                for k, v in zip(pred_phrases, temp_boxes):
                    # try
                    p = inflect.engine()

                    pos = p.ordinal(count)
                    
                    print(f'the {pos} tag is: {str(dict_tags[k.split("(")[0]])}')
                    print(f'the value checked in targets is {k.split("(")[0]}')
                    if k.split('(')[0] in save_targs[0].split(';'):
                        f.write(f'{str(dict_tags[k.split("(")[0]])} ')
                        # print('key success')
                        f.write(" ".join(map(make_str, v)).strip("'[]'"))
                        # print('val success') 
                        f.write(f'\n') 

                    else: pass

                    # except: pass
            mask_img = cv2.cvtColor(cv2.imread(mask_img_path), cv2.COLOR_BGR2RGB)
            tags_out = tags.strip("'[]'")
            tags_out = ",".join(tags_out.split('\n'))
            tags_out = tags_out.replace('\n', ',')
            tags_out = tags_out.replace("'","")
            # print(tags_out)
            cv2.imwrite(f'/notebooks/datasets/{name}/{split}/labeled_imgs/input-{count}.jpg', image_result)
            cv2.imwrite(f'/notebooks/datasets/{name}/{split}/masks/input-{count}.jpg', mask_img)
            with open(f'/notebooks/datasets/{name}/{split}/labeled_imgs/input-{count}.txt', 'w') as img_file:
                img_file.write(caption.strip("'[]'"))
                img_file.write('\n')
                img_file.write(tags_out)
                img_file.write('\n')
                img_file.write(f'/notebooks/datasets/{name}/{split}/masks/input-{count}.jpg')
            count += 1
        # except:
        else:
            pass

    out_pre= os.listdir(f'/notebooks/datasets/{name}/{split}/labeled_imgs')
    out_pre2 =[]
    for i in out_pre:
        if i.split('.')[1] == 'jpg':
            out_pre2.append(i)
    outs =[]
    for i in out_pre2:
        outs.append(f'/notebooks/datasets/{name}/{split}/labeled_imgs/'+i)

    return outs, None, None, None


def show():
    return gr.Textbox.update(visible = True), gr.Textbox.update(visible = True), gr.Image.update(visible = True)

def refresh_auto_gal_fun(name_dataset, split_auto):
    lst = []
    for i in os.listdir(f'datasets/{name_dataset}/{split_auto}/labeled_imgs/'):
        if i.split('.')[1]!='txt':
            lst.append(f'datasets/{name_dataset}/{split_auto}/labeled_imgs/{i}')
    return lst

def gal_select(evt: gr.SelectData, gallery, name, split):
    path = f'datasets/{name}/{split}/labeled_imgs/'
    im_name = gallery[evt.index]['name'].split('/')[-1]
    im_name = im_name.split('.')[0]+'.txt'
    with open(path+im_name, 'rb') as file1:
        lines = file1.readlines()
    capt = str(lines[0], 'utf-8').lstrip("b'").rstrip("'").replace('\n', ' ').rstrip('\n')
    tags_ = str(lines[1], 'utf-8').lstrip("'b").rstrip("'").replace('\n', ' ').rstrip('\n')
    mask_ = str(lines[2], 'utf-8').lstrip("'b").rstrip("'")
    return capt.split('''\n''')[0], tags_.split('''\n''')[0], mask_.split('''\n''')[0]
css = """
#img2img_image, #img2img_image > .fixed-height, #img2img_image > .fixed-height > div, #img2img_image > .fixed-height > div > img
{
    height: var(--height) !important;
    max-height: var(--height) !important;
    min-height: var(--height) !important;
}
#paper-info a {
    color:#008AD7;
    text-decoration: none;
}
#paper-info a:hover {
    cursor: pointer;
    text-decoration: none;
}
"""

rescale_js = """
function(x) {
    const root = document.querySelector('gradio-app').shadowRoot || document.querySelector('gradio-app');
    let image_scale = parseFloat(root.querySelector('#image_scale input').value) || 1.0;
    const image_width = root.querySelector('#img2img_image').clientWidth;
    const target_height = parseInt(image_width * image_scale);
    document.body.style.setProperty('--height', `${target_height}px`);
    root.querySelectorAll('button.justify-center.rounded')[0].style.display='none';
    root.querySelectorAll('button.justify-center.rounded')[1].style.display='none';
    return x;
}
"""

with Blocks(
    css=css,
    analytics_enabled=False,
    title="YOLOv8 Gradio demo",
) as main:
    description_label = """
    <p style="text-align: center;">
        <span style="font-size: 28px; font-weight: bold;">Manually label images for YOLOv8</span>
        <br>
        This tab allows you to label images with drawings! \n
        The sketchpad will automatically detect and compute the bounding box locations,\n
        and create a labels file with the corresponding label ID and bounding box coordinates. \n
        If no folder exists corresponding to the name input, it will generate a new set of folders and a \n
        `data.yaml` file with the labels enterred in the order submitted. Labels can be repeated, but do not include any spaces.
    </p>
    """
    description_gal = """
    <p style="text-align: center;">
        <span style="font-size: 28px; font-weight: bold;">Image Gallery: View image data after labeling</span>
        <br>
        This tab can be used to view the photos we have labeled.
    </p>
    """
    description_train = """
    <p style="text-align: center;">
        <span style="font-size: 28px; font-weight: bold;">Ultralytics YOLOv8: Train a custom model on your automatically labeled data</span>
        <br>
        Now that we have labeled our images, we can train our model! \n
        Select the model type, batch size, and the number of epochs you would like to train for, and then click the Generate button to run training.  
    </p>
    """
    description_inf = """
    <p style="text-align: center;">
        <span style="font-size: 28px; font-weight: bold;">Ultralytics YOLOv8 prediction: Generate object labels on new images</span>
        <br>
        This tab can be used to run predictions on photos using the model we just trained. 
    </p>
    """
    description = """
    <p style="text-align: center;">
        <span style="font-size: 32px; font-weight: bold;">Supervised AutoLabeling of Object Detection Datasets</span>
    </p>
    """
    description2 = """
    <p style="text-align: center;">
        <span style="font-size: 24px; font-weight: bold;">Using Grounded Segment Anything + Dolly v2 to automatically label images for Ultralytics YOLOv8</span>
    </p>
    """
    description3 = """
    <p style="text-align: center;">
        Fill the field "Targeted annotation labels" with the desired labels seperated by semi-colons. This will automatically generate a 'data.yaml' config file, with the categories, number of categories, and paths populated, and a corresponding set of directories for the training, validation, and test data. 
    </p>
    """
    ###New Section
    with gr.Tab('AutoLabel Images'):
        gr.HTML(description)
        gr.HTML(description2)
        gr.HTML(description3)
        with gr.Row():
            with gr.Column():
                # input_image = gr.Image(source='upload', type="pil")
                input_image = gr.File(label = 'Input images', file_count = 'multiple', visible =True, interactive = True)
                targets = gr.Textbox(label = 'Targeted annotation labels (seperated by semicolon with no spaces)')
                name_dataset = gr.Textbox(label = 'Name of your dataset (adjusts save path)')
                split_auto = gr.Radio(choices = ['train', 'test', 'valid'], label = 'Select which split to place image', value = 'train')
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    box_threshold = gr.Slider(
                        label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.001
                    )
                    text_threshold = gr.Slider(
                        label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                    )
                    iou_threshold = gr.Slider(
                        label="IoU Threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.001
                    )
                    area_threshold = gr.Slider(
                        label="Area Threshold", minimum=0.0, maximum=2500, value=100, step=10
                    )

            with gr.Column():
                with gr.Row():
                    image_caption = gr.Textbox(label="Extracted full image Caption", visible = False)
                    identified_labels = gr.Textbox(label="The possible key objects extracted by Grounded SAM & Dolly V2 12B", visible = False)
                    gallery_auto = gr.Gallery(label = 'Image gallery').style(grid=[5], height="auto")
                    mask_gallery_auto = gr.Image(label = 'Mask image',
                        visible = False, 
                    ).style(full_width=True, full_height=True)
                with gr.Row():
                    refresh_auto_gal = gr.Button('Click to refresh the gallery (JSON parsing error tempfix)')
        with gr.Row():
            gr.Image('assets/logo.png').style(height = 53, width = 125, interactive = False)
    ## return to old
    with gr.Tab("Manually label Images"):
        print('lab')
        gr.HTML(description_label)
        with gr.Row():
            with gr.Column(scale=4):
                sketch_pad_trigger = gr.Number(value=0, visible=False)
                sketch_pad_resize_trigger = gr.Number(value=0, visible=False)
                init_white_trigger = gr.Number(value=0, visible=False)
                image_scale = gr.Number(value=0, elem_id="image_scale", visible=False)
                new_image_trigger = gr.Number(value=0, visible=False)
                dir_name= gr.Textbox(
                    label = 'Name of directory holding files'
                )
                split = gr.Radio(label='Which image split does this image fall into?', choices = ['train','test','valid'], value = 'train')
                task = gr.Radio(
                    choices=["Grounded Generation", 'Grounded Inpainting'],
                    type="value",
                    value="Grounded Inpainting",
                    label="Task", visible = False)
                grounding_instruction = gr.Textbox(label="Targeted annotation labels (seperated by semicolon with no spaces) - Do not fill this field in until you have uploaded an image! If you do, the error can be fixed by clicking 'Clear Sketchpads'")

                select_upload_type = gr.Radio(label='Select upload type',choices = ['Upload bulk', 'Single uploads'], value = 'Single uploads')
                upload_bulk = gr.File(label = 'Input images in bulk. After uploading, click "Refresh image dropdown"',file_count = 'multiple', visible = False, interactive = True)
                select_image = gr.Dropdown(choices = None, label = 'Select image to label. The image will be removed from this list, and passed to the sketchpad on the right of the page.', visible = False)
                refresh_dropdown = gr.Button('Refresh image dropdown', visible = False)
                with gr.Accordion("Advanced Options", open=False, visible = False):
                    with gr.Column():
                        alpha_sample = gr.Slider(minimum=0, maximum=1.0, step=0.1,visible=False, value=0.3, label="Scheduled Sampling (τ)")
                        guidance_scale = gr.Slider(minimum=0, maximum=50, step=0.5, value=7.5, visible=False, label="Guidance Scale")
                        batch_size = gr.Slider(minimum=1, maximum=4, step=1, value=2, label="Number of Samples", visible=False)
                        append_grounding = gr.Checkbox(value=True, label="Append grounding instructions to the caption", visible = False)
                        use_actual_mask = gr.Checkbox(value=False, label="Use actual mask for inpainting", visible=False)
                        with gr.Row():
                            fix_seed = gr.Checkbox(value=True, label="Fixed seed", visible = False)
                            rand_seed = gr.Slider(minimum=0, maximum=1000, step=1, value=0, label="Seed")
                        with gr.Row():
                            use_style_cond = gr.Checkbox(value=False, label="Enable Style Condition", visible = False)
                            style_cond_image = gr.Image(type="pil", label="Style Condition", interactive=True, visible = False)
            with gr.Column(scale=4):
                with gr.Row():
                    sketch_pad = ImageMask(label="Input image", elem_id="img2img_image")
                with gr.Row():    
                    out_imagebox = gr.Image(type="pil", label="Annotated image")
                with gr.Row():
                    clear_btn = gr.Button(value='Clear sketchpads')
                    gen_btn = gr.Button(value='Generate labels')

                with gr.Row():
                    out_gen_1 = gr.Image(label = 'Output image', type="pil", visible=False, show_label=True)
                    out_gen_2 = gr.Textbox(visible = True, label = 'YAML Config in dictionary format')
                with gr.Row():
                    out_gen_3 = gr.Image(type="pil", visible=False, show_label=False)
                    out_gen_4 = gr.Image(type="pil", visible=False, show_label=False)
                    out_gen_5 = gr.Image(type="pil", visible=False, show_label=False)

            state = gr.State({})
            
        with gr.Row():
            gr.Image('assets/logo.png').style(height = 53, width = 125, interactive = False)
        
            
    with gr.Tab('Image Gallery'):
        gr.HTML(description_gal)
        with gr.Column():
            with gr.Row():
                get_img_dir = gr.Dropdown(choices = sorted(os.listdir('datasets/')), label = 'Directory name')
            with gr.Row():
                reload_img_dir = gr.Button('Reload image directories')
                get_gallery = gr.Button('Load images', visible = False)
        with gr.Row():
            gallery = gr.Gallery(value = None, label = 'Training images').style(grid=[10], height="auto")
        with gr.Row():
            val_gallery = gr.Gallery(label = 'Validation images', value = None).style(grid=[10], height="auto")
            
    with gr.Tab('Train YOLOv8'):
        print('train')
        gr.HTML(description_train)
        with gr.Row():
            with gr.Column():
                tr_name = gr.Dropdown(choices = sorted(os.listdir('datasets/')), label = 'Directory name')
                refresh_tr = gr.Button(value = 'Click to refresh dataset list')

            tr_model_type = gr.Radio(label = "Model type", choices = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x'], visible =True)
            with gr.Column():
                epochs = gr.Slider(label = "Number of epochs", minimum= 1, maximum = 500, steps = 10.0, value = 10)
                batch_size = gr.Slider(value = 1, minimum = 1, maximum = 128, step = [1,2,4,8,16,32,64,128], label = 'Batch Size')

        with gr.Column():
            train_btn = gr.Button(value = 'Train')
        with gr.Row():
            prog = gr.Textbox(label = 'Training progress', visible = False)
            df = gr.Dataframe(label = 'Final training metrics', headers = [ 'metrics/precision(B)','metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'fitness'])
        with gr.Row():
            file_name = gr.Dropdown(choices = sorted(os.listdir('runs/detect/')), label = 'Select the model to download', interactive = True)
            file_obj = gr.File(label="Output file")
        with gr.Column():
            load_file = gr.Button('Load file')
        with gr.Row():
            gr.Image('assets/logo.png').style(height = 53, width = 125, interactive = False)
                        
    with gr.Tab('Inference with YOLOv8'):
        print('inf')
        gr.HTML(description_inf)
        with gr.Row():
            with gr.Column():
                model_path = gr.Dropdown(value = 'train', label = 'Path to model', choices = sorted(os.listdir('runs/detect/')))
                refresh_inf = gr.Button(value = 'Click to refresh model list')
                inf_model_type = gr.Radio(value = 'YOLOv8n', label = "Model type", choices = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x'])
            with gr.Column():   
                select_inp = gr.Radio(value = 'Image', choices = ['Image', 'Video'], label = 'Select input type')
                url = gr.Textbox(label = 'URL for video or image', interactive = True, visible = True)
                img = gr.Image(label = 'Input image', interactive = True)
                vid = gr.Video(label = 'Input video', interactive = True, visible = False)
        with gr.Row():
            inf_btn = gr.Button(value = 'Generate label predictions')
        with gr.Row():
            outybox = gr.Textbox(label = 'Full Results output (metrics, boxes, etc.)')
            with gr.Column():
                out_inf_img = gr.Image(label = 'Labeled image', type = 'pil') 
                out_inf_vid = gr.Video(label = 'Labeled video', visible =False)
        with gr.Row():
            gr.Image('assets/logo.png').style(height = 53, width = 125, interactive = False)

        

    class Controller:
        def __init__(self):
            self.calls = 0
            self.tracks = 0
            self.resizes = 0
            self.scales = 0

        def init_white(self, init_white_trigger):
            self.calls += 1
            return np.ones((512, 512), dtype='uint8') * 255, 1.0, init_white_trigger+1

        def change_n_samples(self, n_samples):
            blank_samples = n_samples % 2 if n_samples > 1 else 0
            return [gr.Image.update(visible=True) for _ in range(n_samples + blank_samples)] \
                + [gr.Image.update(visible=False) for _ in range(4 - n_samples - blank_samples)]

        def resize_centercrop(self, state):
            self.resizes += 1
            image = state['original_image'].copy()
            inpaint_hw = int(0.9 * min(*image.shape[:2]))
            state['inpaint_hw'] = inpaint_hw
            image_cc = center_crop(image, inpaint_hw)
            # print(f'resize triggered {self.resizes}', image.shape, '->', image_cc.shape)
            return image_cc, state

        def resize_masked(self, state):
            self.resizes += 1
            image = state['original_image'].copy()
            inpaint_hw = int(0.9 * min(*image.shape[:2]))
            state['inpaint_hw'] = inpaint_hw
            image_mask = sized_center_mask(image, image.shape[1], image.shape[0])
            state['masked_image'] = image_mask.copy()
            # print(f'mask triggered {self.resizes}')
            return image_mask, state

        def switch_task_hide_cond(self, task):
            cond = True
            return gr.Checkbox.update(visible=cond, value=False), gr.Image.update(value=None, visible=False), gr.Slider.update(visible=cond), gr.Checkbox.update(visible=(not cond), value=False)

    controller = Controller()
    main.load(
        lambda x:x,
        inputs=sketch_pad_trigger,
        outputs=sketch_pad_trigger,
        queue=False)
    sketch_pad.edit(
        draw,
        inputs=[task, sketch_pad, grounding_instruction, sketch_pad_resize_trigger, state],
        outputs=[out_imagebox, sketch_pad_resize_trigger, image_scale, state],
        queue=False,
    )
    grounding_instruction.change(
        draw,
        inputs=[task, sketch_pad, grounding_instruction, sketch_pad_resize_trigger, state],
        outputs=[out_imagebox, sketch_pad_resize_trigger, image_scale, state],
        queue=False,
    )
    clear_btn.click(
        clear,
        inputs=[task, sketch_pad_trigger, batch_size, state],
        outputs=[sketch_pad, sketch_pad_trigger, out_imagebox, image_scale, out_gen_1, out_gen_2, out_gen_3, out_gen_4, state],
        queue=False)
    task.change(
        partial(clear, switch_task=True),
        inputs=[task, sketch_pad_trigger, batch_size, state],
        outputs=[sketch_pad, sketch_pad_trigger, out_imagebox, image_scale, out_gen_1, out_gen_2, out_gen_3, out_gen_4, state],
        queue=False)
    sketch_pad_trigger.change(
        controller.init_white,
        inputs=[init_white_trigger],
        outputs=[sketch_pad, image_scale, init_white_trigger],
        queue=False)
    sketch_pad_resize_trigger.change(
        controller.resize_masked,
        inputs=[state],
        outputs=[sketch_pad, state],
        queue=False)
    batch_size.change(
        controller.change_n_samples,
        inputs=[batch_size],
        outputs=[out_gen_1, out_gen_2, out_gen_3, out_gen_4],
        queue=False)
    gen_btn.click(
        generate,
        inputs=[
            task, dir_name, split, grounding_instruction, sketch_pad,
            alpha_sample, guidance_scale, batch_size,
            fix_seed, rand_seed,
            use_actual_mask,
            append_grounding, style_cond_image,
            state
        ],
        outputs=[out_gen_1, out_gen_2, state],
        queue=False
    )
    sketch_pad_resize_trigger.change(
        None,
        None,
        sketch_pad_resize_trigger,
        _js=rescale_js,
        queue=False)
    init_white_trigger.change(
        None,
        None,
        init_white_trigger,
        _js=rescale_js,
        queue=False)
    use_style_cond.change(
        lambda cond: gr.Image.update(visible=cond),
        use_style_cond,
        style_cond_image,
        queue=False)
    task.change(
        controller.switch_task_hide_cond,
        inputs=task,
        outputs=[use_style_cond, style_cond_image, alpha_sample, use_actual_mask],
        queue=False)
    train_btn.click(
        train,
        inputs=[tr_name, epochs, tr_model_type, batch_size],
        outputs=[df, file_obj],
        queue=False)
    inf_btn.click(
        infer,
        inputs = [model_path, inf_model_type, img, vid,url],
        outputs = [out_inf_img, out_inf_vid, outybox],
        queue =False)
    clear_btn.click(fix, inputs = None, outputs = [out_gen_3, out_gen_4])
    refresh_tr.click(Dropdown_list, inputs=None, outputs=tr_name)
    refresh_inf.click(Dropdown_list2, inputs=None, outputs=model_path)
    load_file.click(Dropdown_list2, inputs = None, outputs = file_name)
    select_upload_type.select(select_upload_types, inputs = select_upload_type, outputs = [upload_bulk, select_image, refresh_dropdown])
    load_file.click(get_model,inputs = file_name, outputs= file_obj)
    refresh_dropdown.click(refresh_img_select, inputs = upload_bulk, outputs = select_image)
    reload_img_dir.click(Dropdown_list, inputs = None, outputs = get_img_dir) 
    get_img_dir.select(regurg2, None, outputs = gallery)
    get_img_dir.select(regurg3, None, outputs = val_gallery)
    get_gallery.click(regurg, inputs = get_img_dir, outputs = gallery)
    select_image.select(clear, inputs=[task, sketch_pad_trigger, batch_size, state],
        outputs=[sketch_pad, sketch_pad_trigger, out_imagebox, image_scale, out_gen_1, out_gen_2, out_gen_3, out_gen_4, state],
        queue=False)
    
    select_image.select(on_select, inputs = None, outputs = sketch_pad)
    select_image.select(fix, inputs = None, outputs = [out_gen_3, out_gen_4])
    select_inp.select(select_inp_type, inputs = select_inp, outputs = [img, vid, out_inf_img, out_inf_vid])
    ## new additions
    run_button.click(fn=run_grounded_sam, inputs=[
                        input_image, targets, name_dataset, split_auto, box_threshold, text_threshold, iou_threshold, area_threshold], 
                        outputs=[gallery_auto, mask_gallery_auto, image_caption, identified_labels])
    gallery_auto.select(fn = show, inputs = None, outputs = [image_caption, identified_labels, mask_gallery_auto])
    gallery_auto.select(fn = gal_select, inputs = [gallery_auto, name_dataset, split_auto], outputs = [image_caption, identified_labels, mask_gallery_auto])
    refresh_auto_gal.click(fn = refresh_auto_gal_fun, inputs = [name_dataset, split_auto], outputs = [gallery_auto])

main.queue().launch(share=True, debug = True)
