import argparse
import os
import sys
import time
import numpy as np
import json
import torch
from PIL import Image
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (sam_hq_model_registry,
    sam_model_registry,SamPredictor)

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    width, height = image_pil.size
    if width < 256 or height < 256:
        new_size = (width * 2, height * 2)
    else:
        new_size = (width,height)
    image_pil = image_pil.resize(new_size, Image.LANCZOS)
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


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
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([0/255, 0/255, 0/255, 1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)

def process_string(string):
    string = string.replace('/', '=')
    string = string.replace('__', '_')
    string = string.replace('_','=')
    string = string.replace('==','=')
    string = string.replace('=','/')
    string = string.replace('//','/')
    return string
    
def save_mask_data(output_dir, mask_list, box_list, label_list, target, excel_row, runs):
    value = 0
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
        
    value = 0  # 0 for background
    
    string = process_string(output_dir)
    image_path = string+'/fig.png'
    img2 = Image.open(image_path).convert("RGBA") 
    image_path = string+'/mask.png'
    img1 = Image.open(image_path).convert("RGBA")
    
    size1 = img1.size
    size2 = img2.size

    if size1 != size2:
        if size1[0] * size1[1] > size2[0] * size2[1]:
            img2 = img2.resize(size1, Image.LANCZOS)
        else:
            img1 = img1.resize(size2, Image.LANCZOS)
    data1 = np.array(img1)
    data2 = np.array(img2)
    # 创建一个布尔数组，判断哪些像素是白色的
    white_pixels = (data1[:, :, 0] > 200) & (data1[:, :, 1] > 200) & (data1[:, :, 2] > 200)
    black_pixels = (data1[:, :, 0] < 20) & (data1[:, :, 1] < 20) & (data1[:, :, 2] < 20)
    
    swapped_data = np.copy(data1)
    swapped_data[black_pixels] = [255, 255, 255, 255]
    swapped_data[white_pixels] = [0, 0, 0, 255]
    swapped_img = Image.fromarray(swapped_data,'RGBA')
    swapped_img.save(image_path)
    data2[white_pixels] = [0,0,0,0]

    new_img = Image.fromarray(data2,'RGBA')

    string = f'{output_dir}/processed.png'
    string = string.replace('/', '=')
    string = string.replace('__', '_')
    string = string.replace('_','=')
    string = string.replace('==','=')
    string2 = string.replace('=','/')
    string2 = string2.replace('//','/')
    try:
        new_img.save(string2)
        excel_row[4] = 1
        excel_dir = string2.replace(f'output/run{runs + 1}/','')
        excel_dir = excel_dir.replace('/processed.png','')
        excel_row[1] = excel_dir
    except:
        pass

    os.makedirs(f'zest_code/demo_assets/run{runs + 1}', exist_ok = True)

    if target:
      target_dir = 'zest_code/demo_assets/input_imgs'
      new_img.save(target_dir + f'/{string}')
      target_dir = 'zest_code/DPT/input'
      new_img.save(target_dir + f'/{string}')
    else :
      target_dir = 'zest_code/demo_assets'  
      target = f'zest_code/demo_assets/run{runs + 1}'
      string = string.replace('ousput_','')
      string = string.replace('_processed','')
      string = string.replace(f'run{runs + 1}=','')  
      #print(string)  
      new_img.save(target + f'/{string}')  
      excel_row[5] = 1  
    
    return excel_row

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)

    parser.add_argument("--config", type=str,default = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="path to config file")

    parser.add_argument(
        "--grounded_checkpoint", type=str, default="./groundingdino_swint_ogc.pth",  help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default='sam_vit_h_4b8939.pth',help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", default="./input_image", type=str,  help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="output",  help="output directory"
    )
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False, help="bert_base_uncased model path, default=False")

    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    bert_base_uncased_path = args.bert_base_uncased_path
    # make dir
    os.makedirs(output_dir, exist_ok = True)

    def count_subfolders(path):
        if not os.path.isdir(path):
            raise ValueError(f"{path} is an unvalid directory")
    
        subfolder_count = sum(os.path.isdir(os.path.join(path, f)) for f in os.listdir(path))
        
        return subfolder_count

    runs = count_subfolders(output_dir)
    output_dir += f'/run{runs + 1}'
    os.makedirs(output_dir, exist_ok = True)
    
    directory = image_path
    sum = 0
    for dirpath, dirnames, filenames in os.walk(directory):
     for filename in filenames:
         sum += 1
    p = 0

    excel_file = output_dir + '/log.xlsx'
    excel_data = []
    df = pd.DataFrame()
    df.to_excel(excel_file, index=False, header=False) 
    
    for dirpath, dirnames, filenames in os.walk(directory):
     for filename in filenames:
      try:
       valid_extensions = ['.jpg', '.jpeg', '.png']
       if any(filename.lower().endswith(ext) for ext in valid_extensions) and 'checkpoint' not in filename.lower():
       
        from datetime import datetime
        now = datetime.now()
        p = p + 1
        print(f'{p} of {sum}')
  
        file_path = os.path.join(dirpath, filename)
        name = os.path.splitext(filename)[0]   
        file_paths = file_path.lstrip(image_path)
        file_paths = file_paths.rstrip('.jpg')
        mid_path = file_paths.rstrip(f'{name}')
        print(file_paths)
        outdir = output_dir +f'/{mid_path}'+ f'/{name}'
        string = process_string(outdir)
         
        excel_row = [name, '', 0, 0, 0, 0, '-']  
        #excel_row[0]: original fig name
        #excel_row[1]: output dir   
        #excel_row[2]: fig.png
        #excel_row[3]: mask.png
        #excel_row[4]: processed.png
        #excel_row[5]: zest_code/processed.png
        #excel_row[6]: is it target ?   
        if not os.path.exists(string):
         os.makedirs(string, exist_ok=True)
        # load image
        image_pil, image = load_image(file_path)
        # load model
        model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)

        # visualize raw image
        string = outdir+'/fig.png'
        string = process_string(string)
        try:
            image_pil.save(string)
            excel_row[2] = 1
        except:
            pass

        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
    model, image, text_prompt, box_threshold, text_threshold, device=device
    )

        # initialize SAM
        if use_sam_hq:
             predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
        else:
           predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
          boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
          boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
          boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

        masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

        # draw mask image
        plt.figure(figsize=(10, 10))
        image = np.zeros((10,10, 4), dtype=np.uint8)
        plt.imshow(image)
    
        for mask in masks:
          show_mask(mask.cpu().numpy(), plt.gca(), random_color=False)

        plt.axis('off')
        string = outdir+'/mask.png'
        string = process_string(string)
        try:   
          plt.savefig(string,
        bbox_inches="tight", dpi=300, pad_inches=0.0 )
          excel_row[3] = 1  
        except:
            pass

        try:
            name.index('target')
            excel_row[6] = '+' 
            target = 1
        except:
            target = 0 

    
        try:
            excel_row = save_mask_data(outdir, masks, boxes_filt, pred_phrases, target, excel_row, runs)
        except:
            pass
        excel_data.append(excel_row)   
        df = pd.DataFrame(excel_data)  
        df.to_excel(excel_file, index=False, header=False)    

      except : continue