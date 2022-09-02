import numpy as np
import PIL
from PIL import Image
from io import BytesIO
import base64
import io
import os
import csv
import pandas as pd
from regex import F

file_list = os.listdir("/database/jhkim/amster/resized2/")
l1 = []
image_str = []

for file in file_list:
    l1.append(file[:-4])
    
    img = Image.open("/database/jhkim/amster/resized2/"+file)
    format = img.format
    
    img = np.array(img)

    # masking
    width = img.shape[1]
    mask_width = int(width*0.5)
    # mask_width = 128
    # img[:,:mask_width] = 1
    img[:,-mask_width:] = 1

    # array to image 
    img = Image.fromarray(img)   

    # image to string
    img_buffer = BytesIO()
    img.save(img_buffer, format=format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data) # bytes
    base64_str = base64_str.decode("utf-8")
    
    image_str.append(base64_str)

df = pd.DataFrame()
df['c1'] = l1
df['c2'] = l1
df['caption'] = [np.nan for i in range(len(file_list))]
df['c4'] = ['' for i in range(len(file_list))]
df['img_str'] = image_str

path = "/home/wgus5950/OFA/dataset/caption_data"
df.to_csv(f'{path}/amster_half_128.tsv', sep="\t", index=False, header=False)