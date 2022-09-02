import numpy as np
from PIL import Image
import os

file_list = os.listdir('/database/jhkim/glide_dataset/landscape-768-test/')
save_path = "/database/glide_dataset/landscape_masked/"
for file in file_list:
   
    img = Image.open('/database/jhkim/glide_dataset/landscape-768-test/'+file)
    format = img.format
    
    img = np.array(img)

    # masking
    width = img.shape[1]
    mask_width = int(width*0.25)
    img[:,:mask_width] = 1
    img[:,width-mask_width:] = 1
        
    # array to image 
    img = Image.fromarray(img)   
    img.save(save_path + file[:-4] + '.png', 'png')