import json

path = "/home/wgus5950/OFA/results/caption/"
with open (path + "test_predict.json", "r") as f:
    data = json.load(f)

save_path ="/database/jhkim/final_random_checkpoints/landmark/4000_texts/"

for d in data:
    f = open(save_path+d['image_id']+".txt","w")
    f.write(d['caption'])
    f.close()
