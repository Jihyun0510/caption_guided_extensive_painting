import json

path = ""
with open (path + "test_predict.json", "r") as f:
    data = json.load(f)

save_path =""

for d in data:
    f = open(save_path+d['image_id']+".txt","w")
    f.write(d['caption'])
    f.close()
