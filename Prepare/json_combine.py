import json,os
from PIL import Image
filepath = "/home/chandra/Downloads/LayoutLMv2/Prepare/label_studio_csvs/" # Where annotations and images folders are present
output_json = "./label_studio_csvs/layoutlmv2_train.json"
train = 0.8

def write_json(data,filename,indent):
    with open(filename, 'w') as f:
        json.dump(data,f,indent=indent)
        

def new_format(path,imgs):
  f = open(path)
  data = json.load(f)
  return {"id": data["id"], "words": data["words"], "bboxes": data["bboxes"], "ner_tags": data["ner_tags"], "image_path": imgs}

#LayoutLM v2 Format
form = {
        "train": 
                {
                    "features": []
                },
        "test": 
                {
                    "features": []
                }
        }

write_json(form,output_json,4)
annotations_dir = os.listdir(os.path.join(filepath, "annotations"))
images_dir = os.path.join(filepath, "images")
total = len(annotations_dir)
for i,annot in enumerate(annotations_dir):
    annots = os.path.join(os.path.join(filepath, "annotations"),annot)
    imgs = os.path.join(os.path.join(filepath, "images"),annot)
    imgs = imgs.replace("json", "jpg")
    result = new_format(annots,imgs)
    #result = generate_examples(i,annots,images_dir,annot)
    if i< train*total : 
        with open(output_json) as outfile:
            data=json.load(outfile)
            temp = data["train"]["features"]
            temp.append(result)
            write_json(data,output_json,4)
    elif i>= train*total : 
        with open(output_json) as outfile:
            data1=json.load(outfile)
            temp1 = data1["test"]["features"]
            temp1.append(result)
            write_json(data1,output_json,4)
        
print("⏳ {} has generated {} train and {} test annotations from {}".format(output_json,len(temp),len(temp1),filepath))

