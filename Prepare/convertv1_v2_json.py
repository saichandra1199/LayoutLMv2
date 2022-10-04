import json,os
from PIL import Image
filepath = "/home/vishwam/mountpoint/bhanu/LayoutLMv2/Prepare/Sample_Data" # Where annotations and images folders are present
json_file = "layoutlmv2.json"
output_json = os.path.join(filepath,json_file)
train = 0.8

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)

def normalize_bbox(bbox, size):
    return [int(1000 * bbox[0] / size[0]),
            int(1000 * bbox[1] / size[1]),
            int(1000 * bbox[2] / size[0]),
            int(1000 * bbox[3] / size[1])]

def write_json(data,filename,indent):
    with open(filename, 'w') as f:
        json.dump(data,f,indent=indent)

def generate_examples(guid,file_path,img_dir,file):
    words = [] ; bboxes = [] ; ner_tags = []
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)
    image_path = os.path.join(img_dir, file)
    image_path = image_path.replace("json", "png")
    image, size = load_image(image_path)
    for item in data["form"]:
        words_example, label = item["words"], item["label"]
        words_example = [w for w in words_example if w["text"].strip() != ""]
        if len(words_example) == 0:
            continue
        if label == "other":
            for w in words_example:
                words.append(w["text"])
                ner_tags.append("O")
                bboxes.append(normalize_bbox(w["box"], size))
        else:
            words.append(words_example[0]["text"])
            ner_tags.append("B-" + label.upper())
            bboxes.append(normalize_bbox(words_example[0]["box"], size))
            for w in words_example[1:]:
                words.append(w["text"])
                ner_tags.append("I-" + label.upper())
                bboxes.append(normalize_bbox(w["box"], size))
    return {"id": str(guid), "words": words, "bboxes": bboxes, "ner_tags": ner_tags, "image_path": image_path}

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
    print(annot)
    annots = os.path.join(os.path.join(filepath, "annotations"),annot)
    result = generate_examples(i,annots,images_dir,annot)
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
                   
print("⏳ {} has generated {} train and {} test annotations from {}".format(os.path.basename(output_json),len(temp),len(temp1),filepath))
