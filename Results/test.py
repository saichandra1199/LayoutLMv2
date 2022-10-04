from PIL import Image, ImageDraw, ImageFont
from transformers import LayoutLMv2Processor
import torch ,os ,json
import numpy as np

image_path = './DL/inputs/'
model_path = '../Models/layout_train.pt'
txt_path = "../Prepare/Sample_Data/layoutlmv2.txt"
output_path = './DL/outputs/'


def write_json(data,filename,indent):
    with open(filename, 'w') as f:
        json.dump(data,f,indent=indent)

def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def modify_label(label):
    label = label[2:]
    if not label:
      return 'other'
    return label

def preprocess(img):
    image = Image.open(os.path.join(image_path,img))
    image = image.convert("RGB")
    json_file = os.path.join(output_path,img[:-3] + "json")
    write_json(form,json_file,4)
    encoding = processor(image, return_offsets_mapping=True, max_length=512,truncation=True,return_tensors="pt")
    offset_mapping = encoding.pop('offset_mapping')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for k,v in encoding.items():
        encoding[k] = v.to(device)

    return image,encoding,offset_mapping,json_file

def results(image,encoding,offset_mapping):

    outputs = model(**encoding)
    # print("############",outputs)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()
    width, height = image.size
    is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
    true_predictions = [id2label[str(pred)] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]

    return true_predictions,true_boxes
        
def postprocess(json_file,true_predictions,true_boxes,image):
     
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    result = {}
    label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}

    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = modify_label(prediction).lower()
        result[predicted_label] = box
        with open(json_file) as outfile:
            data=json.load(outfile)
            temp = data["Result"]
            temp.append(result)
            write_json(data,json_file,4)
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=label2color[predicted_label], font=font)

    return image

def main(image_path):
    
    for idx,img in enumerate(os.listdir(image_path)):
        try:
            image,encoding,offset_mapping,json_file = preprocess(img)
            true_predictions,true_boxes = results(image,encoding,offset_mapping)
            final_image = postprocess(json_file,true_predictions,true_boxes,image)
            final_image.save(os.path.join(output_path,img))
            print("{} -----> {}".format(idx,img)) 
        except:
            print("! Error : ",img)

with open(txt_path,'r') as f:
    string = f.read()
    id2label = eval(string)

#Output Format
form = {
        "Result": []
        }

model = torch.load(model_path)
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
print("########## Loaded Labels Dictionary \n",id2label)
main(image_path)
