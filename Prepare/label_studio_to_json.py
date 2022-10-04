import numpy as np
import pandas as pd 
from ast import literal_eval
import json
import os
path="/home/chandra/Downloads/LayoutLMv2/Prepare/label_studio_csvs/new_csvs/"
images_path="/home/chandra/Downloads/LayoutLMv2/Prepare/label_studio_csvs/images/"
output_jsons = "/home/chandra/Downloads/LayoutLMv2/Prepare/label_studio_csvs/annotations/"

csvs=os.listdir(path)
frames = [ pd.read_csv(path+f,converters={'label': literal_eval,'transcription' : literal_eval }) for f in csvs ]
d = pd.concat(frames,ignore_index=True)
#d=pd.read_csv("/content/project-13-at-2022-09-14-09-54-67fd2cfe.csv",converters={'label': literal_eval,'transcription' : literal_eval })

x=[]
y=[]
width=[]
height=[]
labels=[]
img=[]
txt=[]
for i in range(len(d["label"])):
  h=d["ocr"][i]

  for j in d["label"][i]:
    print(j)
    k=j
    x_cord=k.get("x")
    print(x_cord)
    x.append(x_cord)
    y_cord=k.get("y")
    y.append(y_cord)
    width_cord=k.get("width")
    width.append(width_cord)
    height_cord=k.get("height")
    height.append(height_cord)
    labels_cord=k.get("labels")
    labels_cord1=labels_cord[0]
    labels.append(labels_cord1)
    img.append(h)
  for n in d["transcription"][i] :
    txt.append(n)

new = pd.DataFrame(list(zip(x,y,width,height,labels,img,txt)),columns =['x1','y1','width','height','labels','img','txt'])
new["x2"]=new["x1"]+new["width"]
new["y2"]=new["y1"]+new["height"]
o=new["img"].str.split("/",expand=True)
new["img_name"]=o[4]
new["img_name"]=new["img_name"].str.split("-",expand=True)[1]
new.drop(columns =["width","height"], inplace = True)
new["x1"]=new["x1"].astype('int32')
new["y1"]=new["y1"].astype('int32')
new["x2"]=new["x2"].astype('int32')
new["y2"]=new["y2"].astype('int32')

new=new.loc[:,['x1','y1','x2','y2','labels','txt','img_name']]

for i in range(len(np.unique(new["img_name"]))):
  print(np.unique(new["img_name"])[i])
  new_image_path=images_path+np.unique(new["img_name"])[i]
  new_df=new.loc[new["img_name"]==np.unique(new["img_name"])[i]]
  new_df.reset_index(inplace = True)
  bboxes=[]
  for j in range(len(new_df)):
    one_box=[int(new_df["x1"][j]),int(new_df["y1"][j]),int(new_df["x2"][j]),int(new_df["y2"][j])]
    bboxes.append(one_box)
  words=new_df["txt"].to_list()
  ner_tags=new_df["labels"].to_list()
  doc={"id":str(i),"words":words,"bboxes":bboxes,"ner_tags":ner_tags,"image_path":new_image_path}
  print(doc)
  with open(output_jsons+np.unique(new["img_name"])[i].replace(".jpg","")+".json", "w") as outfile:
    json.dump(doc, outfile)
