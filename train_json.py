import json,os,torch
from torch.utils.data import DataLoader
from PIL import Image
from transformers import LayoutLMv2Processor,LayoutLMv2ForTokenClassification, AdamW
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D ,load_dataset,load_metric
from tqdm.notebook import tqdm

data_files="./Prepare/Sample_Data/layoutlmv2.json"
pretrained_model = ''
model_to_save = "layout_train_2.pt"
num_train_epochs = 10
batch_size = 10
device = "cpu"  # use "cuda" if you have gpus for training 

print("########################## Using Training Data from {}".format(os.path.basename(data_files)))
datasets = load_dataset("json", data_files=data_files,field = "train")
test_datasets = load_dataset("json",data_files=data_files,field = "test")
with open(data_files) as f:
  data=json.load(f)
  labels = data["train"]['features'][0]['ner_tags']
labels = list(set(labels))

id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}

txt = os.path.basename(data_files).split(".")[0]+".txt"
labels_txt = os.path.join(os.path.split(data_files)[0],txt)
with open(labels_txt,"w") as t:
  t.write(json.dumps(id2label))

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

# we need to define custom features
features = Features({
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(Value(dtype='int64')),
})

def preprocess_data(examples):
  images = [] ; words = [] ; boxes =[] ; word_labels = []
  for idx in range(len(examples['features'])):
    images.append(examples['features'][idx]['image_path'])
    words.append(examples['features'][idx]['words'])
    boxes.append(examples['features'][idx]['bboxes'])
    labels= examples['features'][idx]['ner_tags']
    word_labels.append([label2id[str(labs)] for labs in labels])
  images = [Image.open(path).convert("RGB") for path in images]
  
  encoded_inputs = processor(images, words, boxes=boxes, word_labels=word_labels,max_length = 512,padding="max_length", truncation=True)
  
  return encoded_inputs

train_dataset = datasets['train'].map(preprocess_data, batched=True,remove_columns=datasets['train'].column_names,
                                      features=features)
test_dataset = test_datasets['train'].map(preprocess_data, batched=True,remove_columns=test_datasets['train'].column_names,
                                      features=features)


processor.tokenizer.decode(train_dataset['input_ids'][0])
train_dataset.set_format(type="torch", device=device)
test_dataset.set_format(type="torch", device=device)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1)

batch = next(iter(train_dataloader))
for k,v in batch.items():
  print(k, v.shape)


if pretrained_model == "":
  model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',num_labels=len(labels))
else:
  model = torch.load(pretrained_model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
global_step = 0
t_total = len(train_dataloader) * num_train_epochs # total number of training steps 

#put the model in training mode
model.train() 
for epoch in range(num_train_epochs):  
   print("Epoch:", epoch)
   for batch in tqdm(train_dataloader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(**batch) 
        loss = outputs.loss
        
        # print loss every 100 steps
        if global_step % (len(train_dataset)/batch_size) == 0:
          print(f"Loss : {loss.item()}")

        loss.backward()
        optimizer.step()
        global_step += 1

torch.save(model,os.path.join("/home/vishwam/mountpoint/bhanu/LayoutLMv2/Models/",model_to_save))

# Evaluate 


metric = load_metric("seqeval")

# put model in evaluation mode
model.eval()
for batch in tqdm(test_dataloader, desc="Evaluating"):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        bbox = batch['bbox'].to(device)
        image = batch['image'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        # forward pass
        outputs = model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask, 
                        token_type_ids=token_type_ids, labels=labels)
        
        # predictions
        predictions = outputs.logits.argmax(dim=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [id2label[p.item()] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l.item()] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        metric.add_batch(predictions=true_predictions, references=true_labels)

final_score = metric.compute()
print("######################################### Evaluation Results for {} data points ########################################".format(len(test_dataset)))
for field in final_score:
  print(field,"===============>",final_score[field])