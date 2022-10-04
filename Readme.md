# LayoutLMv2 Setup

## Environment Setup

1. Create new conda env with python==3.10.4

2. Install all the dependencies from requirements.txt
    pip3 install -r requirements.txt

3. Activate created conda environment before further process.

4. Remove the existing detectron2 folder in the repository and run the following commands for installing detectron2 

    git clone https://github.com/facebookresearch/detectron2.git and 
    
    python -m pip install -e detectron2


## Data Preparation

5. Prepare data from Prepare Folder 
    a. place images and annotations of layoutlmv1 in one folder 
    
    b. give required inputs in convertv1_v2_json.py ( above folder path and desired json file name )
        filepath = "./Prepare/Sample_Data/" 
	
        json_file = "layoutlmv2.json"
	
    c. run python3 convertv1_v2_json.py
    
    d. json will be generated.
   
   for training on your own images annotate the images using label studo ( The open source tool for labelling images which includes bounding boxes,labels,text present in the box ).
   
   Then use the csv generated from label studio to convert it into a json for each image.
   
   If you have multiple jsons each for an image and need to combine them for training then use json_combine.py

## Training 

6. Add required inputs in train_json.py

    data_files= path to create json file
    
    pretrained_model = use previously trained .pt model or else keep it as ""
    
    model_to_save = desired trained model name ( ex: "layout_train.pt" )
    
    num_train_epochs = As per requirement ( default 100 )
    
    batch_size = As per requirement based on available data( default 16 ).

7. run python3 train_json.py

8. After completion model will be saved in  Models/ folder and a text file will be saved in your data folder in Prepare/ this file has dictionary of ids and labelled mapped. This is used while testing to predict the correct label according to training labels.


## Model 
	
9. Download the pretrained model from the drive link below and add it in Models/ folder

	https://drive.google.com/file/d/1z78obbtNrn-enWRscWehlltw4LY_o-gy/view?usp=sharing

## Testing

10. Add required inputs in Results/test.py

    image_path = input images path 
    
    model_path = trained .pt model path
    
    txt_path = generated text file path after training for using the labels mapped with respective ids
    
    output_path = empty output folder path where output images and json will be saved.
    
    Change label colors as per your requirement in postprocess function.

11. run python3 test.py
