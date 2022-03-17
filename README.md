# Clothing Instance Segmentation in the Wild: A Supervised Deep Learning Approach
**Research Thesis**  
MSc in Data Science and Business Analytics, Bocconi University   
Supervisor: Francesco Grossetti

<p align="center">
  <img src="https://github.com/francbianc/Clothing_Instance_Segmentation/blob/main/tools/sample4.png" width="100" height="200" alt="Image"/>
</p>


## Goal 
This repo contains a pipeline to perform **clothing instance segmentation on wild images**: identifying with segmentation masks 30 fashion items in un-constrained real-world images, referred to as "wild" and retrieved from Instagram. Wild images are the opposite of "street" images that are out-of-the-studio good quality pictures, usually focused on one professional model, with minor occlusion and clear backgrounds.

To solve the task, a supervised approach has been implemented, with [**Mask R-CNN**](https://arxiv.org/abs/1703.06870) as segmentation model. The train and validation datasets have been created using [DeepFashion2](https://github.com/switchablenorms/DeepFashion2), a publicly-available already-labeled fashion dataset with wild and street photos. The test set has been created completely from scratch using random photos downloaded from Instagram. To improve the training’ speed and accuracy, transfer learning has been performed from [COCO instance segmentation](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-instance-segmentation-baselines-with-mask-r-cnn). 

## Requirements
Create a virtual environment and install the required libraries. Make sure your pip version is updated.
```
conda create -n venv python==3.9.0
conda activate venv
python -m pip install --upgrade pip
pip install -r requirements.txt 
```
Then install [Detectron2](https://github.com/facebookresearch/detectron2), according to the steps from the [official documentation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html), given the following requirements are satisfied. 

1. Linux or macOS with Python ≥ 3.6
2. PyTorch ≥ 1.8 and torchvision that matches the PyTorch installation. Install them together at [pytorch.org](https://pytorch.org/) to make sure of this.
3. gcc & g++ ≥ 5.4 

For example, on a Linux device with CUDA 11.3, run the following lines to install PyTorch and Detectron2: 
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Pipeline
Read this section to understand the purpose of each file in this repo.

### 1. Literature Review
To solve an instance segmentation problem in a supervised manner, two elements are required: images and annotations. Annotating images from scratch is really time consuming, thus the idea to build the train/validation set from one of the publicly-available, already-labeled fashion instance segmentation datasets: [DeepFashion2](https://github.com/switchablenorms/DeepFashion2), [ModaNet](https://github.com/eBay/modanet), or [Fashionpedia](https://github.com/KMnP/fashionpedia-api).

- **```summary_datasets.ipynb```**: contains code to download DeepFashion2 (DF2) and Fashionpedia, to analyze and visualize samples from DF2, ModaNet, Fashionpedia
- **```download_modanet.ipynb```**: contains code to download ModaNet
- **```converters.ipynb```**: contains code to convert json files from DF2 to COCO format (used in: **summary_datasets.ipynb**)

### 2. Dataset Creation
Given the goal to segment wild Instagram pictures, the train/validation set should contain both wild and street images. For this reason, images from DeepFashion2 (DF2) have been used to build three different versions of the train/validation set. The idea is to test how the same model, trained for each experiment on one version of the train set, performs on the same wild test set. 

| Version | Dataset | Source | # images |
|---|---|---|---|
| 1 | Train | Street images from DF2 | 2500 
| | Validation | Street images from DF2 | 500
| 2 | Train | Wild images from DF2 | 5000
| | Validation | Wild images from DF2 | 500
| 3 | Train | Street + Wild images from DF2 | 7500
| | Validation | Street + Wild images from DF2 | 1000
| | Test | Wild images from Instagram | 500

#### 2.1 Data Collection and Cleaning
- **```to_annotate.ipynb```**: contains code to explain the collection and cleaning process for the train, validation, test sets
- **```color_detector.py```**: contains code to detect color vs. no-color images given a folder with images (used in: **to_segment.ipynb**)
- **```face_detector.py```**: contains code to detect face vs. no-face images given a folder with images (used in: **to_segment.ipynb**)

#### 2.2 Data Annotation
**Annotator**: [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/)    
[Download Link](https://www.robots.ox.ac.uk/~vgg/software/via/downloads/via-2.0.11.zip)

VGG Image Annotator (VIA) has four advantages over other annotation tools: it has no limits in terms of number of annotations; it can be downloaded and runs locally, without the need to upload any image anywhere; it gives users the possibility to draw annotations first and define labels in a second moment; it allows to define attributes at image level. The only problem of VIA is that the json file with annotations is not in the standard COCO format, but converting VIA annotations to COCO is easy. 

**Categories**: 30 fashion categories have been considered. They have been retrieved from a fashion ontology, stored in ```tools/fashion_ontology.xlsx```, that maps 211 fashion items to a supercategory, category and subcategory. 

| Label | ID | Label | ID | Label | ID |
|---| ---| ---| ---|  ---| ---| 
| Hat | 1 | Pyjama | 11 | Leggings | 21
| Bag | 2 | Robe | 12 | Joggers |  22
| Boots | 3 | Dress | 13 | Jacket | 23
| Slippers | 4 | Jumpsuit | 14 | Blazer | 24
| Sneakers | 5 | Cover-ups | 15 | Waistcoat | 25
| Heels | 6 | Coat | 16 | Cardigan | 26
| Flat Shoes | 7 | Pants | 17 | Sweater | 27
| Swimsuit | 8 | Jeans | 18 | Sweatshirt | 28
|Bathrobe | 9 | Shorts | 19 | Shirt | 29
|Underwear | 10 |Skirt | 20 | Top | 30

**Annotation**:  
*Train Set*: re-annotate DF2 images according to 30 categories  
*Validation Set*: re-annotate DF2 images according to 30 categories  
*Test Set*: annotate IG images from scratch according to 30 categories  
Annotations in COCO format for each set are available in ```data``` folder.

- **```to_annotate.ipynb```**: contains code to track annotation progress and manipulate json files from VIA
- **```pre_convert.ipynb```**: contains some code to manipulate json files in VIA format, before converting them to COCO format with the code in converters.ipynb
- **```copy_files.py```**: contains code to copy files from a folder in a new specified folder and rename them from 0 to N (used in: **pre_convert.ipynb**)
- **```converters.ipynb```**: contains code to convert json files from VIA to COCO format

#### 2.3 Data Analysis
Given images and annotations, some metrics have been computed for each set to assess the *quality of annotations* (number of masks, average number of masks per image, average number of vertices per mask), the *quality of images* (average resolution, percentage of images with 1 person, with more than 1 person, without any person, percentage of selfies) and to understand the *frequency and average dimension of fashion items* (categories distribution, average mask area over image area).

- **```summary_mydataset.ipynb```**: contains code to compute all those metrics 

### 3. Training and Inference
[Mask R-CNN](https://arxiv.org/abs/1703.06870) has been implemented as the instance segmentation model, using [Detectron2](https://github.com/facebookresearch/detectron2), Facebook AI Research's python library for image segmentation and object detection.   
[Detectron2 Documentation](https://detectron2.readthedocs.io/en/latest/)    

- **```detectron2.ipynb```**: contains code to understand how to install and use Detectron2 library, including how to train, evaluate and make inference with Mask R-CNN
- **```train_evaluate.py```**: contains code to train and evaluate Mask R-CNN, given a train, validation and test set (transfer learning is performed)
- **```experiments.ipynb```**: contains code to access the training and evaluation metrics
