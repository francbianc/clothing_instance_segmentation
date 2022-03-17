# DATASET 
Following the steps in README, three versions of the train/validation set and one single version of the test set have been created. 
All these sets have been stored in a folder with the following structure. Annotations and images for each set are available upon request.  
**'Shop' refers to DF2 street images, 'user' refers to DF2 wild images.**

- ```data```
  - **```test```**
      - ```dlcv_test```: folder with test images (filenames from 0 to 499)
      - coco_test_500.json: json file with test annotations in COCO format 

  - **```shop_df2```**
      - ```train```
          - ```df2_train_shop```: 2500 DF2 STREET images selected for training
          - coco_df2_ts_2500.json: json file with annotations in COCO format
      - ```validation```
          - ```df2_val_shop```: 501 DF2 STREET images selected for validation
          - coco_df2_vs_500.json: json file with annotations in COCO format

  - **```user_df2```**
      - ```train```
          - ```df2_train_user```: 5000 DF2 WILD images selected for training
          - coco_df2_tu_5000.json: json file with annotations in COCO 
      - ```validation``` 
          - ```df2_val_user```: 505 DF2 WILD images selected for validation
          - coco_df2_vu_500.json: json file with annotations in COCO format

  - **```shop_user_df2```**
      - ```validation```
          - ```df2_val_shop_user```: 501 + 505 images from df2_val_shop and df2_val_user
          - coco_df2_vsu_1000.json: json file with annotations in COCO format from coco_df2_vs_500.json and coco_df2_vu_500.json

NB: there wasn't the need to create a shop_user_df2/train set because Detectron2's Trainer object can take as training dataset a list of set
