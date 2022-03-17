# EXPERIMENTS 
This folder contains for each training experiment two files: 
- ```model_config.yaml```: Mask R-CNN configuration 
- ```metrics.json```: training and evaluation metrics
The weights of Mask R-CNN obtained by each experiment are available upon request. 

Here the specifics of each experiment: 
|Exp.| Set | # images | # iterations | LR decrease | Train Loss | Val Loss | Val mAP | Test mAP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | Wild train/val | 5000 / 500 | 5k | 1.5k, 2.5k | 0.19 | 0.33 | 31.37 | 13.27
| 6 | Street train/val | 2500 / 500 | 5k | 1.5k, 2.5k | 0.17|0.37|47.94 | 15.30 
| 7 | Wild + Street train/val| 7500 / 1000 | 5k | 1.5k, 2.5k| 0.24 |0.33 | 32.29 | 16.36
| 8 |Wild + Street train/val| 7500 / 1000 | 10k | 5k, 7k | 0.14	| 0.34 | 40.19 | 19.54
| 9 |Wild + Street train/val| 7500 / 1000 | 15k | 9k, 13k |0.11	| 0.38 | 40.06 | 19.95
