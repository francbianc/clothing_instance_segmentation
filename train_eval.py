import torch
import numpy as np 
import os, json, cv2, random, time, datetime
import matplotlib.pyplot as plt

print('>>> CUDA Information')
print('torch version: ', torch.__version__)
print('cuda version: ', torch.__version__.split("+")[-1])

print('CUDA is currently available: ', torch.cuda.is_available()) 
print('number of GPUs available: ', torch.cuda.device_count())
print('GPUs names: ', [torch.cuda.get_device_name(gpu) for gpu in range(torch.cuda.device_count())])
print('GPUs capbilities: ', [torch.cuda.get_device_capability(gpu) for gpu in range(torch.cuda.device_count())])
print('GPUs properties:')
print('\n'.join([str(torch.cuda.get_device_properties(gpu)) for gpu in range(torch.cuda.device_count())]))

# Setup detectron2 logger
import detectron2
import logging
from detectron2.utils.logger import setup_logger, log_every_n_seconds
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.data import print_instances_class_histogram, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, inference_context, print_csv_format
from detectron2.engine.hooks import HookBase
import detectron2.utils.comm as comm

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
print('\n>>> Library correctly imported')

###############################################################################################################
### Directories 
# @@@ OVERRIDE according to the train/validation set you want to use
DATA_DIR = '.../data'
dir_tr = os.path.join(DATA_DIR, 'shop_df2/train/df2_train_shop')
dir_tr_json = os.path.join(DATA_DIR, 'shop_df2/train/coco_df2_ts_2500.json')

dir_tr2 = os.path.join(DATA_DIR, 'user_df2/train/df2_train_user')
dir_tr_json2 = os.path.join(DATA_DIR, 'user_df2/train/coco_df2_tu_5000.json')

dir_val = os.path.join(DATA_DIR, 'shop_user_df2/validation/df2_val_shop_user')
dir_val_json = os.path.join(DATA_DIR, 'shop_user_df2/validation/coco_df2_vsu_1000.json')

dir_te = os.path.join(DATA_DIR, 'test/dlcv_test')
dir_te_json = os.path.join(DATA_DIR, 'test/coco_test_500.json')

name_train = 'train_df2s'
name_train2 = 'train_df2u'
name_val = 'val_df2su'
num_categories = 30
###############################################################################################################

### Register Dataset
register_coco_instances(name_train, {}, dir_tr_json, dir_tr)
register_coco_instances(name_val, {}, dir_val_json, dir_val)

dataset_train = DatasetCatalog.get(name_train)
dataset_val = DatasetCatalog.get(name_val)

meta_train = MetadataCatalog.get(name_train)
meta_val = MetadataCatalog.get(name_val)

register_coco_instances(name_train2, {}, dir_tr_json2, dir_tr2) # Register also the second train/val set
dataset_train2 = DatasetCatalog.get(name_train2)
meta_train2 = MetadataCatalog.get(name_train2)
print('\n>>> Datasets correctly registered and loaded')

### Train 
# Load Configuration 
#cfg_name = 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
cfg_name =  'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
cfg = get_cfg() 
cfg.merge_from_file(model_zoo.get_config_file(cfg_name))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_name)

cfg.DATASETS.TRAIN = (name_train, name_train2,)           
cfg.DATASETS.TEST = (name_val,)              
cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_categories

cfg.TEST.EVAL_PERIOD = 250                   # default: 0
cfg.SOLVER.MAX_ITER = 15000                   # default: 270k - adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (9000, 13000)             # default: (210k, 250k) - iteration number to decrease learning rate by GAMMA.
print('\n>>> Configuration correctly created')

# Update Default Trainer instance: add 1 hook and validation evaluator
class LossEvalHook(HookBase):
    """Source: github @ortegatron/LossEvalHook.py"""
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()
        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
           
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

class FashionTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Override main method to evaluate on validation set"""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "val_inference")
            os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, True, output_folder) 
                     
    def build_hooks(self): 
        """Override main method to add 1 hook to compute validation loss"""
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(cfg.TEST.EVAL_PERIOD, self.model,
            build_detection_test_loader(self.cfg,self.cfg.DATASETS.TEST[0],DatasetMapper(self.cfg,True))))
        return hooks
print('\n>>> Trainer correctly created')

# Train!
start = time.time()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
with open(os.path.join(cfg.OUTPUT_DIR, "model_config.yaml"), "w") as f:
  f.write(cfg.dump())
trainer = FashionTrainer(cfg)  
print('\n>>> Training Started: ', time.time())
trainer.resume_or_load(resume=False)
trainer.train()
print('\n>>> Training Ended: ', start-time.time())

### Evaluation 
print()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # default: 0.05 - set a custom testing threshold
print('\n>>> Configuration correctly re-created')

register_coco_instances('df2_test', {}, dir_te_json, dir_te)
dataset_test = DatasetCatalog.get('df2_test')
meta_test = MetadataCatalog.get('df2_test')
cfg.DATASETS.TEST = ("df2_test",)

predictor = DefaultPredictor(cfg)
print('\n>>> Predictor correctly created')

test_output_folder = os.path.join(cfg.OUTPUT_DIR, 'test_inference')
os.makedirs(test_output_folder, exist_ok=True)
evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, output_dir=test_output_folder)
val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
results = inference_on_dataset(predictor.model, val_loader, evaluator)
print_csv_format(results)