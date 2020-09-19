import os
import torch
from gpu_utils import get_free_gpu

os.environ["CUDA_VISIBLE_DEVICES"] = str(get_free_gpu())
os.environ["WANDB_PROGRAM"] = "multimodal_driver.py"
DEVICE = torch.device("cuda")

print("DEVICE SET TO: ", get_free_gpu())

DATASET_LOCATION = "/scratch/mhoque_lab/datasets/processed_multimodal_data/"

ACOUSTIC_DIM = 81
VISUAL_DIM = 91
TEXT_DIM = 768

XLNET_INJECTION_INDEX = 1
