#라이브러리 호출 및 GPU 설정

from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.append("/content/drive/MyDrive/Colab Notebooks")
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import StepLR
import wandb
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Hyper Parameter 및 경로 설정 

BATCHSIZE = 32
LR = 1e-5
LR_STEP = 5
LR_GAMMA = 0.9
EPOCH = 20
criterion = nn.CrossEntropyLoss()
new_model_train = True
model_type = "resnet18"
dataset = "Animal90"
save_model_path = f"/content/drive/MyDrive/Colab Notebooks/result/{model_type}{dataset}.pt"
save_history_path = f"/content/drive/MyDrive/Colab Notebooks/result/{model_type}history{dataset}.pt"
