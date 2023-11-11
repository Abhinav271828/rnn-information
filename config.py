import torch
from torch import nn, tensor, mm
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from tqdm import tqdm
from icecream import ic
import numpy as np
from numpy.linalg import matrix_rank, svd, inv
from sklearn.feature_selection import mutual_info_regression

DEVICE = torch.device('cpu')
CPUS = os.cpu_count()
BATCH_SIZE = 2048