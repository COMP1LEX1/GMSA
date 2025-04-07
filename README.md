# GMSA
# Spatial Transcriptomics Alignment - Environment Info

This script requires the following Python packages:

## Required Python Packages

```python
import anndata
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn.metrics.pairwise
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
import os

# Tested Environment (Reproducible Setup)

This code was tested under the following environment:

- Python 3.8.20
- OS: Linux (Ubuntu)
- CUDA: 12.1
- PyTorch: 2.4.1
- GPU available: Yes (`cuda:1`)

## Required Python Packages

Install packages via pip:

```bash
pip install anndata==0.9.2
pip install scanpy==1.9.8
pip install numpy==1.24.4
pip install scipy==1.10.1
pip install pandas==1.2.4
pip install matplotlib==3.7.5
pip install seaborn==0.11.2
pip install networkx==3.1
pip install scikit-learn==1.3.2
pip install plotly==5.24.1
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

