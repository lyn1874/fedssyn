import numpy as np 
import torch 
from itertools import count


for i in count(0):
    a = torch.zeros([1]).to(torch.device("cuda"))