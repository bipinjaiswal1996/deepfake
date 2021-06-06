import os
import glob
import json
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
    
pad = 40
s='/home/bipin/face-cdf/val/real/'
k=os.listdir(s)
d='/home/bipin/final-cdf/val/real/'
for l in k:
        h=os.path.join(s, l)
        t=os.listdir(h)
        path = os.path.join(d,l ) 
        os.mkdir(path)  
        for j in t:
                r=os.path.join(h,j)
                image =cv2.imread(r)
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image=cv2.resize(image,(128,128))
                os.chdir(path)
                cv2.imwrite(j,image)
