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
s='/home/bipin/final-cdf/train/fake/'
k=os.listdir(s)
d='/home/bipin/final10-cdf/train/fake/'
for l in k:
        h=os.path.join(s, l)
        t=os.listdir(h)
        path = os.path.join(d,l ) 
        os.mkdir(path)  
        c=0
        for j in t:
                if(c==10): 
                    break
                r=os.path.join(h,j)
                image =cv2.imread(r)
                os.chdir(path)
                cv2.imwrite(j,image)
                c+=1
