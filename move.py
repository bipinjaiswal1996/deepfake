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
s='/media/bipin/3A8AD7E58AD79C27/datasetfacenew2/validation/real10/'
k=os.listdir(s)
d='/media/bipin/3A8AD7E58AD79C27/datasetfacenew2/validation/real/'
for l in k:
        h=os.path.join(s, l)
        t=os.listdir(h)
        c=0
        for j in t:
                r=os.path.join(h,j)
                image =cv2.imread(r)
                image=cv2.resize(image,(128,128))
                os.chdir(d)
                cv2.imwrite(j,image)
