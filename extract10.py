import cv2
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from facenet_pytorch import MTCNN


s='/media/bipin/3A8AD7E58AD79C27/datasetfacenew2/validation/realface/'
k=os.listdir(s)
d='/media/bipin/3A8AD7E58AD79C27/datasetfacenew2/validation/real10/'


for l in k:
        h=os.path.join(s, l)
        path = os.path.join(d,l ) 
        os.mkdir(path)
        dfpath=os.path.join(h,'scores.csv')
        df=pd.read_csv(dfpath)
        # print(df['image_name'])
        df=df.sort_values(by=['area'],ascending=False)
        images=df['image_name'].tolist()    
        # print(images)
        for j in range(20):
                r=os.path.join(h,images[j])
                # print(r)
                image =cv2.imread(r)
                os.chdir(path)
                if(image is not None):
                        cv2.imwrite(images[j],image)
