import cv2
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from facenet_pytorch import MTCNN

pad = 45
s='/media/bipin/3A8AD7E58AD79C27/datasetfacenew2/validation/realframes/'
k=os.listdir(s)
d='/media/bipin/3A8AD7E58AD79C27/datasetfacenew2/validation/realface/'
mtcnn = MTCNN(device='cuda:0')



for l in k:
        h=os.path.join(s, l)
        t=os.listdir(h)
        path = os.path.join(d,l ) 
        os.mkdir(path)
        images=[]
        conf_score=[]  
        area=[]
        for j in t:
                r=os.path.join(h,j)
                image =cv2.imread(r)
                # print(image.shape)
                # image.resize((int(image.shape[0]/4),int(image.shape[1]/4)))
                #     plt.imshow(image)
                frame = Image.fromarray(image)
                face_locations ,confidence= mtcnn.detect(frame)
                if((face_locations is not None) and confidence[0]>=0.98):
                        images.append(j)
                        conf_score.append(confidence[0])
                        face_location = face_locations[0]
                        x1, y1, x2, y2 = face_location
                        area.append(abs(x2-x1)*abs(y2-y1))
                        print(confidence[0],j)
                        face_image = image[int(y1-pad):int(y2+pad), int(x1-pad):int(x2+pad)]
                        os.chdir(path)
                        x,y,z=face_image.shape
                        # print(face_image.shape)
                        if(x!=0 and y!=0):    
                                cv2.imwrite(j,face_image)
        dict={'image_name':images,'confidence':conf_score,'area':area   }
        df=pd.DataFrame(dict)
        df.to_csv('scores.csv')