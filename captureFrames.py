import cv2
import math
import os
import numpy as np
import matplotlib.pyplot as plt

n_frames = 40
source_dir='/media/bipin/3A8AD7E58AD79C27/datasetfacenew2/validation/fakevideo/'
l=os.listdir(source_dir)
print(len(l))
h=0
save_dir='/media/bipin/3A8AD7E58AD79C27/datasetfacenew2/validation/fakeframes/'
count=1
for j in l:
    h+=1
#     print(h)
    for i in range(len(j)):
        if(j[i]=='.'):
            break

    print(count)
    count+=1
    video_name=j[0:i]
#     print(video_name)
    directory = video_name
    parent_dir = save_dir
    path = os.path.join(parent_dir, directory)
    if(os.path.exists(path)):
        continue
    os.mkdir(path)  
    # Create video reader and find length
    filename=source_dir+j
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Pick 'n_frames' evenly spaced frames to sample

    sample = np.linspace(0, v_len - 1, n_frames).astype(int)
            # Loop through frames
#     print(v_cap)   
    k=0
    os.chdir(path)
    for j in range(v_len):
        success = v_cap.grab()
        if j in sample:
                    # Load frame
            success, frame = v_cap.retrieve()
            if not success:
                continue
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         frame = Image.fromarray(frame)
            img_name=video_name+'@'+str(k)+'.jpg'
            cv2.imwrite(img_name, frame)
            k+=1
    #         save_path = os.path.join(save_dir, f'{j}.jpg')
    v_cap.release()
