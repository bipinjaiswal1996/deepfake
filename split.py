import json
import shutil,os
import random

imagesFolder = '/media/bipin/3A8AD7E58AD79C27/datasetfacenew2/videos/'
dest1='/media/bipin/3A8AD7E58AD79C27/datasetfacenew2/test/realvideo/'
dest2='/media/bipin/3A8AD7E58AD79C27/datasetfacenew2/validation/realvideo/'

l=os.listdir(imagesFolder)
# print(l[0:5])
# print(l[0:5])
items=len(l)
random.shuffle(l)
random.shuffle(l)
random.shuffle(l)
# print(l[0:5])
# print(l[0:5])
test_len=int(0.15*items)
val_len=int(0.15*items)
# print(test_len)
# print(val_len)

for i in range(test_len):
    shutil.move(imagesFolder+l[i],dest1)

for i in range(val_len):
    shutil.move(imagesFolder+l[i+test_len],dest2)

# #for i in json_data:
# #print(i,json_data[i]["label"])
# if(json_data[i]["label"]=='FAKE'):
# shutil.move("/home/bipin/train_sample_videos/"+i,"/home/bipin/fake")
# else:
# shutil.move("/home/bipin/train_sample_videos/"+i,"/home/bipin/real")

