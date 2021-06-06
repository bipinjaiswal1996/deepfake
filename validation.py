from tqdm import tqdm
import argparse
import math
import model
import cv2
import sys
import pandas as pd
sys.setrecursionlimit(15000)
import os
import torch
import numpy as np
from torch.autograd import Variable
from sklearn import metrics
from PIL import Image
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='dataset')
parser.add_argument('--test_set', default ='dataset/validation')
parser.add_argument('--real', default ='validation/real')
parser.add_argument('--fake', default ='validation/fake')
parser.add_argument('--batchSize', type=int, default=10)
parser.add_argument('--imageSize', type=int, default=128)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--outf', default='checkpoints/deepfakes')
parser.add_argument('--random_sample', type=int, default=0)
parser.add_argument('--random', action='store_true', default=False)
parser.add_argument('--id',type=int,default=100)
parser.add_argument('--input_dim', type=int, default=3, help='input dimension of lstm')
parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension of lstm')
parser.add_argument('--layer_dim', type=int, default=1, help='layer dimension of lstm')
parser.add_argument('--output_dim', type=int, default=2, help='output dimension of lstm')
opt = parser.parse_args()
print(opt)

img_ext_lst = ('jpg', 'jpeg', 'bmp', 'png', 'ppm', 'gif', 'tiff')
vid_ext_lst = ('avi', 'mkv', 'mpeg', 'mpg', 'mp4')

def get_file_list(path, ext_lst):
    file_lst = []

    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)):
            if f.lower().endswith(ext_lst):
                file_lst.append(f)

    return file_lst

def extract_file_name_without_count(in_str, sep_char='@'):
    n = len(in_str)
    pos = 0
    for i in range(n):
        if in_str[i] == sep_char:
            pos = i

    return in_str[0:pos]

def process_file_list(file_lst, sep_char='@'):
    result_lst = []

    for i in range(len(file_lst)):
        #remove extension
        filename = os.path.splitext(file_lst[i])[0]

        filename = extract_file_name_without_count(filename, sep_char)
        result_lst.append(filename)

    return result_lst

def classify_batch(vgg_ext, model, batch):
    n_sub_imgs = len(batch)
    
    batchSize =n_sub_imgs 
    batch = torch.stack(batch)
    t,s,c,h,w=batch.shape
    batch = batch.view(t * s, c, h, w)

    img_tmp=batch

    if opt.gpu_id >= 0:
        img_tmp = img_tmp.cuda(opt.gpu_id)

    input_v = Variable(img_tmp, requires_grad = False)
    # print(input_v.shape)
    x = vgg_ext(input_v)
    # print(x.shape,"shape &&&&&&&&&&&&&")
    class_ = model(x)
    output_p = []
    for t in class_:
        if t[0]>t[1]:
            output_p.append(0)
        else:
            output_p.append(1)
    
    # output_pred = np.concatenate((output_pred, output_p))
    return output_p

transform_fwd = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def classify_frames(vgg_ext, model, path, label):	
    file_lst = get_file_list(path, img_ext_lst)
    file_lst.sort()
    length = len(file_lst)
    frames =[]
    if (length == 0):
        return
    elif (length == 1):
        print('Error: Only one file!')
        return

    correct = 0
    count_vid = 0
    file_edited = process_file_list(file_lst)
    vid_name = file_edited[0]
    test_img = transform_fwd(Image.open(os.path.join(path, file_lst[0])))
    # print(type(test
    frames.append(test_img.unsqueeze(0))

    for i in range(1, length):
        if file_edited[i] == vid_name:
            test_img = transform_fwd(Image.open(os.path.join(path, file_lst[i])))
            frames.append(test_img.unsqueeze(0))
        else:
            # clasify the previous frames
            if(len(frames)==10):
                m=classify_batch(vgg_ext, model, frames)
                # print(m)
                o=0
                if(mean(m)>0.5):
                    o=1
                else:
                    o=0 
                if o == label:
                    correct = correct + 1
                count_vid = count_vid + 1

            # get new items
            del frames
            frames =[]
            vid_name = file_edited[i]

            #print(vid_name)
            test_img = transform_fwd(Image.open(os.path.join(path, file_lst[i])))
            frames.append(test_img.unsqueeze(0))


    #classify the last frames
    # cls = classify_batch(vgg_ext, model, frames)
    # if cls == label:
    #     correct = correct + 1
    if(len(frames)==10):
        m=classify_batch(vgg_ext, model, frames)
        o=0
        if(mean(m)>0.5):
            o=1
        else:
            o=0 
        if o == label:
            correct = correct + 1
        count_vid = count_vid + 1
        # print(path)
        # print('Number of files: %d' %(length))
        # print('Number of videos: %d' %(count_vid))
        # print('Number of correct classifications: %d' %(correct))

    return count_vid, correct


if __name__ == '__main__':
    path_real = os.path.join(opt.dataset, opt.real)
    path_fake = os.path.join(opt.dataset, opt.fake)

    # print(os.listdir(path_real))
    vgg_ext = model.VggExtractor()
    acc=[]
    for i in range(76,100):
        lstm = model.LSTM_Model(opt.gpu_id,opt.input_dim, opt.hidden_dim, opt.layer_dim, opt.output_dim)
        lstm.load_state_dict(torch.load(os.path.join(opt.outf,'lstm_' + str(i+1) + '.pt')))
        lstm.eval()

        if opt.gpu_id >= 0:
            vgg_ext.cuda(opt.gpu_id)
            lstm.cuda(opt.gpu_id)

        ###################################################################
        tol_count_vid = 0
        tol_correct = 0

        # real data
        count_vid, correct = classify_frames(vgg_ext, lstm, path_real, 1)
        tol_count_vid = tol_count_vid + count_vid
        tol_correct = tol_correct + correct

        # fake data
        count_vid, correct = classify_frames(vgg_ext, lstm, path_fake, 0)
        tol_count_vid = tol_count_vid + count_vid
        tol_correct = tol_correct + correct

    # print('##################################')
    # print('Number of videos: %d' %(tol_count_vid))
    # print('Number of correct classifications: %d' %(tol_correct))
    # print('Accuracy: %.2f' %(tol_correct/tol_count_vid*100))
        print('epoch : ',i,' , val acc %.2f' %(tol_correct/tol_count_vid*100))
        acc.append((tol_correct/tol_count_vid*100))
    accuracy={'validation Acc.' :acc }
    df = pd.DataFrame(accuracy)
    df.to_csv('val.csv')       