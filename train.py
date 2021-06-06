import sys
sys.setrecursionlimit(15000)
import os
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
from torch.optim import Adam
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
import model


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='dataset', help='path to root dataset')
parser.add_argument('--train_set', default ='train', help='train set')
parser.add_argument('--val_set', default ='validation', help='validation set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=100, help='batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--input_dim', type=int, default=3, help='input dimension of lstm')
parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension of lstm')
parser.add_argument('--layer_dim', type=int, default=1, help='layer dimension of lstm')
parser.add_argument('--output_dim', type=int, default=2, help='output dimension of lstm')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--resume', type=int, default=0, help="choose a epochs to resume from (0 to train from scratch)")
parser.add_argument('--outf', default='checkpoints', help='folder to output model checkpoints')
parser.add_argument('--disable_random', action='store_true', default=False, help='disable randomness for routing matrix')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()


if __name__ == "__main__":

    if opt.gpu_id >= 0:
        cudnn.benchmark = True

    if opt.resume > 0:
        text_writer = open(os.path.join(opt.outf, 'train.csv'), 'a')
    else:
        text_writer = open(os.path.join(opt.outf, 'train.csv'), 'w')


    vgg_ext = model.VggExtractor()
    
    lstm = model.LSTM_Model(opt.gpu_id,opt.input_dim, opt.hidden_dim, opt.layer_dim, opt.output_dim)
    optimizer = Adam(lstm.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    
    if opt.resume > 0:
        lstm.load_state_dict(torch.load(os.path.join(opt.outf,'lstm_' + str(opt.resume) + '.pt')))
        lstm.train(mode=True)
        optimizer.load_state_dict(torch.load(os.path.join(opt.outf,'optim_' + str(opt.resume) + '.pt')))

        if opt.gpu_id >= 0:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(opt.gpu_id)

    if opt.gpu_id >= 0:
        lstm.cuda(opt.gpu_id)
        vgg_ext.cuda(opt.gpu_id)

    transform_fwd = transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    dataset_train = dset.ImageFolder(root=os.path.join(opt.dataset, opt.train_set), transform=transform_fwd)
    assert dataset_train
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

    dataset_val = dset.ImageFolder(root=os.path.join(opt.dataset, opt.val_set), transform=transform_fwd)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))


    error = nn.CrossEntropyLoss()
    if opt.gpu_id >= 0:
        error.cuda(opt.gpu_id)

    for epoch in range(opt.resume+1, opt.niter+1):
        count = 0
        loss_train = 0
        loss_test = 0

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)

        for img_data, labels_data in tqdm(dataloader_train):
            # print(type(img_data))
            # print(img_data)
            # print(labels_data)
            # img_data = np.expand_dims(img_data, axis=1)
            batch_size,c, h, w = img_data.size()
            if(batch_size <20):
                break
            img_data = torch.FloatTensor(img_data)
            # print(img_data.shape,"img_data")

            if torch.cuda.is_available():
                img_label = labels_data.numpy().astype(np.float)
                optimizer.zero_grad()

                if opt.gpu_id >= 0:
                    c_in = img_data.cuda(opt.gpu_id)
                    labels_data = labels_data.cuda(opt.gpu_id)
                input_v = Variable(c_in)
                # print(input_v.shape)
                x = vgg_ext(input_v)
                # print(c_in.shape,"img_data")
                # print(x.shape,"shape &&&&&&&&&&&&&")
                class_ = lstm(x)

                # print(class_,"output predicted")
                loss_dis = error(class_,labels_data)
                loss_dis_data = loss_dis.item()

                loss_dis.backward()
                optimizer.step()
                output_pred = []
                #Pick the class with maximum weight
                for t in class_:
                    if t[0]>t[1]:
                        output_pred.append(0)
                    else:
                        output_pred.append(1)
                output_pred=torch.tensor(output_pred).float()

                tol_label = np.concatenate((tol_label, img_label))
                tol_pred = np.concatenate((tol_pred, output_pred))

                loss_train += loss_dis_data
                count += 1
                # print("hello")


        acc_train = metrics.accuracy_score(tol_label, tol_pred)
        loss_train /= count

        ########################################################################

        # do checkpointing & validation
        torch.save(lstm.state_dict(), os.path.join(opt.outf, 'lstm_%d.pt' % epoch))
        torch.save(optimizer.state_dict(), os.path.join(opt.outf, 'optim_%d.pt' % epoch))

        lstm.eval()

        tol_label = np.array([], dtype=np.float)
        tol_pred = np.array([], dtype=np.float)

        count = 0

        for img_data, labels_data in dataloader_val:
            batch_size,c, h, w = img_data.size()
            if(batch_size <20):
                break
            img_label = labels_data.numpy().astype(np.float)
            img_data = torch.FloatTensor(img_data)
            if opt.gpu_id >= 0:
                c_in = img_data.cuda(opt.gpu_id)
                labels_data = labels_data.cuda(opt.gpu_id)
            input_v = Variable(c_in)

            x = vgg_ext(input_v)
                # print(c_in.shape,"img_data")
            # print("error")
            class_ = lstm(x)

            loss_dis = error(class_,labels_data)
            loss_dis_data = loss_dis.item()
            output_pred = []
                #Pick the class with maximum weight
            for t in class_:
                if t[0]>t[1]:
                    output_pred.append(0)
                else:
                    output_pred.append(1)
            output_pred=torch.tensor(output_pred).float()
 

            tol_label = np.concatenate((tol_label, img_label))
            tol_pred = np.concatenate((tol_pred, output_pred))

            loss_test += loss_dis_data
            count += 1

        acc_test = metrics.accuracy_score(tol_label, tol_pred)
        loss_test /= count

        print('[Epoch %d] Train loss: %.4f   acc: %.2f | Test loss: %.4f  acc: %.2f'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100))

        text_writer.write('%d,%.4f,%.2f,%.4f,%.2f\n'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100))

        text_writer.flush()
        lstm.train(mode=True)

    text_writer.close()