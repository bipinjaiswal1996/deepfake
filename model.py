import sys
sys.setrecursionlimit(15000)
import torch
import torch.nn.functional as F
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.models as models

class StatsNet(nn.Module):
    def __init__(self):
        super(StatsNet, self).__init__()

    def forward(self, x):
        x = x.view(x.data.shape[0], x.data.shape[1], x.data.shape[2]*x.data.shape[3])

        mean = torch.mean(x, 2)
        std = torch.std(x, 2)

        return torch.stack((mean, std), dim=1)

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


class VggExtractor(nn.Module):
    def __init__(self):
        super(VggExtractor, self).__init__()

        self.vgg_1 = self.Vgg(models.vgg19(pretrained=True), 0, 18)
        self.vgg_1.eval()

    def Vgg(self, vgg, begin, end):
        features = nn.Sequential(*list(vgg.features.children())[begin:(end+1)])
        return features

    def forward(self, input):
        return self.vgg_1(input)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.ext_1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            StatsNet(),

            nn.Conv1d(2, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1),
            View(-1, 8),
            )

        self.ext_2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            StatsNet(),

            nn.Conv1d(2, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1),
            View(-1, 8),
            )

        self.ext_3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            StatsNet(),

            nn.Conv1d(2, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1),
            View(-1, 8),
            )

    def forward(self, x):
        output_1 = self.ext_1(x.detach())
        output_2 = self.ext_2(x.detach())
        output_3 = self.ext_3(x.detach())

        output = torch.stack((output_1, output_2, output_3), dim=-1)

        return output
        

class LSTM_Model(nn.Module):
    def __init__(self, gpu_id, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM_Model, self).__init__()

        self.fea_ext = FeatureExtractor()
        self.fea_ext.apply(self.weights_init)

        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True) # batch_first=True (batch_dim, seq_dim, feature_dim)

        # Readout layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        # Initialize hidden state with zeros
        
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        if torch.cuda.is_available():
            h0 = h0.cuda() 
            c0 = c0.cuda()
        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        c_out = self.fea_ext(x)
        batch_size, h, w = c_out.size()
        # print(batch_size)
        batch_size=round(batch_size/10)
        # print(batch_size)
        c_out.reshape([batch_size,10,h,w])
        out, (hn, cn) = self.lstm(c_out, (h0.detach(), c0.detach()))
        # print("hello")
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc1(out[:, -1, :])
        x = F.relu(out)
        x = F.dropout(x)
        x = self.fc2(x)
        # Apply softmax to x
        output = F.softmax(x, dim=1)
        return output
