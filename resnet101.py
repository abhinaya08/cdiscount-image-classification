import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
from torch.autograd import Variable

BN_EPS = 1e-5


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1):
        super(ConvBn2d, self).__init__()
        self.is_bn = True

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                              stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)

    def forward(self, x):
        x = self.conv(x)
        if self.is_bn:
            x = self.bn(x)
        return x

    def merge_bn(self):
        assert(self.conv.bias == None)
        conv_weight = self.conv.weight.data
        bn_weight = self.bn.weight.data
        bn_bias = self.bn.bias.data
        bn_running_mean = self.bn.running_mean
        bn_running_var = self.bn.running_var
        bn_eps = self.bn.eps

        # https://github.com/sanghoon/pva-faster-rcnn/issues/5
        # https://github.com/sanghoon/pva-faster-rcnn/commit/39570aab8c6513f0e76e5ab5dba8dfbf63e9c68c

        N, C, KH, KW = conv_weight.size()
        std = 1/(torch.sqrt(bn_running_var + bn_eps))
        std_bn_weight =(std * bn_weight).repeat(C * KH * KW, 1).t().contiguous().view(N, C, KH, KW )
        conv_weight_hat = std_bn_weight * conv_weight
        conv_bias_hat = (bn_bias - bn_weight * std * bn_running_mean)

        self.is_bn = False
        self.bn = None
        self.conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                              kernel_size=self.conv.kernel_size, padding=self.conv.padding, stride=self.conv.stride,
                              dilation=self.conv.dilation, groups=self.conv.groups, bias=True)
        self.conv.weight.data = conv_weight_hat  # fill in
        self.conv.bias.data = conv_bias_hat


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, out_planes, is_downsample=False, stride=1):
        super(Bottleneck, self).__init__()
        self.is_downsample = is_downsample

        self.conv_bn1 = ConvBn2d(in_planes, planes, kernel_size=1, padding=0, stride=1)
        self.conv_bn2 = ConvBn2d(planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv_bn3 = ConvBn2d(planes, out_planes, kernel_size=1, padding=0, stride=1)

        if self.is_downsample:
            self.downsample = ConvBn2d(in_planes, out_planes, kernel_size=1, padding=0, stride=stride)

    def forward(self, x):
        z = self.conv_bn1(x)
        z = F.relu(z, inplace=True)
        z = self.conv_bn2(z)
        z = F.relu(z, inplace=True)
        z = self.conv_bn3(z)
        if self.is_downsample:
            z += self.downsample(x)
        else:
            z += x
        z = F.relu(z, inplace=True)
        return z


def make_layer(in_planes, planes, out_planes, num_blocks, stride):
    layers = []
    layers.append(Bottleneck(in_planes, planes, out_planes, is_downsample=True, stride=stride))
    for i in range(1, num_blocks):
        layers.append(Bottleneck(out_planes, planes, out_planes))
    return nn.Sequential(*layers)


def make_layer0(in_channels, out_planes):
    layers = [
        ConvBn2d(in_channels, out_planes, kernel_size=7, stride=2, padding=3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    ]
    return nn.Sequential(*layers)


class ResNet101(nn.Module):
        # torch.save(state_dict,save_model_file)
    def __init__(self, in_shape=(3, 180, 180), num_classes=5270):
        super(ResNet101, self).__init__()
        in_channels, height, width = in_shape
        self.num_classes = num_classes

        self.layer0 = make_layer0(in_channels, 64)
        self.layer1 = make_layer(64,  64,  256, num_blocks=3, stride=1)  # out =  64*4 =  256
        self.layer2 = make_layer(256, 128,  512, num_blocks=4, stride=2)  # out = 128*4 =  512
        self.layer3 = make_layer(512, 256, 1024, num_blocks=23, stride=2)  # out = 256*4 = 1024
        self.layer4 = make_layer(1024, 512, 2048, num_blocks=3, stride=2)  # out = 512*4 = 2048
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # logits

    def merge_bn(self):
        print('merging bn ....')
        for name, m in self.named_modules():
            try:
                m.merge_bn()
                print('\t%s' % name)
            except:

            #if isinstance(m, ConvBn2d):
                print(name, isinstance(m, ConvBn2d))
            #    print('\t%s' % name)
             #   m.merge_bn()
        print('')

    def load_pretrain_file(self, pretrain_file, skip=[]):
        try:
            pretrain_state_dict = torch.load(pretrain_file)
        except AssertionError:
            pretrain_state_dict = torch.load(pretrain_file, map_location=lambda storage, location: storage)

        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip):
                continue
            pretrain_key = key
            if 'layer0.0.conv.' in key:
                pretrain_key = key.replace('layer0.0.conv.', 'conv1.')
            if 'layer0.0.bn.' in key:
                pretrain_key = key.replace('layer0.0.bn.', 'bn1.')
            if '.conv_bn1.conv.' in key:
                pretrain_key = key.replace('.conv_bn1.conv.', '.conv1.')
            if '.conv_bn1.bn.' in key:
                pretrain_key = key.replace('.conv_bn1.bn.', '.bn1.')
            if '.conv_bn2.conv.' in key:
                pretrain_key = key.replace('.conv_bn2.conv.', '.conv2.')
            if '.conv_bn2.bn.' in key:
                pretrain_key = key.replace('.conv_bn2.bn.', '.bn2.')
            if '.conv_bn3.conv.' in key:
                pretrain_key = key.replace('.conv_bn3.conv.', '.conv3.')
            if '.conv_bn3.bn.' in key:
                pretrain_key = key.replace('.conv_bn3.bn.', '.bn3.')
            if '.downsample.conv.' in key:
                pretrain_key = key.replace('.downsample.conv.', '.downsample.0.')
            if '.downsample.bn.' in key:
                pretrain_key = key.replace('.downsample.bn.', '.downsample.1.')
            state_dict[key] = pretrain_state_dict[pretrain_key]
        self.load_state_dict(state_dict)


if __name__ == '__main__':

    def run_check_net_imagenet():
        num_classes = 1000
        C, H, W = 3, 224, 224
        net = ResNet101(in_shape=(C, H, W), num_classes=num_classes)
        net.load_pretrain_file('./saved_models/resnet101-5d3b4d8f.pth')
        net.eval()

        image = cv2.imread('./data/lufei.jpg')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224)).astype(np.float32)
        image = image.transpose((2, 0, 1))
        image = image / 255

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image[0] = (image[0] - mean[0]) / std[0]
        image[1] = (image[1] - mean[1]) / std[1]
        image[2] = (image[2] - mean[2]) / std[2]

        logits = net(Variable(torch.from_numpy(image).unsqueeze(0).float()))
        probs = F.softmax(logits).data.numpy().reshape(-1)
        print('results ', np.argmax(probs), 'w.p.', probs[np.argmax(probs)])


    def run_check_net():
        # https://discuss.pytorch.org/t/print-autograd-graph/692/8
        batch_size = 1
        num_classes = 5270
        C, H, W = 3, 180, 180

        inputs = torch.randn(batch_size, C, H, W)
        labels = torch.randn(batch_size, num_classes)
        in_shape = inputs.size()[1:]

        net = ResNet101(in_shape=in_shape, num_classes=num_classes)
        net.load_pretrain_file('./saved_models/resnet101-5d3b4d8f.pth', skip=['fc.'])
        net.train()

        x = Variable(inputs)
        y = Variable(labels)
        logits = net.forward(x)
        probs = F.softmax(logits)
        loss = F.binary_cross_entropy(logits, y)
        loss.backward()

        # print(net)
        # print(probs)

        # merging ----
        net.eval()
        net.merge_bn()


    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_net()
