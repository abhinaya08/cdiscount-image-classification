import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from tqdm import tqdm
from utils import get_state_dict
from label_id_dict import label_to_category_id
from torch.autograd import Variable


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def load_pretrained_model(self, pretrained_model_file=None, skip=[]):
        if pretrained_model_file:
            pretrain_state_dict = get_state_dict(pretrained_model_file)
            state_dict = self.state_dict()
            keys = list(state_dict.keys())
            for key in keys:
                if any(s in key for s in skip):
                    continue
                try:
                    state_dict[key] = pretrain_state_dict[key]
                except KeyError:
                    print("KeyError: {} dosen't lie in pretrain state dict".format(key))
                    continue
        else:
            state_dict = model_zoo.load_url(model_urls[self.name])
        self.load_state_dict(state_dict)
        pass

    def predict(self, data_loader):
        predicts = []
        for data in tqdm(data_loader):
            data = data.cuda() if self.use_cuda else data
            data = Variable(data, volatile=True)
            output = self.forward(data)
            category_label = output.data.max(1)[1]
            for i in range(len(category_label)):
                predicts.append(label_to_category_id[category_label[i]])
        return predicts

    def save(self, file):
        torch.save(self.state_dict(), file)


class ResNet18(ResNet):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes)
        self.name = 'resnet18'


class ResNet34(ResNet):
    def __init__(self, num_classes=1000):
        super(ResNet34, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes)
        self.name = 'resnet34'


class ResNet50(ResNet):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes)
        self.name = 'resnet50'


class ResNet101(ResNet):
    def __init__(self, num_classes=1000):
        super(ResNet101, self).__init__(Bottleneck, [3, 4, 23, 3], num_classes)
        self.name = 'resnet101'


class ResNet152(ResNet):
    def __init__(self, num_classes=1000):
        super(ResNet152, self).__init__(Bottleneck, [3, 8, 36, 3], num_classes)
        self.name = 'resnet152'