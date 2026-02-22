import jittor as jt
import jittor.nn as nn
import numpy as np

# 修复conv2d函数参数类型不匹配的问题
# 重写nn.Conv2d类，确保参数类型正确
class FixedConv2d(nn.Conv2d):
    def execute(self, x):
        # 使用父类的execute方法，避免直接调用nn.conv2d
        return super().execute(x)

def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, FixedConv2d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.fill_(1)
        module.bias.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.zero_()

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        #init.constant(m.bias, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def execute(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.relu(out)
        return out

    # 添加forward方法，确保与PyTorch兼容
    def forward(self, x):
        return self.execute(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def execute(self, x):
        # 打印输入形状
        print(f"Bottleneck input shape: {x.shape}")
        # 第一个卷积层
        out = self.conv1(x)
        print(f"After conv1 shape: {out.shape}")
        out = self.bn1(out)
        out = nn.relu(out)
        # 第二个卷积层
        out = self.conv2(out)
        print(f"After conv2 shape: {out.shape}")
        out = self.bn2(out)
        out = nn.relu(out)
        # 第三个卷积层
        out = self.conv3(out)
        print(f"After conv3 shape: {out.shape}")
        out = self.bn3(out)
        #  shortcut
        shortcut_out = self.shortcut(x)
        print(f"Shortcut shape: {shortcut_out.shape}")
        out += shortcut_out
        out = nn.relu(out)
        print(f"Bottleneck output shape: {out.shape}")
        return out

    # 添加forward方法，确保与PyTorch兼容
    def forward(self, x):
        return self.execute(x)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_input_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def execute(self, x):
        # 打印输入形状
        print(f"ResNet input shape: {x.shape}")
        # 第一层卷积
        out = self.conv1(x)
        print(f"After conv1 shape: {out.shape}")
        out = self.bn1(out)
        out = nn.relu(out)
        # 各层
        print("Before layer1")
        out = self.layer1(out)
        print(f"After layer1 shape: {out.shape}")
        print("Before layer2")
        out = self.layer2(out)
        print(f"After layer2 shape: {out.shape}")
        print("Before layer3")
        out = self.layer3(out)
        print(f"After layer3 shape: {out.shape}")
        print("Before layer4")
        out = self.layer4(out)
        print(f"After layer4 shape: {out.shape}")
        # 池化和全连接
        out = nn.avg_pool2d(out, 4)
        print(f"After avg_pool2d shape: {out.shape}")
        out = out.reshape(out.shape[0], -1)
        print(f"After reshape shape: {out.shape}")
        out = self.linear(out)
        print(f"ResNet output shape: {out.shape}")
        return out

    # 添加forward方法，确保与PyTorch兼容
    def forward(self, x):
        return self.execute(x)


def resnet18(num_classes=10, num_input_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, num_input_channels)


def resnet34(num_classes=10,num_input_channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, num_input_channels)


def resnet50(num_classes=10, num_input_channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, num_input_channels)


def resnet101(num_classes=10, num_input_channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, num_input_channels)


def resnet152(num_classes=10,num_input_channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, num_input_channels)


def test():
    net = resnet18()
    y = net(jt.randn(1,3,32,32))
    print(y.shape)
