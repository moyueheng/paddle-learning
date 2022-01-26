from PIL import Image


def show_each_layer():
    # 输入数据形状是 [N, 1, H, W]
    # 这里用np.random创建一个随机数组作为输入数据
    x = np.random.randn(*[3,1,28,28])
    x = x.astype('float32')

    # 创建LeNet类的实例，指定模型名称和分类的类别数目
    model = LeNet(num_classes=10)
    # 通过调用LeNet从基类继承的sublayers()函数，
    # 查看LeNet中所包含的子层
    print(model.sublayers())

    x = paddle.to_tensor(x)
    for item in model.sublayers():
        # item是LeNet类中的一个子层
        # 查看经过子层之后的输出数据形状
        try:
            x = item(x)
        except:
            x = paddle.reshape(x, [x.shape[0], -1])
            x = item(x)
        if len(item.parameters())==2:
            # 查看卷积和全连接层的数据和参数的形状，
            # 其中item.parameters()[0]是权重参数w，item.parameters()[1]是偏置参数b
            print(item.full_name(), x.shape, item.parameters()[0].shape, item.parameters()[1].shape)
        else:
            # 池化层没有参数
            print(item.full_name(), x.shape)


# -*- coding: utf-8 -*-
# LeNet 识别手写数字

# 导入需要的包
import paddle
import numpy as np
from paddle.nn import Conv2D, MaxPool2D, Linear

## 组网
import paddle.nn.functional as F

# 定义 LeNet 网络结构
class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(LeNet, self).__init__()
        # 创建卷积和池化层
        # 创建第1个卷积层
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 尺寸的逻辑：池化层未改变通道数；当前通道数为6
        # 创建第2个卷积层
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 创建第3个卷积层
        self.conv3 = Conv2D(in_channels=16, out_channels=120, kernel_size=4)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        # 输入size是[28,28]，经过三次卷积和两次池化之后，C*H*W等于120
        # 如果输入size是[28,28], 三次卷积和两次池化之后刚好变成了 120*1*1
        self.fc1 = Linear(in_features=120, out_features=64)
        # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
        self.fc2 = Linear(in_features=64, out_features=num_classes)

    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        # 每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        x = F.sigmoid(x)
        x = self.max_pool1(x)
        x = F.sigmoid(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x) # 120*1*1
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x



import os
import random
import paddle
import numpy as np
import paddle
from paddle.vision.transforms import ToTensor
from paddle.vision.datasets import MNIST
EPOCH_NUM = 5

# 定义模型训练过程
def train(model: LeNet, opt: paddle.optimizer.Momentum, train_loader: paddle.io.DataLoader, valid_loader: paddle.io.DataLoader):
    use_gpu = True
    paddle.device.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    print('start_traing......')
    model.train()
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader):
            img = data[0] # 10，1，28，28
            label = data[1] # [10, 1]
            # 计算模型输出
            logits = model(img) # [10, 10]
            # 计算损失函数
            loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits, label) # [10, 1]
            avg_loss = paddle.mean(loss)
            if batch_id % 2000 == 0:
                print('epoch: {}, batch_id: {}, loss is: {:.4f}'.format(epoch, batch_id, float(avg_loss.numpy())))
            avg_loss.backward() # 反向传播
            opt.step() # 优化器启动
            opt.clear_grad()

        model.eval() # 转成预测模式
        accuracies = []
        losses = []

        for batch_id, data in enumerate(valid_loader()):
            img = data[0]
            label = data[1]
            # 计算模型输出
            logits = model(img)
            pred = F.softmax(logits)
            # 计算损失函数
            loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits, label)
            acc = paddle.metric.accuracy(pred, label)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())

        print("[validation] accuracy/loss: {:.4f}/{:.4f}".format(np.mean(accuracies), np.mean(losses)))
        model.train() # 转回训练模式

    # 保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')



def main():
    # 创建模型
    model = LeNet(num_classes=10)
    # 设置迭代轮数
    EPOCH_NUM = 5
    opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
    train_loader = paddle.io.DataLoader(MNIST(mode='train', transform=ToTensor()), batch_size=10,shuffle=True )
    valid_loader = paddle.io.DataLoader(MNIST(mode='test', transform=ToTensor()), batch_size=10)
    train(model, opt, train_loader, valid_loader)


# 读取一张本地的样例图片，转变成模型输入的格式
def load_image(img_path):
    """

    :param img_path:
    :return: 1，28，28
    """
    # 从img_path中读取图像，并转为灰度图
    im = Image.open(img_path).convert('L')
    # print(np.array(im))
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    # 图像归一化，保持和数据集的数据范围一致
    im =  im / 255
    return im


def test():

    # 导入图像读取第三方库
    import numpy as np
    from PIL import Image

    img_path = './imgs/example_0.png'
    # 生成模型实例
    model = LeNet()
    model_dict = paddle.load('./mnist.pdparams')
    model.load_dict(model_dict)
    model.eval() # 转为预测模式

    tensor_img = load_image(img_path) # 1，784归一化的数据
    result = model(paddle.to_tensor(tensor_img)) # ndarray要变tensor才能进行预测
    print('result',result)
    #  预测输出取整，即为预测的数字，打印结果
    print("本次预测的数字是", result.numpy().astype('int32'))

    pass


if __name__ == '__main__':
    # show_each_layer()
    # main()
    test()