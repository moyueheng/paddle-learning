#加载飞桨和相关类库
import numpy as np
import paddle
import paddle.nn.functional as F

# 确保从paddle.vision.datasets.MNIST中加载的图像数据是np.ndarray类型
from PIL import Image

paddle.vision.set_image_backend('cv2')

# 图像归一化函数，将数据范围为[0, 255]的图像归一化到[0, 1]
def norm_img(img):
    # 验证传入数据格式是否正确，img的shape为[batch_size, 28, 28]
    assert len(img.shape) == 3
    batch_size, img_h, img_w = img.shape[0], img.shape[1], img.shape[2]
    # 归一化图像数据
    img = img / 255
    # 将图像形式reshape为[batch_size, 784]
    img = paddle.reshape(img, [batch_size, img_h * img_w])
    return img






# 模型设计
class MINIST(paddle.nn.Layer):
    def __init__(self):
        super(MINIST, self).__init__()
        # 定义一层全连接网络，输出维度是1
        self.fc = paddle.nn.Linear(in_features=784, out_features=1)
    # 定义网络前向传播过程
    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs

# 训练配置
def train():
    # 读取数据
    train_loader = paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='train'), batch_size=16, shuffle=True)
    model = MINIST()
    # 启动训练模式
    model.train()
    # 定义优化器
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters()) # 模型里面产生的参数就是优化器的参数
    EPOCH_NUM = 2
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()): # 这个地方有问题
            images = norm_img(data[0]).astype('float32')
            labels = data[1].astype('float32')

            #前向计算的过程
            predicts = model(images)

            # 计算损失
            loss = F.square_error_cost(predicts, labels)
            avg_loss = paddle.mean(loss)

            #每训练了1000批次的数据，打印下当前Loss的情况
            if batch_id % 1000 == 0:
                print("epoch_id: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))

            #后向传播，更新参数的过程
            avg_loss.backward()
            # 启动优化器
            opt.step()
            # 清空梯度
            opt.clear_grad()
    # 以静态图的形式保存
    paddle.save(model.state_dict(), './mnist.pdparams')


# 读取一张本地的样例图片，转变成模型输入的格式
def load_image(img_path):
    # 从img_path中读取图像，并转为灰度图
    im = Image.open(img_path).convert('L')
    # print(np.array(im))
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, -1).astype(np.float32)
    # 图像归一化，保持和数据集的数据范围一致
    im = 1 - im / 255
    return im

def test():
    # 导入图像读取第三方库
    import numpy as np
    from PIL import Image

    img_path = './imgs/example_0.png'
    # 生成模型实例
    model = MINIST()
    model_dict = paddle.load('./mnist.pdparams')
    model.load_dict(model_dict)
    model.eval() # 转为预测模式

    tensor_img = load_image(img_path)
    result = model(paddle.to_tensor(tensor_img)) # ndarray要变tensor才能进行预测
    print('result',result)
    #  预测输出取整，即为预测的数字，打印结果
    print("本次预测的数字是", result.numpy().astype('int32'))
def main():
    train()
    test()
if __name__ == '__main__':
    main()