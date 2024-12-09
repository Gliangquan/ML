# from tensorflow.keras import layers, models, Model, Sequential
#
#
# def AlexNet_v1(im_height=224, im_width=224, num_classes=1000):
#     # tensorflow中的tensor通道排序是NHWC
#     input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")  # output(None, 224, 224, 3)
#     x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)                      # output(None, 227, 227, 3)
#     x = layers.Conv2D(48, kernel_size=11, strides=4, activation="relu")(x)       # output(None, 55, 55, 48)
#     x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 27, 27, 48)
#     x = layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(x)  # output(None, 27, 27, 128)
#     x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 13, 13, 128)
#     x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
#     x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
#     x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 128)
#     x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 6, 6, 128)
#
#     x = layers.Flatten()(x)                         # output(None, 6*6*128)
#     x = layers.Dropout(0.2)(x)
#     x = layers.Dense(2048, activation="relu")(x)    # output(None, 2048)
#     x = layers.Dropout(0.2)(x)
#     x = layers.Dense(2048, activation="relu")(x)    # output(None, 2048)
#     x = layers.Dense(num_classes)(x)                  # output(None, 5)
#     predict = layers.Softmax()(x)
#
#     model = models.Model(inputs=input_image, outputs=predict)
#     return model
from tensorflow.keras import layers, models

# AlexNet_v1模型 卷积神经网络模型，用于图像分类任务。
def AlexNet_v1(im_height=224, im_width=224, num_classes=1000):
    # tensorflow中的tensor通道排序是NHWC
    # 定义输入层，输入图像的形状为(高度, 宽度, 通道数)，这里是224x224的RGB图像
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")  # output(None, 224, 224, 3)

    # 对输入图像进行零填充，使其尺寸变为227x227
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)                      # output(None, 227, 227, 3)

    # 第一个卷积层，使用48个11x11的卷积核，步幅为4，激活函数为ReLU
    x = layers.Conv2D(48, kernel_size=11, strides=4, activation="relu")(x)       # output(None, 55, 55, 48)

    # 第一个池化层，使用3x3的池化核，步幅为2
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 27, 27, 48)

    # 第二个卷积层，使用128个5x5的卷积核，填充方式为same（保持图像尺寸）
    x = layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(x)  # output(None, 27, 27, 128)

    # 第二个池化层
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 13, 13, 128)

    # 第三个卷积层，使用192个3x3的卷积核
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)

    # 第四个卷积层
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)

    # 第五个卷积层
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 128)

    # 第三个池化层
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 6, 6, 128)

    # 将特征图展平为一维向量
    x = layers.Flatten()(x)                         # output(None, 6*6*128)

    # 添加Dropout层，防止过拟合
    x = layers.Dropout(0.2)(x)

    # 第一个全连接层，输出2048维，激活函数为ReLU
    x = layers.Dense(2048, activation="relu")(x)    # output(None, 2048)

    # 添加Dropout层
    x = layers.Dropout(0.2)(x)

    # 第二个全连接层
    x = layers.Dense(2048, activation="relu")(x)    # output(None, 2048)

    # 输出层，输出num_classes个类别的得分
    x = layers.Dense(num_classes)(x)                  # output(None, num_classes)

    # Softmax激活函数用于输出概率
    predict = layers.Softmax()(x)

    # 创建模型
    model = models.Model(inputs=input_image, outputs=predict)
    return model  # 返回构建的模型
