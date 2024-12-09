import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from model import AlexNet_v1

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    im_height = 224  # 图像高度
    im_width = 224   # 图像宽度

    # 加载图像
    img_path = "tulip.jpg"
    assert os.path.exists(img_path), "文件: '{}' 不存在.".format(img_path)  # 检查文件是否存在
    img = Image.open(img_path)  # 打开图像文件

    # 将图像调整为224x224的大小
    img = img.resize((im_width, im_height))
    plt.imshow(img)  # 显示原始图像

    # 将像素值缩放到(0-1)范围
    img = np.array(img) / 255.0

    # 将图像添加到一个批次中，这里只有一张图像
    img = np.expand_dims(img, 0)

    # 读取类别索引
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "文件: '{}' 不存在.".format(json_path)  # 检查文件是否存在

    with open(json_path, "r", encoding="utf-8") as f:  # 指定编码为utf-8
        class_indict = json.load(f)  # 读取类别索引文件

    # 创建模型
    model = AlexNet_v1(num_classes=5)  # 假设模型有5个类别
    weighs_path = "./save_weights/myAlex.h5"  # 权重文件路径
    assert os.path.exists(weighs_path), "文件: '{}' 不存在.".format(weighs_path)  # 检查权重文件是否存在
    model.load_weights(weighs_path)  # 加载模型权重

    # 进行预测
    result = np.squeeze(model.predict(img))  # 预测结果
    predict_class = np.argmax(result)  # 获取预测的类别索引

    # 打印预测结果
    print_res = "类别: {}   概率: {:.3}".format(class_indict[str(predict_class)],
                                                 result[predict_class])
    plt.title(print_res)  # 设置图像标题为预测结果
    for i in range(len(result)):
        print("类别: {:10}   概率: {:.3}".format(class_indict[str(i)],
                                                  result[i]))  # 打印所有类别及其对应的概率
    plt.show()  # 显示图像及其预测结果

if __name__ == '__main__':
    main()  # 运行主程序
