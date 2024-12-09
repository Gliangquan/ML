# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt
# from model import AlexNet_v1
# import tensorflow as tf
# import json
# import os
#
#
# def main():
#     # 获取当前工作目录
#     current_dir = os.getcwd()
#
#     # 设置flower_data的路径为当前目录下的flower_data文件夹
#     image_path = os.path.join(current_dir, "flower_data")  # flower data set path
#
#     # 构造训练和验证数据的路径
#     train_dir = os.path.join(image_path, "train")
#     validation_dir = os.path.join(image_path, "val")
#
#     # 确保训练和验证数据的路径存在
#     assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
#     assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)
#
#     # create direction for saving weights
#     if not os.path.exists("save_weights"):
#         os.makedirs("save_weights")
#
#     im_height = 224
#     im_width = 224
#     batch_size = 32
#     epochs = 10
#
#     #  载入图像数据并预处理
#     train_image_generator = ImageDataGenerator(rescale=1. / 255,
#                                                horizontal_flip=True)
#     validation_image_generator = ImageDataGenerator(rescale=1. / 255)
#
#     train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
#                                                                batch_size=batch_size,
#                                                                shuffle=True,
#                                                                target_size=(im_height, im_width),
#                                                                class_mode='categorical')
#     total_train = train_data_gen.n
#
#     # get class dict
#     class_indices = train_data_gen.class_indices
#
#     # transform value and key of dict
#     inverse_dict = dict((val, key) for key, val in class_indices.items())
#     # write dict into json file
#     json_str = json.dumps(inverse_dict, indent=4)
#     with open('class_indices.json', 'w') as json_file:
#         json_file.write(json_str)
#
#     val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
#                                                                   batch_size=batch_size,
#                                                                   shuffle=False,
#                                                                   target_size=(im_height, im_width),
#                                                                   class_mode='categorical')
#     total_val = val_data_gen.n
#     print("using {} images for training, {} images for validation.".format(total_train,
#                                                                            total_val))
#
#     # 载入图像信息
#
#     sample_training_images, sample_training_labels = next(train_data_gen)  # label is one-hot coding
#     #
#     # # This function will plot images in the form of a grid with 1 row
#     # # and 5 columns where images are placed in each column.
#     # def plotImages(images_arr):
#     #     fig, axes = plt.subplots(1, 5, figsize=(20, 20))
#     #     axes = axes.flatten()
#     #     for img, ax in zip(images_arr, axes):
#     #         ax.imshow(img)
#     #         ax.axis('off')
#     #     plt.tight_layout()
#     #     plt.show()
#     #
#     #
#     # plotImages(sample_training_images[:5])
#
#     model = AlexNet_v1(im_height=im_height, im_width=im_width, num_classes=5)
#
#     # model.build((batch_size, 224, 224, 3))  # when using subclass model
#     model.summary()
#
#     # using keras high level api for training
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#                   loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
#                   metrics=["accuracy"])
#
#     callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlex.h5',
#                                                     save_best_only=True,
#                                                     save_weights_only=True,
#                                                     monitor='val_loss')]
#
#     # tensorflow2.1 recommend to using fit
#     history = model.fit(x=train_data_gen,
#                         steps_per_epoch=total_train // batch_size,
#                         epochs=epochs,
#                         validation_data=val_data_gen,
#                         validation_steps=total_val // batch_size,
#                         callbacks=callbacks)
#
#     # plot loss and accuracy image
#     history_dict = history.history
#     train_loss = history_dict["loss"]
#     train_accuracy = history_dict["accuracy"]
#     val_loss = history_dict["val_loss"]
#     val_accuracy = history_dict["val_accuracy"]
#
#     # figure 1
#     plt.figure()
#     plt.plot(range(epochs), train_loss, label='train_loss')
#     plt.plot(range(epochs), val_loss, label='val_loss')
#     plt.legend()
#     plt.xlabel('epochs')
#     plt.ylabel('loss')
#
#     # figure 2
#     plt.figure()
#     plt.plot(range(epochs), train_accuracy, label='train_accuracy')
#     plt.plot(range(epochs), val_accuracy, label='val_accuracy')
#     plt.legend()
#     plt.xlabel('epochs')
#     plt.ylabel('accuracy')
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import AlexNet_v1
import tensorflow as tf
import json
import os


def main():
    # 获取当前工作目录
    current_dir = os.getcwd()

    # 设置flower_data的路径为当前目录下的flower_data文件夹
    image_path = os.path.join(current_dir, "flower_data")  # flower data set path

    # 构造训练和验证数据的路径
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")

    # 确保训练和验证数据的路径存在
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    # 创建保存模型权重的目录
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    im_height = 224  # 图像高度
    im_width = 224   # 图像宽度
    batch_size = 32  # 批处理大小
    epochs = 1      # 训练轮数

    # 载入图像数据并预处理
    train_image_generator = ImageDataGenerator(rescale=1. / 255,  # 将像素值归一化到0到1之间
                                               horizontal_flip=True)  # 随机水平翻转
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # 仅进行归一化

    # 从目录中生成训练数据
    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,  # 随机打乱数据
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical')  # 类别模式为one-hot编码
    total_train = train_data_gen.n  # 训练集总数

    # 获取类别字典
    class_indices = train_data_gen.class_indices

    # 反转字典的键值对
    inverse_dict = dict((val, key) for key, val in class_indices.items())
    # 将字典写入JSON文件
    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 从目录中生成验证数据
    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                  batch_size=batch_size,
                                                                  shuffle=False,  # 不打乱验证数据
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')  # 类别模式为one-hot编码
    total_val = val_data_gen.n  # 验证集总数
    print("使用 {} 张图像进行训练, 使用 {} 张图像进行验证.".format(total_train, total_val))

    # 载入图像信息
    sample_training_images, sample_training_labels = next(train_data_gen)  # label是one-hot编码

    # 绘制训练样本图像
    # def plotImages(images_arr):
    #     fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    #     axes = axes.flatten()
    #     for img, ax in zip(images_arr, axes):
    #         ax.imshow(img)
    #         ax.axis('off')
    #     plt.tight_layout()
    #     plt.show()
    #
    # plotImages(sample_training_images[:5])

    # 创建模型
    model = AlexNet_v1(im_height=im_height, im_width=im_width, num_classes=5)

    # 显示模型结构
    model.summary()

    # 使用Keras高层API进行训练
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Adam优化器
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # 损失函数
                  metrics=["accuracy"])  # 评价指标

    # 设置回调函数以保存最佳模型
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlex.h5',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='val_loss')]

    # 使用fit方法进行模型训练
    history = model.fit(x=train_data_gen,
                        steps_per_epoch=total_train // batch_size,  # 每个epoch的步数
                        epochs=epochs,
                        validation_data=val_data_gen,
                        validation_steps=total_val // batch_size,  # 验证集的步数
                        callbacks=callbacks)

    # 绘制损失和准确率的图像
    history_dict = history.history
    train_loss = history_dict["loss"]  # 训练损失
    train_accuracy = history_dict["accuracy"]  # 训练准确率
    val_loss = history_dict["val_loss"]  # 验证损失
    val_accuracy = history_dict["val_accuracy"]  # 验证准确率

    # 图1：绘制损失
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    # 图2：绘制准确率
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':
    main()
