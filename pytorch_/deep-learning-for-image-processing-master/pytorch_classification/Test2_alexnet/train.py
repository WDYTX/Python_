import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))#打印是使用CPU还是用GPU设备

    data_transform = {#字典
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),#对图像进行随机处理
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path获得
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])#datasets.ImageFolder加载数据集
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx#根据文件夹里所对应的类别分别构建字典如上
    cla_dict = dict((val, key) for key, val in flower_list.items())#将生成上述字典的键值互换
    #在上述代码中，使用flower_list.items()是为了遍历原始字典flower_list中的键值对。
    # items()是字典对象的一个方法，用于返回一个包含字典所有键值对的可迭代对象。每个键值对都为一个元组，元组的第一个元素是键，第二个元素是值。
    # 在这里，我们需要遍历flower_list字典的所有键值对，将键和值进行交换以构建新的字典。通过调用items()方法，我们可以方便地获取到这些键值对。
    # 具体来说，(val, key) for key, val in flower_list.items()这部分使用了一个生成器表达式。它遍历flower_list.items()返回的可迭代对象
    # 将每个键值对进行处理。key变量用于表示元组中的键，val变量用于表示元组中的值。
    # 在生成器表达式中，我们通过(val, key)将原始键值对进行了交换，将原始字典中的值作为新字典的键，将原始字典中的键作为新字典的值。
    # 最后，通过dict(...)将生成器表达式的结果转换为字典。
    # 因此，使用items()方法是为了能够方便地遍历字典的键值对，并进行进一步的操作，例如交换键值对构建新的字典。
    json_str = json.dumps(cla_dict, indent=4)# 将cla_dict字典转变为json
    with open('class_indices.json', 'w') as json_file:#'w' 参数指示文件的打开模式为写入（write）模式
        # 如果文件不存在就创建一个新文件，如果文件已经存在则先清空文件内容，然后可以写入新的内容。
        # with语句可以帮助自动管理文件的打开和关闭，它会在代码块结束时自动关闭文件，无论是否发生异常。
        json_file.write(json_str)#把json_str写入class_indices.json文件
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 10
    save_path = './AlexNet.pth'#训练好的网络模型的路径
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()#梯度清零
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)#保存模型中最好的参数

    print('Finished Training')


if __name__ == '__main__':
    main()
