import torch
from torch import nn
import sys
sys.path.append('train_utils')
import distributed_utils as utils
#-----------------------------
from torchvision import transforms
from PIL import Image
import numpy as np
import json
# palette_path = "./palette.json"
# with open(palette_path, "rb") as f:
#     pallette_dict = json.load(f)
#     pallette = []
#     for v in pallette_dict.values():
#         pallette += v
#----------------------------------------------------------
def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

            # --------------------------------------------------------------
            # model.to(device)
            # img_path = 'D:\Pascal_dataset\VOCdevkit\VOC2012\JPEGImages\\2007_000648.jpg'
            # # load image
            # original_img = Image.open(img_path)
            #
            # # from pil image to tensor and normalize
            # data_transform = transforms.Compose([transforms.Resize(520),
            #                                      transforms.ToTensor(),
            #                                      transforms.Normalize(mean=(0.485, 0.456, 0.406),
            #                                                           std=(0.229, 0.224, 0.225))])
            # img = data_transform(original_img)
            # # expand batch dimension
            # img = torch.unsqueeze(img, dim=0)
            #
            # model.eval()  # 进入验证模式
            # with torch.no_grad():
            #     # init model
            #     img_height, img_width = img.shape[-2:]
            #     init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            #     model(init_img)
            #
            #     output = model(img.to(device))
            #
            #     prediction = output['out'].argmax(1).squeeze(0)
            #     # 通道数即类别数，输出形状为[b,c,h,w]，通过argmax(1)获取每个像素值的类别并以squeeze(0)
            #     # 压缩batch维度得到预测值
            #     prediction = prediction.to("cpu").numpy()
            #     prediction = prediction.astype(np.uint8)
            #     mask = Image.fromarray(prediction)  # 将数值转换成PIL图像
            #     mask.putpalette(pallette)  # 其中pallette为一个列表
            #     # mask.putpalette()是PIL库中Image对象的一个方法，用于设置调色板
            #     # 将mask图像对象进行映射得到彩色标签图
            #     mask.save("test_result.png")  # 保存得到的彩色标签图
            # ---------------------------------------------------------------
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
