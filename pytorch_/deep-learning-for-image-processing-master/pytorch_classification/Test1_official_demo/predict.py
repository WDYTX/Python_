import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = LeNet()
    net.load_state_dict(torch.load('Lenet.pth'))#将参数加载到网络模型中

    im = Image.open('img.png')#PIL导入通道[H,W,C]
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()#通过numpy将tensor转换成普通列表
    print(classes[int(predict)])
if __name__ == '__main__':
    main()
