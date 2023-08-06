import torch
import torchvision
from PIL import Image
from torch import nn, tensor

image_path = "imgs/v2iqiw1mvyrv2iqiw1mvyr.jpg"
image = Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()

])

image = transform(image)
image = image.cuda()
print(image.shape)

class Phd(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), (1, 1), 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, (5, 5), (1, 1), 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 5), (1, 1), 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, input):
        output = self.model(input)
        return output

model = torch.load("phd_30_gpu.pth")
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
val = output.argmax(1)
print(val)

my_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
for i in range(9):
    if val == tensor([i], device='cuda:0'):
        print("The object in the picture is {}!".format(my_list[i]))
