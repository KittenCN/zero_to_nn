# 导入需要的库
import torch
from torchvision import models, transforms
from PIL import Image

# 加载预训练的模型
model = models.resnet50(pretrained=True)

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载图像
image = Image.open("your_image.jpg")
image = transform(image).unsqueeze(0)

# 使用模型进行预测
model.eval()
with torch.no_grad():
    output = model(image)

# 获取预测的类别
predicted_class = output.argmax(dim=1).item()

# 打印预测的类别
print(f"The image is classified as class {predicted_class}")
