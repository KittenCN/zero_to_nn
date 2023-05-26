import torch
import torchvision

# 加载预训练的模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# 设置模型为评估模式
model.eval()

from PIL import Image
from torchvision.transforms import ToTensor

# 加载图像
image = Image.open("test.jpg")

# 将图像转换为PyTorch张量
image = ToTensor()(image)

# 添加一个额外的维度
image = image.unsqueeze(0)

# 使用模型进行预测
with torch.no_grad():
    prediction = model(image)

# 打印预测结果
print(prediction)
