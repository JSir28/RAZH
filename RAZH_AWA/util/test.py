import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from thop import profile

# 加载 Faster R-CNN 模型并将 backbone 换为 ResNet-101
model = fasterrcnn_resnet50_fpn(pretrained=False)
model.backbone.body = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=False)

# 将模型设置为评估模式
model.eval()

# 创建一个虚拟输入（输入尺寸通常为 800x800）
input = torch.rand(1, 3, 800, 800)

# 使用 thop 计算 FLOPs 和参数量
flops, params = profile(model, inputs=(input,))

# 将结果转换为 G 表示（十亿）
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"参数量: {params / 1e6:.2f} M")
