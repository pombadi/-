import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# 定义图像预处理操作
transform = transforms.Compose([
    transforms.Resize([224, 224]),  # 调整图像大小为 224x224
    transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对图像进行标准化
])


# 加载 ResNet-18 模型
def load_resnet18_model(model_path=None, num_classes=4):
    # 使用 weights 参数加载预训练权重
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # 加载预训练权重

    # 修改最后的全连接层，适应 4 类分类任务
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if model_path:
        try:
            model.load_state_dict(torch.load(model_path, weights_only=True))  # 如果有权重文件，加载它
            print("Model loaded successfully from", model_path)
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")

    model.eval()  # 设置为评估模式
    return model


# 图像分类推理函数
def predict_image(image_path, model):
    # 加载图像并进行预处理
    try:
        image = Image.open(image_path)
        image = image.convert("RGB")  # 确保是RGB图像
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    image = transform(image)  # 图像预处理
    image = image.unsqueeze(0)  # 增加batch维度

    # 推理过程
    with torch.no_grad():
        outputs = model(image)  # 获取模型输出
        _, predicted = torch.max(outputs, 1)  # 获取预测类别
        class_idx = predicted.item()
        return class_idx


# 可视化并显示图像
def visualize_prediction(image_path, predicted_class, class_names):
    try:
        image = Image.open(image_path)
        plt.imshow(image)
        plt.title(f"Predicted Class: {class_names[predicted_class]}")
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"Error displaying image: {e}")


# 主函数
def main():
    model_path = r"D:\QQ\智慧医疗实验3\智慧医疗\model_pth\best_resnet18.pth"  # 如果有保存的模型权重文件，指定路径
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist. Using pre-trained model.")

    # 加载模型
    model = load_resnet18_model(model_path=model_path)

    # 图像路径
    image_path = r"new_COVID_19_Radiography_Dataset/test/COVID/images/COVID-13.png"
    if not os.path.exists(image_path):
        print(f"Image path {image_path} does not exist.")
        return

    # 类别标签
    class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

    predicted_class = predict_image(image_path, model)

    if predicted_class is not None:
        print(f"Predicted Class: {class_names[predicted_class]}")
        visualize_prediction(image_path, predicted_class, class_names)


# 运行主函数
if __name__ == "__main__":
    main()
