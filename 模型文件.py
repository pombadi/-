import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理变换
transform_train = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
trainset = datasets.ImageFolder(root=os.path.join('new_COVID_19_Radiography_Dataset', 'train'),
                                transform=transform_train)
testset = datasets.ImageFolder(root=os.path.join('new_COVID_19_Radiography_Dataset', 'val'),
                               transform=transform_test)

# 创建数据加载器
train_loader = DataLoader(trainset, batch_size=32, num_workers=4, shuffle=True, pin_memory=True)
test_loader = DataLoader(testset, batch_size=32, num_workers=4, shuffle=False, pin_memory=True)

# 使用预训练的 ResNet18 模型
class PretrainedResNet(nn.Module):
    def __init__(self, num_classes=4):
        super(PretrainedResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 定义训练函数
def train(model, train_loader, criterion, optimizer, num_epochs=100):
    best_accuracy = 0.0  # 保存最佳准确率
    train_losses, val_losses, accuracies = [], [], []  # 存储训练损失、验证损失和准确率

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        # 训练阶段
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})

        # 计算训练损失和准确率
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = 100.0 * correct_preds / total_preds
        train_losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # 在每个epoch结束后评估模型
        accuracy, val_loss = evaluate(model, test_loader, criterion)
        val_losses.append(val_loss)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, save_path)
            print("Model saved with best accuracy:", best_accuracy)

    # 绘制训练过程中的损失和准确率曲线
    plot_metrics(train_losses, val_losses, accuracies)

# 评估模型
def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = 100.0 * correct / total

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    return accuracy, avg_loss

# 保存模型
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

# 绘制训练过程中的损失和准确率图
def plot_metrics(train_losses, val_losses, accuracies):
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss during Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy')
    plt.title('Accuracy during Training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    num_epochs = 5  # 训练轮数
    learning_rate = 0.001  # 学习率
    num_classes = 4
    save_path = "model_pth/best_resnet18.pth"  # 模型保存路径

    model = PretrainedResNet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 优化器

    # 训练模型
    train(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

    # 评估模型
    evaluate(model, test_loader, criterion)
