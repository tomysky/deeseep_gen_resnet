import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------- 超参数设置 --------------------
BATCH_SIZE = 24
EPOCHS = 30
LEARNING_RATE = 1e-4
INPUT_SIZE = (128, 128)
DATA_DIR = "./datasets"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- 数据预处理 --------------------
# 数据增强（参考网页6、9）
train_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train():
    # 加载数据集并划分（参考网页5、6）
    dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建DataLoader（参考网页4、8）
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    class_names = dataset.classes
    print(class_names)

    # -------------------- 模型定义 --------------------
    # 加载预训练ResNet101并调整分类头（参考网页9、10）
    model = models.resnet101(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    model = model.to(DEVICE)

    # -------------------- 损失函数与优化器 --------------------
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)


    # -------------------- 训练与验证循环 --------------------
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss, train_correct, total = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]", ncols=100)

        for images, labels in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            total += labels.size(0)
            train_loss += loss.item()

            train_bar.set_postfix({
                'loss': train_loss / len(train_loader),
                'acc': 100 * train_correct / total
            })

        # 记录训练指标
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100 * train_correct / total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)

        # 验证阶段
        model.eval()
        val_loss, val_correct, total = 0.0, 0, 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]", ncols=100)

        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                total += labels.size(0)
                val_loss += loss.item()

                val_bar.set_postfix({
                    'loss': val_loss / len(val_loader),
                    'acc': 100 * val_correct / total
                })

        # 记录验证指标
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100 * val_correct / total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        # 保存最佳模型为ONNX（参考网页13、14）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            dummy_input = torch.randn(1, 3, *INPUT_SIZE).to(DEVICE)
            torch.onnx.export(
                model,
                dummy_input,
                f"models/best_model_epoch{epoch + 1}.onnx",
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    train()

