# 5k CIFAR-10训练集 2层cnn 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10  # 示例图像数据集
import swanlab  # 新增：实验跟踪



# 1. 多模态模型框架（同之前）
class MultimodalFramework(nn.Module):
    def __init__(self, text_hidden=768, image_hidden=512, fusion_hidden=256, num_classes=10):
        super(MultimodalFramework, self).__init__()
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.image_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # CIFAR-10 3通道
        self.image_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.image_pool = nn.MaxPool2d(2, 2)
        self.image_fc = nn.Linear(128 * 8 * 8, image_hidden)
        self.fusion_fc = nn.Linear(text_hidden + image_hidden, fusion_hidden)  # 修复打字: fusion_hidden
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(fusion_hidden, num_classes)

    def forward(self, text_input, attention_mask, image_input):
        text_outputs = self.text_encoder(
            input_ids = text_input,
            attention_mask = attention_mask
        ).last_hidden_state  # [batch, seq_len, 768]
        text_features = text_outputs.mean(dim=1)  # 修复: 平均seq_len维, 保持 [batch, 768]
        
        x = self.image_pool(F.relu(self.image_conv1(image_input)))  # [batch, 64, 16, 16]
        x = self.image_pool(F.relu(self.image_conv2(x)))  # [batch, 128, 8, 8]
        x = x.view(x.size(0), -1)  # [batch, 8192]
        image_features = F.relu(self.image_fc(x))  # [batch, 512]
        
        fused = torch.cat([text_features, image_features], dim=1)  # [batch, 1280] - 现在2维, 正常拼接
        fused = self.dropout(F.relu(self.fusion_fc(fused)))  # [batch, 256]
        return self.classifier(fused)  # [batch, 10]



class MultimodalTrainDataset(torch.utils.data.Dataset):
    def __init__(self, root='E:/temp/data', num_samples=50000, transform=None):
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']
        self.dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
        self.num_samples = min(num_samples, len(self.dataset))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = 16

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        assert isinstance(image, torch.Tensor), f"Expected Tensor, got {type(image)}"
        if torch.rand(1).item() < 0.5:
            text_label = label
        else:
            text_label = torch.randint(0, 10, (1,)).item()
        text = f"a photo of a {self.classes[text_label]}"
        encoded = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        return image, encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0), label


# 测试用 Test
class MultimodalTestDataset(torch.utils.data.Dataset):
    def __init__(self, root='E:/temp/data', transform=None):
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']
        self.dataset = CIFAR10(root=root, train=False, download=True, transform=transform)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = 16
        
        # 预生成所有文本
        self.text_inputs = []
        for i in range(len(self.dataset)):
            label = self.dataset.targets[i]
            if torch.rand(1).item() < 0.5:
                text_label = label
            else:
                text_label = torch.randint(0, 10, (1,)).item()
            text = f"a photo of a {self.classes[text_label]}"
            enc = self.tokenizer(
                text,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
            self.text_inputs.append({
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0)
            })

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        text = f"a photo of a {self.classes[label]}"
        assert isinstance(image, torch.Tensor), f"Expected Tensor, got {type(image)}"
        encoded = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        return image, encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0), label




def train_epoch(model, dataloader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images, text_ids, attention_mask, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(text_ids, attention_mask, images)  # [batch, 10]
        # targets = torch.randint(0, 10, (images.size(0),))  # 模拟标签
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        # SwanLab日志（每50步，删掉图像/文本log，只留指标）
        if batch_idx % 50 == 0:
            swanlab.log({
                "epoch": epoch,
                "batch": batch_idx,
                "loss": loss.item(),
                "accuracy": 100. * correct / total,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = 100. * correct / total
    print(f'Epoch {epoch+2}: Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%')
    
    # Epoch末日志
    swanlab.log({
        "epoch_end_loss": avg_loss,
        "epoch_end_acc": avg_acc
    })


if __name__ == '__main__':
    # 初始化SwanLab
    swanlab.init(project="multimodal-object-detection", anonymous=True)  # 改项目名，填token如果需要


    # 2. 示例数据集（CIFAR-10图像 + 模拟文本标签）
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 设置训练集
    dataset = MultimodalTrainDataset(num_samples=50000, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # 训练用 Train

    # 3. 训练循环（加SwanLab日志）
    model = MultimodalFramework()
    criterion = nn.CrossEntropyLoss()


    # 先结冻 BERT 训练6轮
    for param in model.text_encoder.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    for epoch in range(2):
        train_epoch(model, dataloader, optimizer, criterion, epoch)

    # 再解冻 BERT 训练
    for param in model.text_encoder.parameters():
        param.requires_grad = True

    # 4. 重建 optimizer 跑训练（30 epochs测试）
    optimizer = optim.Adam(model.parameters(), lr = 1e-5)
    for epoch in range(2, 5):
        train_epoch(model, dataloader, optimizer, criterion, epoch)

    # 设置测试集 CIFAR-10 Terst
    # 测试集
    test_dataset = MultimodalTestDataset(root='E:/temp/data', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, text_ids, attention_mask, labels in test_loader:
            outputs = model(text_ids, attention_mask, images)
            pred = outputs.argmax(dim = 1)
            test_correct += (pred == labels).sum().item()
            test_total += labels.size(0)

    test_acc = 100. * test_correct / test_total  #  乘法在前
    print(f"Test Accuracy: {test_acc:.2f}%")
    swanlab.log({"test_accuracy": test_acc})

    # 保存模型本地（网页已记录指标）
    torch.save(model.state_dict(), 'multimodal_model_epuch50_1.pth')
    print('模型保存完成！')

    swanlab.finish()  # 结束实验，网页曲线可用