# client.py
import socket
import pickle
from torch.autograd import Variable
from torchvision.models import resnet18
from tqdm import tqdm
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from collections import OrderedDict
import struct
import select
import warnings

warnings.filterwarnings("ignore")

host = 'localhost'
port = 8083
learning_rate = 0.001 
batch_size = 128 
epochs = 2 
partition_id = 0

import torch.nn.init as init 

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        se = self.global_avg_pool(x).view(batch, channels)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se).view(batch, channels, 1, 1)
        return x * se


class WideResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, widen_factor=2):
        super(WideResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * widen_factor, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels * widen_factor)
        self.conv2 = nn.Conv2d(out_channels * widen_factor, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, negative_slope=0.01)
        return out


class TinyViT(nn.Module):
    def __init__(self, embed_dim=64, num_heads=2, num_layers=3, patch_sizes=[4, 8], img_size=32, num_classes=10):
        super(TinyViT, self).__init__()
        self.embed_dim = embed_dim
        self.patch_sizes = patch_sizes
        self.num_classes = num_classes

        self.proj = nn.ModuleList([
            nn.Linear(patch_size * patch_size * 256, embed_dim) for patch_size in patch_sizes
        ])

        self.num_patches = [((img_size // patch_size) ** 2) for patch_size in patch_sizes]
        
        self.positional_encoding = nn.Parameter(torch.zeros(1, sum(self.num_patches), embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.gelu = nn.GELU()
        
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()

        patch_list = []
        for i, patch_size in enumerate(self.patch_sizes):
            patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
            patches = patches.view(batch_size, -1, num_channels * patch_size * patch_size)
            patch_list.append(self.proj[i](patches))
        
        x = torch.cat(patch_list, dim=1)
        x += self.positional_encoding[:, :x.size(1), :]
        
        x = self.transformer(x)
        x = self.gelu(x)
        x = x.mean(dim=1)
        
        x = self.fc(x)
        
        return x


class SimpleNetV4(nn.Module):
    def __init__(self):
        super(SimpleNetV4, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.wide_resnet_block = WideResNetBlock(64, 256, stride=1, widen_factor=1)
        self.se_block = SEBlock(256, reduction=1)

        self.tiny_vit = TinyViT(embed_dim=64, num_heads=4, num_layers=3, patch_sizes=[2, 4, 8], num_classes=10)

        self._initialize_weights()

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = self.pool1(x)

        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = self.pool2(x)

        x = self.wide_resnet_block(x)
        x = self.se_block(x)

        x = self.tiny_vit(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.TransformerEncoderLayer):
                init.xavier_uniform_(m.self_attn.in_proj_weight)
                if m.self_attn.in_proj_bias is not None:
                    init.constant_(m.self_attn.in_proj_bias, 0)
                for param in [m.linear1.weight, m.linear2.weight]:
                    init.xavier_uniform_(param)
                for param in [m.linear1.bias, m.linear2.bias]:
                    if param is not None:
                        init.constant_(param, 0)


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image, label = sample["img"], sample["label"]
        if self.transform:
            image = self.transform(image)
        return image, label

def train(model, criterion, optimizer, train_loader, test_loader):

    best_accuracy = 0.0

    model.to(device)

    for epoch in range(epochs):
        running_corrects = 0
        running_loss = 0.0
        total = 0

        for i, (images, labels) in enumerate(tqdm(train_loader, desc="Train"), 0):
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_corrects.double() / total

        model.eval()
        accuracy = 0.0
        total = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Test"):

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                accuracy += (predicted==labels).sum().item()
        accuracy = (100 * accuracy / total)
        print(f"Epoch [{epoch + 1}/{epochs}] => Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy * 100:.2f}% | Test Accuracy: {accuracy:.2f}%")

    return model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("\nThe model will be running on", device, "device")
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image,label = sample["img"], sample["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_partition_data(alpha, partition_id, path_prefix="CIFAR10", transform=None):
    print(f"Partition ID : {partition_id}")
    if partition_id == 0:
        file_path = f"./data/train_image/partition0/alpha_{alpha}_{path_prefix}_partition_{partition_id}.pt"
    else:
        file_path = f"./data/train_image/partition1/alpha_{alpha}_{path_prefix}_partition_{partition_id}.pt"

    data = torch.load(file_path)
    dataset = CustomDataset(data, transform=transform)

    return dataset

def train(model, criterion, optimizer, train_loader, test_loader):

    best_accuracy = 0.0

    model.to(device)

    for epoch in range(epochs):
        running_corrects = 0
        running_loss = 0.0
        total = 0

        for i, (images, labels) in enumerate(tqdm(train_loader, desc="Train"), 0):
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_corrects.double() / total

        model.eval()
        accuracy = 0.0
        total = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Test"):

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                accuracy += (predicted==labels).sum().item()
        accuracy = (100 * accuracy / total)
        print(f"Epoch [{epoch + 1}/{epochs}] => Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy * 100:.2f}% | Test Accuracy: {accuracy:.2f}%")

    return model

def main():
    model = SimpleNetV4()

    train_dataset = load_partition_data(alpha=1, partition_id=partition_id, path_prefix="CIFAR10", transform=train_transform)

    test_dataset = torch.load('./data/test_image/test_dataset.pt')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))

    while True:
        data_size = struct.unpack('>I', client.recv(4))[0]
        rec_payload = b""

        remaining_payload = data_size
        while remaining_payload != 0:
            rec_payload += client.recv(remaining_payload)
            remaining_payload = data_size - len(rec_payload)
        dict_weight = pickle.loads(rec_payload)
        weight = OrderedDict(dict_weight)
        print("\nReceived updated global model from server")

        model.load_state_dict(weight, strict=True)

        read_sockets, _, _ = select.select([client], [], [], 0)
        if read_sockets:
            print("Federated Learning finished")
            break

        model = train(model, criterion, optimizer, train_loader, test_loader)

        model_data = pickle.dumps(dict(model.state_dict().items()))
        client.sendall(struct.pack('>I', len(model_data)))
        client.sendall(model_data)

        print("Sent updated local model to server.")

if __name__ == "__main__":
    time.sleep(1)
    main()