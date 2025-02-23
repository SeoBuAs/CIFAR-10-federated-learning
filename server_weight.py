# Laptop code
import threading
import socket
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision.transforms as transforms
import struct
from tqdm import tqdm
import random
import copy
import warnings

warnings.filterwarnings("ignore")

target_accuracy = 80.0  
global_round = 100  

batch_size = 128 
num_samples = 10000   
host = 'localhost'  
port = 8083  

import torch.nn.init as init 

test_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
])

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

        # 가중치 초기화
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


def measure_accuracy(global_model, test_loader): 
    model = SimpleNetV4()
    model.load_state_dict(global_model)
    model.to(device)
    model.eval()

    accuracy = 0.0
    total = 0.0

    inference_start = time.time()
    with torch.no_grad():
        print("\n")
        for inputs, labels in tqdm(test_loader, desc="Test"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

        accuracy = (100 * accuracy / total)
    inference_end = time.time()

    print(f"Inference time for {num_samples} images : {(inference_end - inference_start):.2f} seconds")

    return accuracy, model

cnt = []
models = [] 
semaphore = threading.Semaphore(0)

global_model = None
global_accuracy = 0.0
current_round = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def handle_client(conn, addr, model, test_loader):
    global models, global_model, global_accuracy, current_round, cnt
    print(f"Connected by {addr}")

    while True:
        if len(cnt) < 2:
            cnt.append(1)
            weight = pickle.dumps(dict(model.state_dict().items()))
            conn.send(struct.pack('>I', len(weight)))
            conn.send(weight)

        data_size = struct.unpack('>I', conn.recv(4))[0]
        received_payload = b""
        remaining_payload_size = data_size
        while remaining_payload_size != 0:
            received_payload += conn.recv(remaining_payload_size)
            remaining_payload_size = data_size - len(received_payload)
        model = pickle.loads(received_payload)

        models.append(model)

        if len(models) == 2:
            current_round += 1
            global_model = average_models(models)
            global_accuracy, global_model = measure_accuracy(global_model, test_loader)
            print(f"Global round [{current_round} / {global_round}] Accuracy : {global_accuracy}%")
            get_model_size(global_model)
            models = []
            semaphore.release()
        else:
            semaphore.acquire()

        if (current_round == global_round) or (global_accuracy >= target_accuracy):
            weight = pickle.dumps(dict(global_model.state_dict().items()))
            conn.send(struct.pack('>I', len(weight)))
            conn.send(weight)
            conn.close()
            break
        else:
            weight = pickle.dumps(dict(global_model.state_dict().items()))
            conn.send(struct.pack('>I', len(weight)))
            conn.send(weight)

def get_model_size(global_model):
    model_size = len(pickle.dumps(dict(global_model.state_dict().items())))
    print(f"Model size : {model_size / (1024 ** 2):.4f} MB")

def get_random_subset(dataset, num_samples):
    if num_samples > len(dataset):
        raise ValueError(f"num_samples should not exceed {len(dataset)} (total number of samples in test dataset).")

    indices = random.sample(range(len(dataset)), num_samples)
    subset = Subset(dataset, indices)
    return subset

def average_models(models):
    weight_avg = copy.deepcopy(models[0])

    for key in weight_avg.keys():
        for i in range(1, len(models)):
            weight_avg[key] += models[i][key]
        weight_avg[key] = torch.div(weight_avg[key], len(models))

    return weight_avg


def measure_accuracy(global_model, test_loader):
    model = SimpleNetV4()
    model.load_state_dict(global_model)
    model.to(device)
    model.eval()

    accuracy = 0.0
    total = 0.0

    inference_start = time.time()
    with torch.no_grad():
        print("\n")
        for inputs, labels in tqdm(test_loader, desc="Test"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

        accuracy = (100 * accuracy / total)
    inference_end = time.time()

    print(f"Inference time for {num_samples} images : {(inference_end - inference_start):.2f} seconds")

    return accuracy, model

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen()
    connection = []
    address = []

    test_dataset = torch.load('./data/test_image/test_dataset.pt')

    test_dataset = get_random_subset(test_dataset, num_samples)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Server is listening on {host}:{port}")
    model = SimpleNetV4()

    while len(address) < 2 and len(connection) < 2:
        conn, addr = server.accept()
        connection.append(conn)
        address.append(addr)

    training_start = time.time()

    connection1 = threading.Thread(target=handle_client, args=(connection[0], address[0], model, test_loader))
    connection2 = threading.Thread(target=handle_client, args=(connection[1], address[1], model, test_loader))

    connection1.start();connection2.start()
    connection1.join();connection2.join()

    training_end = time.time()
    total_time = training_end - training_start
    print(f"\nTraining time: {int(total_time // 3600)} hours {int((total_time % 3600) // 60)} minutes {(total_time % 60):.2f} seconds")

    print("Federated Learning finished")


if __name__ == "__main__":
    main()

