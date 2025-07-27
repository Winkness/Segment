import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

# 确保CUDA可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据集路径
data_dir = "Data2/压缩/Data3/archive"

# 自定义数据集加载
class RoadDataset(Dataset):
    def __init__(self, folder, img_size=(256, 256), shuffle=True):
        self.image_dir = os.path.join(folder, "image")
        self.label_dir = os.path.join(folder, "label")
        self.image_paths = sorted([f for f in os.listdir(self.image_dir) if not f.startswith('.')])
        self.label_paths = sorted([f for f in os.listdir(self.label_dir) if not f.startswith('.')])
        self.img_size = img_size
        self.shuffle = shuffle
        
        # 数据转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # 打乱数据
        if self.shuffle:
            indices = np.random.permutation(len(self.image_paths))
            self.image_paths = [self.image_paths[i] for i in indices]
            self.label_paths = [self.label_paths[i] for i in indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        label_path = os.path.join(self.label_dir, self.label_paths[idx])

        # 加载图像并检查是否为空
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        if label is None:
            raise ValueError(f"Failed to load label at {label_path}")

        # 调整大小和归一化
        image = cv2.resize(image, self.img_size) / 255.0
        label = cv2.resize(label, self.img_size) / 255.0
        
        # 转换为Tensor (PyTorch默认通道在前)
        image = self.transform(image).float()
        label = torch.from_numpy(label).unsqueeze(0).float()  # 添加通道维度
        
        return image, label


# 定义 Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(ResidualBlock, self).__init__()
        self.shortcut = nn.Conv2d(in_channels, filters, kernel_size=1, padding='same')
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(filters)
        self.add = nn.Add()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.add([shortcut, x])
        x = self.relu2(x)
        return x


# 引入深度可分离卷积和多尺度卷积
class LocalStreamAdvanced(nn.Module):
    def __init__(self, in_channels, filters):
        super(LocalStreamAdvanced, self).__init__()
        # 多尺度卷积
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, padding='same', dilation=1)
        self.conv2 = nn.Conv2d(in_channels, filters, kernel_size=3, padding='same', dilation=2)
        self.conv3 = nn.Conv2d(in_channels, filters, kernel_size=5, padding='same', dilation=1)
        
        # 深度可分离卷积
        self.depthwise = nn.Conv2d(filters*3, filters*3, kernel_size=3, padding='same', groups=filters*3)
        self.pointwise = nn.Conv2d(filters*3, filters, kernel_size=1)
        self.bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差连接
        self.shortcut = nn.Conv2d(in_channels, filters, kernel_size=1, padding='same')
        self.add = nn.Add()
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 多尺度卷积
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        conv3_out = self.conv3(x)
        
        # 合并多尺度特征
        x_multi_scale = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)
        
        # 深度可分离卷积
        x_separable = self.depthwise(x_multi_scale)
        x_separable = self.pointwise(x_separable)
        x_separable = self.bn(x_separable)
        x_separable = self.relu(x_separable)
        
        # 残差连接
        shortcut = self.shortcut(x)
        x = self.add([shortcut, x_separable])
        x = self.final_relu(x)
        return x


# Swin Transformer Block 的实现
class SwinTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=8, shift_size=4, dropout=0.1):
        super(SwinTransformerBlock, self).__init__()
        assert 0 <= shift_size < window_size, "shift_size must be in 0 <= shift_size < window_size"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_path = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def window_partition(self, x):
        # x: (batch, height, width, channels) -> (batch, num_windows, window_size*window_size, channels)
        batch, height, width, channels = x.shape
        x = x.view(batch, height // self.window_size, self.window_size, 
                  width // self.window_size, self.window_size, channels)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (batch, num_windows_h, num_windows_w, window_size, window_size, channels)
        windows = windows.view(batch, -1, self.window_size * self.window_size, channels)  # (batch, num_windows, window_size*window_size, channels)
        return windows

    def window_reverse(self, windows, height, width):
        # windows: (batch, num_windows, window_size*window_size, channels) -> (batch, height, width, channels)
        batch = windows.shape[0]
        num_windows_h = height // self.window_size
        num_windows_w = width // self.window_size
        
        x = windows.view(batch, num_windows_h, num_windows_w, self.window_size, self.window_size, self.embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (batch, num_windows_h, window_size, num_windows_w, window_size, channels)
        x = x.view(batch, height, width, self.embed_dim)
        return x

    def forward(self, x):
        # x: (batch, channels, height, width) -> 转换为 (batch, height, width, channels)
        x = x.permute(0, 2, 3, 1).contiguous()
        batch, height, width, channels = x.shape
        
        # 移位窗口
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # 窗口划分
        windows = self.window_partition(shifted_x)  # (batch, num_windows, window_size*window_size, channels)
        batch_windows, num_windows, window_size_sq, channels = windows.shape
        windows = windows.view(-1, window_size_sq, channels)  # (batch*num_windows, window_size*window_size, channels)
        
        # 多头自注意力
        windows_norm = self.norm1(windows)
        attn_output, _ = self.attn(windows_norm, windows_norm, windows_norm)
        attn_output = self.drop_path(attn_output)
        windows = windows + attn_output  # 残差连接
        
        # MLP
        windows_norm = self.norm2(windows)
        mlp_output = self.mlp(windows_norm)
        windows = windows + mlp_output  # 残差连接
        
        # 恢复窗口
        windows = windows.view(batch_windows, num_windows, window_size_sq, channels)
        x = self.window_reverse(windows, height, width)  # (batch, height, width, channels)
        
        # 逆移位
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        # 转回 (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# 图神经网络（GNN）模块的实现（Graph Attention Network）
class GraphAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout_rate=0.1):
        super(GraphAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (batch, num_nodes, embed_dim)
        x_norm = self.norm(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        attn_output = self.dropout(attn_output)
        x = self.activation(x + attn_output)  # 残差连接和激活
        return x


# 可学习的空间注意力
class LearnableSpatialAttention(nn.Module):
    def __init__(self):
        super(LearnableSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(None, 1, kernel_size=1, activation='sigmoid')

    def forward(self, x):
        attention_map = self.conv(x)  # (batch, 1, height, width)
        return x * attention_map  # 广播相乘


# 自定义 GlobalStreamAdvanced 层
class GlobalStreamAdvanced(nn.Module):
    def __init__(self, in_channels, embed_dim=64, num_heads=4, window_size=8, shift_size=4):
        super(GlobalStreamAdvanced, self).__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1, padding='same')
        self.swin_block = SwinTransformerBlock(embed_dim, num_heads, window_size=window_size, shift_size=shift_size)
        self.gnn_block = GraphAttentionBlock(embed_dim, num_heads=num_heads)

    def forward(self, x):
        # x: (batch, channels, height, width)
        x = self.conv(x)  # (batch, embed_dim, height, width)
        x = self.swin_block(x)  # (batch, embed_dim, height, width)
        
        # 转换为图格式: (batch, num_nodes, embed_dim)
        batch_size, embed_dim, height, width = x.shape
        num_nodes = height * width
        x_graph = x.view(batch_size, embed_dim, num_nodes).permute(0, 2, 1).contiguous()  # (batch, num_nodes, embed_dim)
        
        # GNN模块
        x_graph = self.gnn_block(x_graph)  # (batch, num_nodes, embed_dim)
        
        # 转换回特征图: (batch, embed_dim, height, width)
        x = x_graph.permute(0, 2, 1).contiguous().view(batch_size, embed_dim, height, width)
        return x


# Global Stream 架构
def global_stream(x, in_channels, embed_dim=64, num_heads=4, window_size=8, shift_size=4):
    global_stream_layer = GlobalStreamAdvanced(in_channels, embed_dim=embed_dim, num_heads=num_heads, 
                                              window_size=window_size, shift_size=shift_size)
    return global_stream_layer(x)


# 定义融合模块
def fusion_module(local_features, global_features):
    # 拼接
    return torch.cat([local_features, global_features], dim=1)


# 定义双流网络的U-Net模型
class DualStreamUNet(nn.Module):
    def __init__(self, input_channels=3):
        super(DualStreamUNet, self).__init__()
        
        # 下采样部分
        self.c1 = ResidualBlock(input_channels, 64)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c2 = ResidualBlock(64, 128)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 局部和全局流特征提取
        self.c3_local = LocalStreamAdvanced(128, 256)
        self.c3_global = GlobalStreamAdvanced(128, embed_dim=64, num_heads=4, window_size=8, shift_size=4)
        self.c3_fused = nn.Conv2d(256 + 64, 256, kernel_size=1, padding='same')  # 融合后调整通道数
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c4_local = LocalStreamAdvanced(256, 512)
        self.c4_global = GlobalStreamAdvanced(256, embed_dim=64, num_heads=4, window_size=8, shift_size=4)
        self.c4_fused = nn.Conv2d(512 + 64, 512, kernel_size=1, padding='same')
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c5_local = LocalStreamAdvanced(512, 1024)
        self.c5_global = GlobalStreamAdvanced(512, embed_dim=64, num_heads=4, window_size=8, shift_size=4)
        self.c5_fused = nn.Conv2d(1024 + 64, 1024, kernel_size=1, padding='same')
        
        # 上采样部分
        self.u4_up = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0)
        self.u4_concat = nn.Conv2d(512 + 512, 512, kernel_size=1, padding='same')
        self.u4 = ResidualBlock(512, 512)
        
        self.u3_up = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.u3_concat = nn.Conv2d(256 + 256, 256, kernel_size=1, padding='same')
        self.u3 = ResidualBlock(256, 256)
        
        self.u2_up = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.u2_concat = nn.Conv2d(128 + 128, 128, kernel_size=1, padding='same')
        self.u2 = ResidualBlock(128, 128)
        
        self.u1_up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.u1_concat = nn.Conv2d(64 + 64, 64, kernel_size=1, padding='same')
        self.u1 = ResidualBlock(64, 64)
        
        # 可学习的空间注意力
        self.spatial_attention = LearnableSpatialAttention()
        
        # 输出层
        self.output = nn.Conv2d(64, 1, kernel_size=1, activation='sigmoid')

    def forward(self, x):
        # 下采样
        c1 = self.c1(x)
        p1 = self.p1(c1)
        
        c2 = self.c2(p1)
        p2 = self.p2(c2)
        
        # 局部和全局流特征提取与融合
        c3_local = self.c3_local(p2)
        c3_global = self.c3_global(p2)
        c3_fused = fusion_module(c3_local, c3_global)
        c3_fused = self.c3_fused(c3_fused)
        p3 = self.p3(c3_fused)
        
        c4_local = self.c4_local(p3)
        c4_global = self.c4_global(p3)
        c4_fused = fusion_module(c4_local, c4_global)
        c4_fused = self.c4_fused(c4_fused)
        p4 = self.p4(c4_fused)
        
        c5_local = self.c5_local(p4)
        c5_global = self.c5_global(p4)
        c5_fused = fusion_module(c5_local, c5_global)
        c5_fused = self.c5_fused(c5_fused)
        
        # 上采样
        u4_up = self.u4_up(c5_fused)
        u4_concat = torch.cat([u4_up, c4_fused], dim=1)
        u4_concat = self.u4_concat(u4_concat)
        u4 = self.u4(u4_concat)
        
        u3_up = self.u3_up(u4)
        u3_concat = torch.cat([u3_up, c3_fused], dim=1)
        u3_concat = self.u3_concat(u3_concat)
        u3 = self.u3(u3_concat)
        
        u2_up = self.u2_up(u3)
        u2_concat = torch.cat([u2_up, c2], dim=1)
        u2_concat = self.u2_concat(u2_concat)
        u2 = self.u2(u2_concat)
        
        u1_up = self.u1_up(u2)
        u1_concat = torch.cat([u1_up, c1], dim=1)
        u1_concat = self.u1_concat(u1_concat)
        u1 = self.u1(u1_concat)
        
        # 空间注意力
        u1 = self.spatial_attention(u1)
        
        # 输出
        outputs = self.output(u1)
        return outputs


# 可视化函数
def visualize_predictions(model, dataset, indices=None):
    if indices is None:
        raise ValueError("Please provide a list of indices to visualize specific images.")
    
    model.eval()
    num_samples = len(indices)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    with torch.no_grad():
        for idx, img_idx in enumerate(indices):
            # 获取数据
            image, label = dataset[img_idx]
            image = image.unsqueeze(0).to(device)  # 添加批次维度并移至设备
            
            # 预测结果
            prediction = model(image)
            prediction = (prediction.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            
            # 处理显示数据
            image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            label_np = label.squeeze().numpy()
            
            # 原始图像
            axes[idx, 0].imshow(cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            axes[idx, 0].set_title("Original Image")
            axes[idx, 0].axis("off")
            
            # 标签图像
            axes[idx, 1].imshow(label_np, cmap="gray")
            axes[idx, 1].set_title("Ground Truth")
            axes[idx, 1].axis("off")
            
            # 预测结果图像
            axes[idx, 2].imshow(prediction, cmap="gray")
            axes[idx, 2].set_title("Prediction")
            axes[idx, 2].axis("off")
    
    plt.tight_layout()
    plt.show()
    model.train()


# 定义评估指标
def calculate_metrics(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(np.int32)
    y_true = y_true.astype(np.int32)
    
    # 计算IoU
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    iou = intersection / (union + 1e-7)
    
    # 计算Dice系数
    dice = (2 * np.sum(y_pred[y_true == 1]) + 1e-7) / (np.sum(y_pred) + np.sum(y_true) + 1e-7)
    
    # 计算混淆矩阵和其他指标
    confusion = confusion_matrix(y_true.flatten(), y_pred.flatten())
    if confusion.size < 4:  # 处理可能的单类情况
        confusion = np.pad(confusion, ((0, max(0, 2 - confusion.shape[0])), 
                                      (0, max(0, 2 - confusion.shape[1]))), 
                          mode='constant')
    
    accuracy = np.trace(confusion) / (np.sum(confusion) + 1e-7)
    precision = confusion[1, 1] / (confusion[1, 1] + confusion[0, 1] + 1e-7)
    recall = confusion[1, 1] / (confusion[1, 1] + confusion[1, 0] + 1e-7)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    oa = accuracy  # 总体准确率
    pe = np.sum(np.sum(confusion, axis=0) * np.sum(confusion, axis=1)) / (np.sum(confusion) **2 + 1e-7)
    kappa = (oa - pe) / (1 - pe + 1e-7)
    
    return {
        "IoU": iou,
        "Dice Coefficient": dice,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "Overall Accuracy (OA)": oa,
        "Kappa Coefficient": kappa
    }


# 主函数
def main():
    # 初始化数据
    batch_size = 8
    train_dataset = RoadDataset(os.path.join(data_dir, "Train"), shuffle=True)
    val_dataset = RoadDataset(os.path.join(data_dir, "Validation"), shuffle=False)
    test_dataset = RoadDataset(os.path.join(data_dir, "Test"), shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 构建模型
    model = DualStreamUNet().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练模型
    epochs = 50
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * images.size(0)
            predicted = (outputs > 0.5).float()
            train_total += labels.numel()
            train_correct += (predicted == labels).sum().item()
        
        # 计算训练集指标
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                predicted = (outputs > 0.5).float()
                val_total += labels.numel()
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print('-' * 50)
    
    # 可视化预测结果
    fixed_indices = [0, 1, 2, 3, 4, 5]
    visualize_predictions(model, test_dataset, indices=fixed_indices)
    
    # 测试模型并计算评估指标
    model.eval()
    y_true_list, y_pred_list = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            y_true_list.append(labels.numpy())
            y_pred_list.append(outputs.cpu().numpy())
    
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    
    metrics = calculate_metrics(y_true, y_pred)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
