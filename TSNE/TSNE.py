import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# ---------------------- 配置项（用户可修改）--------------------s--
# 两个图像数据集的文件夹路径（请替换为你的实际路径）
DATASET_PATH_1 = "/mnt/data/WH-Code/Segment/MassaRoad/Train/images"  # 对应蓝色散点
DATASET_PATH_2 = "/mnt/data/WH-Code/Segment/MassaRoad/Test/images"  # 对应红色散点
# DATASET_PATH_1 = "/mnt/data/WH-Code/Segment/deepglobe/Train/images"  # 对应蓝色散点
# DATASET_PATH_2 = "/mnt/data/WH-Code/Segment/deepglobe/Test/images"  # 对应蓝色散点
# DATASET_PATH_2 = "/mnt/data/WH-Code/Segment/Global-scale/SpaceNet_test_567/img"  # 对应红色散点
# DATASET_PATH_2 = "/mnt/data/WH-Code/Segment/Global-scale/RoadTracer_test_1920/img"  # 对应红色散点

# 图像预处理配置（适配预训练模型）
IMAGE_SIZE = 224
# ---------------------- 配置项结束 ----------------------

def init_feature_extractor():
    """
    初始化特征提取器：使用预训练ResNet50，移除顶层分类器，提取图像的2048维特征
    （无需训练，仅用于特征提取，轻量高效）
    """
    # 加载预训练ResNet50模型
    resnet50 = models.resnet50(pretrained=True)
    # 移除最后一层全连接层（保留特征提取部分）
    feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
    # 设置为评估模式（关闭Dropout、BatchNorm等训练相关层）
    feature_extractor.eval()
    
    # 自动使用GPU（如果有），否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.to(device)
    
    return feature_extractor, device

def get_image_transform():
    """定义图像预处理流程（适配ResNet50的输入要求）"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 缩放图像
        transforms.ToTensor(),  # 转换为张量（0-1范围）
        transforms.Normalize(  # 标准化（ResNet预训练的均值和标准差）
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform

def load_image_features(folder_path, feature_extractor, device, transform):
    """
    加载单个文件夹下的所有图像，提取特征
    :param folder_path: 图像文件夹路径
    :return: 二维numpy数组（样本数 × 特征维度）
    """
    features = []
    # 支持的图像格式（可扩展）
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp")
    
    # 遍历文件夹下的所有图像文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_formats):
            image_path = os.path.join(folder_path, filename)
            try:
                # 加载图像并预处理
                image = Image.open(image_path).convert("RGB")  # 转为RGB（避免灰度图报错）
                image_tensor = transform(image).unsqueeze(0)  # 添加batch维度（1, 3, 224, 224）
                image_tensor = image_tensor.to(device)
                
                # 无梯度提取特征（提高效率，避免显存占用）
                with torch.no_grad():
                    feature = feature_extractor(image_tensor)
                    # 转换为numpy数组并展平（2048维向量）
                    feature_np = feature.cpu().numpy().flatten()
                    features.append(feature_np)
            except Exception as e:
                print(f"跳过损坏图像 {filename}：{e}")
    
    if not features:
        raise ValueError(f"文件夹 {folder_path} 中未找到有效图像")
    
    return np.array(features)

def plot_tsne(dataset1_features, dataset2_features):
    """
    合并两个数据集特征，执行t-SNE降维，并绘制红蓝散点图
    :param dataset1_features: 数据集1的特征数组（蓝色）
    :param dataset2_features: 数据集2的特征数组（红色）
    """
    # 1. 合并特征和标签（0表示数据集1，1表示数据集2）
    all_features = np.vstack((dataset1_features, dataset2_features))
    all_labels = np.hstack((
        np.zeros(len(dataset1_features)),  # 数据集1标签：0
        np.ones(len(dataset2_features))    # 数据集2标签：1
    ))
    
    # 2. 先使用PCA降维到50维（加速t-SNE，避免高维数据计算量过大）
    # t-SNE直接处理高维数据效率低，PCA预降维是行业最佳实践
    pca = PCA(n_components=50, random_state=42)
    all_features_pca = pca.fit_transform(all_features)
    
    # 3. 执行t-SNE降维（降到2维，方便可视化）—— 修复n_iter→n_iter_max
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    all_features_tsne = tsne.fit_transform(all_features_pca)
    
    # 4. 绘制散点图
    plt.figure(figsize=(12, 8))
    
    # 数据集1：蓝色散点
    idx_1 = all_labels == 0
    plt.scatter(
        all_features_tsne[idx_1, 0], all_features_tsne[idx_1, 1],
        c='blue', label='MassRoad Train', alpha=0.6, s=50  # alpha：透明度，s：点大小
    )
    
    # 数据集2：红色散点
    idx_2 = all_labels == 1
    plt.scatter(
        all_features_tsne[idx_2, 0], all_features_tsne[idx_2, 1],
        c='red', label='MassRoad Test', alpha=0.6, s=50
    )
    
    # 5. 图表美化
    plt.title('t-SNE Distribution of MassRoad Train and Test Datasets', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('./result4.jpg', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主流程：加载数据→提取特征→t-SNE可视化"""
    try:
        # 1. 初始化工具
        feature_extractor, device = init_feature_extractor()
        transform = get_image_transform()
        print(f"使用设备：{device}")
        print("开始提取图像特征...")
        
        # 2. 提取两个数据集的特征
        dataset1_features = load_image_features(
            DATASET_PATH_1, feature_extractor, device, transform
        )
        dataset2_features = load_image_features(
            DATASET_PATH_2, feature_extractor, device, transform
        )
        
        print(f"数据集1加载完成：{len(dataset1_features)} 张图像")
        print(f"数据集2加载完成：{len(dataset2_features)} 张图像")
        print("开始执行t-SNE降维（可能需要几分钟，请耐心等待）...")
        
        # 3. 绘制t-SNE分布图
        plot_tsne(dataset1_features, dataset2_features)
        
    except Exception as e:
        print(f"程序运行出错：{e}")

if __name__ == "__main__":
    main()