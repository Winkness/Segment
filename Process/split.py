import os
import shutil
import random
from pathlib import Path


def split_dataset(images_dir, masks_dir, train_count, val_count, test_count, output_dir):
    """
    按照指定数量将图像和掩码划分为训练集、验证集和测试集

    参数:
    images_dir (str): 包含RGB图像的文件夹路径
    masks_dir (str): 包含掩码图像的文件夹路径
    train_count (int): 训练集图像数量
    val_count (int): 验证集图像数量
    test_count (int): 测试集图像数量
    output_dir (str): 输出数据集的根目录
    """
    # 创建输出目录结构
    for subset in ['Train', 'Val', 'Test']:
        for subdir in ['images', 'masks']:
            os.makedirs(os.path.join(output_dir, subset, subdir), exist_ok=True)

    # 获取所有图像文件并提取ID
    image_files = {}
    for f in os.listdir(images_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 提取ID，例如从"1_sat.jpg"中提取"1"
            parts = f.split('_')
            if len(parts) >= 2:
                img_id = parts[0]
                image_files[img_id] = f

    # 匹配对应的掩码文件
    matched_pairs = []
    for img_id, img_file in image_files.items():
        # 构建对应的掩码文件名
        mask_file = f"{img_id}_mask.png"
        if mask_file in os.listdir(masks_dir):
            print(mask_file)
            matched_pairs.append((img_file, mask_file))
        else:
            print(f"警告: 未找到与图像 {img_file} 对应的掩码文件 {mask_file}")

    # 打乱匹配对列表以确保随机划分
    random.shuffle(matched_pairs)

    # 检查文件数量是否足够
    total_required = train_count + val_count + test_count
    if len(matched_pairs) < total_required:
        raise ValueError(f"匹配的图像-掩码对数量不足，需要{total_required}对，但只有{len(matched_pairs)}对")

    # 划分数据集
    train_pairs = matched_pairs[:train_count]
    val_pairs = matched_pairs[train_count:train_count + val_count]
    test_pairs = matched_pairs[train_count + val_count:train_count + val_count + test_count]

    subsets = {
        'Train': train_pairs,
        'Val': val_pairs,
        'Test': test_pairs
    }

    # 处理每个子集
    for subset_name, pairs in subsets.items():
        for img_file, mask_file in pairs:
            # 复制图像和掩码到目标目录
            try:
                shutil.copy2(os.path.join(images_dir, img_file),
                             os.path.join(output_dir, subset_name, 'images', img_file))
                shutil.copy2(os.path.join(masks_dir, mask_file),
                             os.path.join(output_dir, subset_name, 'masks', mask_file))
                print(f"已复制: {img_file} 和 {mask_file} 到 {subset_name}")
            except Exception as e:
                print(f"错误: 复制文件 {img_file} 或 {mask_file} 时出错: {e}")


if __name__ == "__main__":
    # 配置参数
    IMAGES_DIR = "./deepglobe/images"  # 替换为实际图像文件夹路径
    MASKS_DIR = "./deepglobe/masks"  # 替换为实际掩码文件夹路径
    OUTPUT_DIR = "./deepglobe"  # 替换为实际输出文件夹路径

    # 数据集划分数量
    TRAIN_COUNT = 4980
    VAL_COUNT = 622
    TEST_COUNT = 624

    # 执行数据集划分
    split_dataset(IMAGES_DIR, MASKS_DIR, TRAIN_COUNT, VAL_COUNT, TEST_COUNT, OUTPUT_DIR)    