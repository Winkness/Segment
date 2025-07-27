import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanIoU
from sklearn.metrics import confusion_matrix
# 数据集路径
data_dir = "Data2/压缩/Data3/archive"

# 自定义数据集加载
class RoadDataset(tf.keras.utils.Sequence):
    def __init__(self, folder, batch_size, img_size=(256, 256), shuffle=True):
        self.image_paths = sorted(os.listdir(os.path.join(folder, "image")))
        self.label_paths = sorted(os.listdir(os.path.join(folder, "label")))
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.folder = folder
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, labels = [], []
        for i in batch_indices:
            # Load image
            image_path = os.path.join(self.folder, "image", self.image_paths[i])
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Load label
            label_path = os.path.join(self.folder, "label", self.label_paths[i])
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if label is None:
                raise ValueError(f"Failed to load label: {label_path}")

            # Process the image and label
            label = label / 255.0
            image = cv2.resize(image, self.img_size)
            label = cv2.resize(label, self.img_size)

            # Append to the batch
            images.append(image)
            labels.append(label)

        return np.array(images) / 255.0, np.expand_dims(np.array(labels), axis=-1)


    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

# 定义SegNet模型
def segnet_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # 编码器部分
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 128x128

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 64x64

    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 32x32

    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)  # 16x16

    # 解码器部分
    x = layers.Conv2DTranspose(512, (3, 3), padding="same", activation="relu")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.UpSampling2D((2, 2))(x)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# 初始化数据
batch_size = 8
train_dataset = RoadDataset(os.path.join(data_dir, "Train"), batch_size=batch_size)
val_dataset = RoadDataset(os.path.join(data_dir, "Validation"), batch_size=batch_size)
test_dataset = RoadDataset(os.path.join(data_dir, "Test"), batch_size=batch_size)

# 构建和编译模型
model = segnet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_dataset, validation_data=val_dataset, epochs=50)

# 进行可视化

# 进行可视化
def visualize_predictions(model, dataset, indices=None):
    if indices is None:
        raise ValueError("Please provide a list of indices to visualize specific images.")
    
    num_samples = len(indices)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    for idx, img_idx in enumerate(indices):
        # 手动加载指定索引的图片和标签
        image_path = os.path.join(dataset.folder, "image", dataset.image_paths[img_idx])
        label_path = os.path.join(dataset.folder, "label", dataset.label_paths[img_idx])
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dataset.img_size) / 255.0
        label = cv2.resize(label, dataset.img_size) / 255.0
        
        # 预测结果
        predictions = model.predict(np.expand_dims(image, axis=0))
        prediction = (predictions[0, :, :, 0] > 0.5).astype(np.uint8)

        # 原始图像
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title("Original Image")
        axes[idx, 0].axis("off")

        # 标签图像
        axes[idx, 1].imshow(label, cmap="gray")
        axes[idx, 1].set_title("Ground Truth")
        axes[idx, 1].axis("off")

        # 预测结果图像
        axes[idx, 2].imshow(prediction, cmap="gray")
        axes[idx, 2].set_title("Prediction")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    plt.show()

# 指定要可视化的6组图片索引
fixed_indices = [1, 2, 3, 4, 5,6]
visualize_predictions(model, test_dataset, indices=fixed_indices)





# 定义评估指标
def calculate_metrics(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(np.int32)
    y_true = y_true.astype(np.int32)

    iou_metric = MeanIoU(num_classes=2)
    iou_metric.update_state(y_true, y_pred)
    iou = iou_metric.result().numpy()

    dice = np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

    confusion = confusion_matrix(y_true.flatten(), y_pred.flatten())
    accuracy = np.trace(confusion) / np.sum(confusion)
    precision = confusion[1, 1] / (confusion[1, 1] + confusion[0, 1])
    recall = confusion[1, 1] / (confusion[1, 1] + confusion[1, 0])
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    oa = np.trace(confusion) / np.sum(confusion)
    pe = np.sum(np.sum(confusion, axis=0) * np.sum(confusion, axis=1)) / (np.sum(confusion) ** 2)
    kappa = (oa - pe) / (1 - pe)

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

# 测试模型并计算评估指标
y_true_list, y_pred_list = [], []
for images, labels in test_dataset:
    predictions = model.predict(images)
    y_true_list.append(labels)
    y_pred_list.append(predictions)

y_true = np.concatenate(y_true_list, axis=0)
y_pred = np.concatenate(y_pred_list, axis=0)

metrics = calculate_metrics(y_true, y_pred)
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")
    