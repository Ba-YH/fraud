import numpy as np
import pandas as pd

def flip_labels(labels, flip_percentage=0.1):
    """
    随机翻转数据集中每个类别的标签。  少数类数量自动计算。

    Args:
        data (np.ndarray or pd.DataFrame): 输入数据。
        labels (np.ndarray or pd.Series): 数据的标签。
        flip_percentage (float): 翻转的百分比，基于少数类数量计算。 默认为 0.1 (10%)。

    Returns:
        np.ndarray: 翻转后的标签。
    """

    unique_labels = np.unique(labels)
    flipped_labels = labels.copy()  # 创建标签的副本，避免修改原始数据

    # 计算每个类别的样本数量
    class_counts = {}
    for label in unique_labels:
        class_counts[label] = np.sum(labels == label)

    # 找到最小的类别数量（即少数类的数量）
    minority_class_size = min(class_counts.values())

    for label in unique_labels:
        # 找到当前类别的所有样本的索引
        indices = np.where(labels == label)[0]

        # 计算需要翻转的样本数量
        num_to_flip = int(minority_class_size * flip_percentage)
        num_to_flip = min(num_to_flip, len(indices)) # 确保翻转数量不超过当前类别的样本数量

        # 随机选择需要翻转的样本索引
        indices_to_flip = np.random.choice(indices, size=num_to_flip, replace=False)

        # 翻转选定样本的标签
        for index in indices_to_flip:
            # 随机选择一个不同的标签
            other_labels = unique_labels[unique_labels != label]
            new_label = np.random.choice(other_labels)
            flipped_labels[index] = new_label

    return flipped_labels


if __name__ == '__main__':
    # 示例用法
    # 1. 创建一个模拟的五分类数据集
    num_samples = 60
    num_minority = 10  # 每个少数类的样本数量
    num_majority = num_samples - 4 * num_minority # 多数类的样本数量

    # 创建标签
    labels = np.concatenate([
        np.zeros(num_majority),  # 多数类 (标签 0)
        np.ones(num_minority),   # 少数类 (标签 1)
        np.full(num_minority, 2), # 少数类 (标签 2)
        np.full(num_minority, 3), # 少数类 (标签 3)
        np.full(num_minority, 4)  # 少数类 (标签 4)
    ])

    # 创建一些随机数据 (这里只是为了演示，实际应用中替换成你的数据)
    data = np.random.rand(num_samples, 10)

    # 2. 设置翻转百分比
    flip_percentage = 0.1  # 翻转 10% 的标签

    # 3. 调用标签翻转函数
    flipped_labels = flip_labels(labels, flip_percentage)

    print(flipped_labels)
