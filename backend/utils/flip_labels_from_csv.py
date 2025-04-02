import random

import numpy as np
import pandas as pd

def flip_labels_from_csv(csv_file, label_column='label', flip_percentage=0.1):
    """
    从 CSV 文件读取数据，并随机翻转指定列中的标签。  少数类数量自动计算。

    Args:
        csv_file (str): CSV 文件的路径。
        label_column (str): 包含标签的列的名称。默认为 'label'。
        flip_percentage (float): 翻转的百分比，基于少数类数量计算。 默认为 0.1 (10%)。

    Returns:
        pd.DataFrame: 包含翻转后标签的 DataFrame。
    """

    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 检查标签列是否存在
    if label_column not in df.columns:
        raise ValueError(f"标签列 '{label_column}' 不存在于 CSV 文件中。")

    # 获取标签
    labels = df[label_column].values

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

    # 将翻转后的标签更新到 DataFrame
    df[label_column] = flipped_labels
    df.to_csv('../dataset/label01234-raw.csv', index=False)
    return df


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    csv_file='../dataset/label01234-clean.csv'
    label_column='label'
    flip_percentage=0.1
    flipped_df = flip_labels_from_csv(csv_file, label_column, flip_percentage)

