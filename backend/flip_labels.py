import numpy as np

def flip_labels(y_train, flip_rate=0.1):
    """
    标签翻转
    :param y_train: 训练集标签
    :param flip_rate: 噪声率
    :return: y_train_flipped:加噪后的标签
    """
    y_train=np.array(y_train)
    unique_labels=np.unique(y_train)

    # 确定少数类和多数类
    counts=np.bincount(y_train)
    minority_class=unique_labels[np.argmin(counts)]
    majority_class=unique_labels[np.argmax(counts)]

    # 计算需要翻转的数量
    minority_count=np.sum(y_train==minority_class)
    majority_count=np.sum(y_train==majority_class)

    num_minority_flips=int(minority_count*flip_rate)
    num_majority_flips=int(majority_count*flip_rate)

    # 找到少数类和多数类的索引
    minority_indices=np.where(y_train==minority_class)[0]
    majority_indices=np.where(y_train==majority_class)[0]

    # 随机选择需要翻转的索引
    minority_flip_indices=np.random.choice(minority_indices, num_minority_flips, replace=False)
    majority_flip_indices=np.random.choice(majority_indices, num_majority_flips, replace=False)

    # 复制标签，避免修改原始数据
    y_train_flipped=y_train.copy()

    # 翻转标签
    for i in minority_flip_indices:
        y_train_flipped[i]=majority_class
    for i in majority_flip_indices:
        y_train_flipped[i]=minority_class

    return y_train_flipped

if __name__=='__main__':
    ytrain=np.array([1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
    print(flip_labels(ytrain,0.1))

