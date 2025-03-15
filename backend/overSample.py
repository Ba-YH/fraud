import math
import random
from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist


class mySMOTE:

    def cal_w(self, Minority_data, k):

        MinorityKNN=NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(Minority_data)

        MinorityKNN_indices=MinorityKNN.kneighbors(Minority_data)[1]

        # 去除每个点本身的索引
        MinorityKNN_indices=MinorityKNN_indices[:, 1:]

        # 初始化密度数组
        Sparse=np.zeros(Minority_data.shape[0])

        for i in range(Minority_data.shape[0]):
            pos1=MinorityKNN_indices[i]

            Sparse[i]=np.sum(cdist(Minority_data[pos1], [Minority_data[i]]))

        Sparse=Sparse/np.sum(Sparse)
        w=Sparse

        w=w/np.sum(w)
        return w

    def smote_rd_populate(self, N, Minority_data, Minority_label, base, NN_Matrix, k):
        numattrs=Minority_data.shape[1]
        Synthetic_samples=[]
        Synthetic_label=[]
        # print(Minority_data)
        while N>0:
            # 随机选择一个邻居
            random_NN=math.ceil(np.random.rand()*k)-1
            base_data=Minority_data[base, :]
            base_label=Minority_label[base]
            base_NNs=NN_Matrix[base, :]
            base_random_NN=base_NNs[random_NN]

            # 创建合成样本
            new_sample=np.zeros(numattrs)
            for i in range(numattrs):
                dif=Minority_data[base_random_NN, i]-base_data[i]
                gap=np.random.rand()
                new_sample[i]=base_data[i]+gap*dif
                # print(Minority_data[base_random_NN])
            # 添加到合成样本列表
            Synthetic_samples.append(new_sample)
            Synthetic_label.append(base_label)

            N-=1

        return np.array(Synthetic_samples), np.array(Synthetic_label)

    def smote_rd_generation(self, Minority_data, Minority_label, N, k):
        # SMOTE
        Synthetic_samples=[]
        Synthetic_label=[]

        # 计算每个样本的k个最近邻
        NN_model=NearestNeighbors(n_neighbors=k+1, algorithm='auto')
        NN_model.fit(Minority_data)
        NN_Matrix=NN_model.kneighbors(Minority_data, return_distance=False)  # 获取最近邻的索引
        NN_Matrix=NN_Matrix[:, 1:]  # 去掉每个点自己作为邻居
        for i in range(len(Minority_data)):
            if N[i]>0:
                new_samples, new_label=self.smote_rd_populate(N[i], Minority_data, Minority_label, i, NN_Matrix, k)
                Synthetic_samples.append(new_samples)
                Synthetic_label.append(new_label)

        return np.vstack(Synthetic_samples), np.hstack(Synthetic_label)

    def create_synthetic_samples(self, Minority_data, Minority_label, Majority_data, Majority_label, k, total):

        w=self.cal_w(Minority_data, k)
        # 计算合成样本的总量
        Total_generation=total
        # 计算每个样本的生成数量
        weight=np.floor(Total_generation*w).astype(int)

        remaining=Total_generation-np.sum(weight)
        indices=np.argsort(w)[::-1]  # 从大到小排序
        for i in range(remaining):
            weight[indices[i%len(w)]]+=1
        # 生成合成样本
        Synthetic_samples, Synthetic_label=self.smote_rd_generation(Minority_data, Minority_label, weight, k)

        Minority_data=np.vstack((Minority_data, Synthetic_samples))
        np.random.shuffle(Minority_data)

        X_my=np.vstack([Majority_data, Minority_data])
        y_my=np.hstack([Majority_label, Minority_label, Synthetic_label])

        return X_my, y_my

    def fit_resample_1(self, X, y, k):
        majority_class=Counter(y).most_common()[0][0]
        minority_class=Counter(y).most_common()[1][0]

        Minority_data=X[y==minority_class]
        Minority_label=y[y==minority_class]
        Majority_data=X[y==majority_class]
        Majority_label=y[y==majority_class]

        total=len(Majority_label)-len(Minority_label)

        X_my, y_my=self.create_synthetic_samples(Minority_data, Minority_label, Majority_data,
                                                 Majority_label, k, total)

        return X_my[y_my==minority_class], y_my[y_my==minority_class]

    def fit_resample(self, X, y, k=5):
        label_counts=Counter(y)
        majority_class=label_counts.most_common(1)[0][0]
        minority_classes=[label for label in np.unique(y) if label!=majority_class]

        Majority_data=X[y==majority_class]
        Majority_label=y[y==majority_class]

        X_my=Majority_data
        y_my=Majority_label

        for minority_class in minority_classes:
            Minority_data=X[y==minority_class]
            Minority_label=y[y==minority_class]
            syn_data,syn_label=self.fit_resample_1(np.vstack((Majority_data,Minority_data)),np.hstack((Majority_label,Minority_label)),k)
            X_my=np.vstack((X_my,syn_data))
            y_my=np.hstack((y_my,syn_label))

        return X_my,y_my


if __name__=='__main__':
    np.random.seed(42)
    random.seed(42)
    X=np.array(
        [[1, 1], [2, 1], [3, 1], [2, 2], [1, 3], [2, 3], [3, 3], [1, 4], [1, 5], [2, 5], [5, 1], [5, 2], [3.5, 3],
         [5.5, 3], [6, 3], [7, 3], [5, 4], [5, 5], [7, 5], [4, 4], [4, 5], [4, 6]])
    y=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2])
    overSampler=mySMOTE()
    X_my,y_my=overSampler.fit_resample(X, y,k=2)
    print(y_my)
