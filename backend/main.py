import os
import csv
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.metrics import geometric_mean_score
import random
import sklearn
from collections import Counter
from imbalancedmetrics import ImBinaryMetric
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTEN
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import SMOTE
from overSample import FD_SMOTE
import joblib  # 或者使用 pickle
import dill
from flip_labels import flip_labels
np.random.seed(42)
random.seed(42)

warnings.filterwarnings("ignore")
ar = 1


class Fraud_detection(BaseEstimator, ClassifierMixin):
    base_estimator = DecisionTreeClassifier()
    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 n_estimators=10,
                 random_state=42,
                 ar= ar):
        self.base_estimator = base_estimator
        self.estimators_ = []
        self.weight_ = []
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.ar = 1
        self.min_num = []
        self.max_num = []
    @classmethod
    def fit_base_estimator(self, X, y):
        """Private function used to train a single base estimator."""
        return sklearn.base.clone(self.base_estimator).fit(X, y)

    def random_under_sampling(self, X_maj, y_maj, X_min, y_min):
        """Private function used to perform random under-sampling."""
        maj_idx = self.random_sampling(len(X_maj), len(X_min))
        X_train = np.concatenate([X_maj[maj_idx], X_min])
        y_train = np.concatenate([y_maj[maj_idx], y_min])
        return X_train, y_train
    def random_sampling(self, X_len, n):
        np.random.seed(self.random_state)
        idx = np.random.choice(X_len, n, replace=True)
        return idx
    def get_samples(self, prob, sampling_cnt,tag):
        sampled_bins = []
        step = (prob.max() - prob.min()) / sampling_cnt
        interval = np.arange(prob.min(), prob.max(), step)  # interval of bins
        part_bin = np.digitize(prob, interval)  # partition probability of the majority class
        part_cnt = Counter(part_bin)  # count the number of samples in each bin (e.g., {bin_id:cnt,...})
        s = np.zeros(sampling_cnt + 2)
        noempty_bin_key = [i for i in part_cnt.keys()]
        noempty_bin_key.sort()
        # print(f"noempty_bin_key:{noempty_bin_key}")
        # print(f"part_cnt:{part_cnt}")
        # print(f"tag:{tag}")
        s[noempty_bin_key] = 1
        if noempty_bin_key[0] == 1:
            s[noempty_bin_key[0]] = 1
        else:
            s[noempty_bin_key[0]] = noempty_bin_key[0]
            s[:noempty_bin_key[0]] = 0
        if tag == 1:
            # mid = int(np.median(noempty_bin_key))
            mid = np.digitize([0], interval)[0]
            # print(f"mid1:{mid}")
            # print(f"mid2:{np.digitize([0], interval)[0]}")
            for i in range(1,mid):
                if i not in noempty_bin_key:
                    s[i] = 0
                else:
                    min_index = 0
                    for j in range(1,i):
                        if j in noempty_bin_key:
                            min_index = j
                    s[i] = i-min_index
            for i in range(mid,len(s)):
                if i not in noempty_bin_key:
                    s[i] = 0
                else:
                    min_index = len(s) - 1
                    for j in range(len(s),i,-1):
                        if j in noempty_bin_key:
                            min_index = j
                    s[i] = min_index - i
        else:
            diff = np.diff(noempty_bin_key)
            s[noempty_bin_key[1:]] += diff - 1
        s = np.floor(s).astype(int)  ## 向下取整
        # s = np.ceil(s).astype(int) ## 向上取整
        # print(f"增加桶所需要采样的数量{s}")

        start_s_index = 1  # the index of the first sampling count
        cur_needs = 0
        pre_key = noempty_bin_key[0]  # the previous index of the noempty bins
        k = 0
        for key in noempty_bin_key:
            temp_elements = []
            ele_cnt = part_cnt[key]
            cur_needs = s[start_s_index:key].sum()
            start_s_index = key + 1  # record the next bin which will be sampled.
            # The bins from pre_index to key-1 need to sample, but there aren't enough samples in them.
            if cur_needs > 0:
                if pre_key == 0:  # pre_key is the last bin,need  to start from the first element in part_bin
                    temp_elements = np.where(part_bin == (len(noempty_bin_key) - 1))[-1].tolist()
                else:
                    temp_elements = np.where(part_bin == pre_key)[-1].tolist()
                    # sampled_bins.extend(np.random.choice(temp_elements, cur_needs, replace=False))

            if s[key] <= ele_cnt:  # the number of the samples in currrent bin are greater than that of needing samples.
                sampled_bins.extend(np.random.choice(
                    np.where(part_bin == key)[-1].tolist(), s[key], replace=False))
            ##np.random.choice()随机不重复采样
            else:
                sampled_bins.extend(np.random.choice(
                    np.where(part_bin == key)[-1].tolist(), s[key], replace=True))
            pre_key = key
        # remaining bins
        if start_s_index < sampling_cnt:
            cur_needs = s[start_s_index:sampling_cnt].sum()
            # print(f"cur_needs:{cur_needs}")
            temp_elements = np.where(part_bin == pre_key)[-1].tolist()
            sampled_bins.extend(np.random.choice(
                temp_elements, cur_needs, replace=False))
        ##### 取的样本目前小于少数类样本数，取出不够的那部分
        # s_bin_length = sum([len(sampled_bins) for sublist in sampled_bins])
        s_bin_length = len(sampled_bins)
        # print(f"sample = {s_bin_length}")
        # print(f"sample = {sampled_bins}")
        max_legth = sampling_cnt
        # print(f"max_legth:{max_legth}")
        if s_bin_length < max_legth:
            index_x = list(i for i in range(len(prob)) if i not in sampled_bins)
            sampled_bins.extend(np.random.choice(index_x, max_legth-s_bin_length, replace=False))
        # print(part_cnt)
        return sampled_bins

    def equalization_sampling(self, X_maj, y_maj,tag):
        if tag == 1:
            prob_maj = self.y_pred_maj_min
        else:
            prob_maj = self.y_pred_maj
        # print(f"tag:{tag}")
        # If the probabilitys are not distinguishable, perform random smapling
        if prob_maj.max() == prob_maj.min():
            index = list(self.random_sampling(len(X_maj), self.k_bins))
            new_X_maj = X_maj[index]
        else:
            index = self.get_samples(prob_maj, self.k_bins,tag = tag)
            # index = np.concatenate(maj_sampled_bins, axis=0)
            new_X_maj = X_maj[index]
        new_y_maj = np.full(new_X_maj.shape[0], y_maj[0])
        # print(f"new多数类样本数量：{len(new_X_maj)}")
        self.max_num.append(len(new_y_maj))
        # print(f"index:{index}")
        return new_X_maj, new_y_maj,index

    def overlap_enhanced(self,X_maj):
        from sklearn.metrics import pairwise_distances
        # 找到接近0.5的样本
        threshold = 0.05
        close_to_half_indices = np.where((self.y_pred_maj >= 0.5 - threshold) & (self.y_pred_maj <= 0.5 + threshold))[0]

        # 计算距离矩阵
        distances = pairwise_distances(X_maj)

        # 找到每个接近0.5的样本的最近5个样本，并去掉重复样本
        nearest_samples = {}
        unique_samples = set()  # 用于存储唯一的样本索引

        for index in close_to_half_indices:
            # 获取当前样本的距离
            current_distances = distances[index]
            # 获取距离最近的5个样本的索引（排除自己）
            nearest_indices = np.argsort(current_distances)[1:6]  # 排序并取前5个
            # 将最近样本添加到集合中，确保唯一性
            for nearest_index in nearest_indices:
                unique_samples.add(nearest_index)

        # 将结果转换为列表
        unique_samples_list = list(unique_samples)
        new_maj_y2 = np.full(len(unique_samples_list), 1)
        return unique_samples_list,new_maj_y2


    def fit(self, X, y, label_maj=0, label_min=1):
        self.estimators_ = []
        self.weight_ = []
        self.hard = []
        self.ar = 1
        self.precision = []
        self.recall = []
        # Initialize by spliting majority / minority set
        X_maj = X[y == label_maj];
        y_maj = y[y == label_maj]
        X_min = X[y != label_maj];
        y_min = y[y != label_maj]
        # print(f"多数类样本数量：{len(X_maj)}")
        # print(f"少数类样本数量：{len(X_min)}")
        self.k_bins = X_min.shape[0]
        self.y_pred_maj = np.zeros(X_maj.shape[0])
        self.y_pred_min = np.zeros(X_min.shape[0])
        self.y_pred_maj_min = np.zeros(X_maj.shape[0])
        tag = 0
        # overlap_maj_index, _,flag = self.cal_index(X_maj, X_min)
        for i_estimator in range(0, self.n_estimators):
            # print(i_estimator)
            if i_estimator == 0:
                X_train, y_train = self.random_under_sampling(
                    X_maj, y_maj, X_min, y_min)
                clf = self.fit_base_estimator(X_train, y_train)
                self.y_pred_maj = clf.predict_proba(X_maj)[:, 0]
                self.y_pred_maj_min = clf.predict_proba(X_maj)[:, 1]
                self.weight_.append(1)
                self.estimators_.append(clf)
                continue
            else:
                self.min_num.append(len(X_min))
                new_maj_X1, new_maj_y1 ,index1= self.equalization_sampling(X_maj,
                                                                           y_maj, tag=0)
                index2,new_maj_y2 = self.overlap_enhanced(X_maj)
                index1.extend(index2)
                index = index1
                new_maj_x_ = X_maj[index]
                new_maj_y_ = np.full(new_maj_x_.shape[0], y_maj[0])
                # X_, y_ =np.vstack([new_maj_x_, X_min]),np.hstack([new_maj_y_, y_min])
                smote= FD_SMOTE()
                X_,y_ = smote.fit_resample(np.vstack([new_maj_x_, X_min]), np.hstack([new_maj_y_, y_min]))
                clf = self.fit_base_estimator(X_, y_)
                y_pred = clf.predict(X)
                im_metric = ImBinaryMetric(y, y_pred)
                # eps = im_metric.f1()
                self.weight_.append(1)
                W = np.array(self.weight_)
                self.estimators_.append(clf)
                temp_y_pred_maj = clf.predict_proba(X_maj)[:, 0]
                proba = clf.predict_proba(X_maj)
                top_two_probs = np.sort(proba, axis=1)[:, -2:]  # 取每行的前两个最大值
                # max_other_probs = np.max(proba[:, 1:], axis=1)
                # 计算前两个置信度的差值
                # temp_y_pred_min = temp_y_pred_maj- max_other_probs
                temp_y_pred_min = top_two_probs[:, 1] - top_two_probs[:, 0]
                self.y_pred_maj = self.y_pred_maj * (W[0:-1].sum() / W.sum()) + temp_y_pred_maj * (W[-1] / W.sum())
                self.y_pred_maj_min = self.y_pred_maj_min * (W[0:-1].sum() / W.sum()) + temp_y_pred_min * (
                        W[-1] / W.sum())
        # print(f"w:{self.weight_}")
        # print(f"persicion:{self.precision}")
        # print(f"recall:{self.recall}")
        return self

    def predict_proba(self, X):
        w = np.array(self.weight_)
        w = w / w.sum()
        # print(f"w_len:{w},#estimators:{len(self.estimators_)}")
        y_pred = np.array(
            [model.predict_proba(X) * w[i] for i, model in enumerate(self.estimators_)]
        ).sum(axis=0)
        return y_pred

    # def predict(self, X):
    #     y_pred = self.predict_proba(X)
    #     return y_pred.argmax(axis=1)
    def predict(self, X):
        y_pred_proba = self.predict_proba(X)  # 获取每个类别的预测概率
        y_pred = y_pred_proba.argmax(axis=1)   # 返回概率最大的类别索引
        return y_pred, y_pred_proba             # 返回预测类别和概率


    def score(self, X, y):
        return average_precision_score(
            y, self.predict_proba(X)[:, 1])

def parse():
    '''Parse system arguments.'''
    parser = argparse.ArgumentParser(
        description='General executing Ensemble method',
        usage='genealrunEnsemble.py -dir <datasetpath> -alg <algorithm name> -est <number of estimators>'
              ' -n <n-fold> -ar <additional parameter>'
    )
    parser.add_argument("-dir", dest="dataset", help="path of the datasets or a dataset")
    parser.add_argument("-alg", dest="algorithm", nargs='+', help="list of the algorithm names ")
    parser.add_argument("-est", dest="estimators", default=10, type=int, help="number of estimators")
    parser.add_argument("-n", dest="nfold", default=5, type=int, help="n fold")
    parser.add_argument("-ar", dest="additional_param", help="additional parameter")
    return parser.parse_args()

def main():
    algs = ['EASE']
    # ar_list = [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4]
    ar_list = [10]
    for ff in ar_list:
        print(f"{ff}:")
        for baseline in ['AHSE']:
            # print(f"baseline: {baseline}\n")
            df = pd.DataFrame(columns=['file_name', 'f1', 'MCC','AUC'])
            folder_path = 'process_data'  # 文件夹路径

            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):  # 筛选出以 .data 结尾的文件
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        print(file_name)
                        tsvreader = csv.reader(f, delimiter='\t')
                        header = next(tsvreader)
                        temp0 = list(tsvreader)
                        # 寻找第一个不以@开头的行的索引
                        start_index = 0
                        for i, row in enumerate(temp0):
                            if not row[0].startswith('@'):
                                start_index = i
                                break
                        # 保留从第一个不以@开头的行开始的所有数据
                        temp = temp0[start_index:]
                    majority_label1 = '0'
                    # minority_label1 = ' 冒充公检法及政府机关类'
                    majority_label2 = '0'
                    minority_label1 = '冒充公检法及政府机关类'
                    minority_label2 = '贷款、代办信用卡类'
                    minority_label3 = '冒充客服服务'
                    minority_label4 = '冒充领导、熟人类'
                    a = []
                    for i in temp:
                        a.append(i[0].split(","))
                    data = []
                    for k in a:
                        if 'abalone' in file_name:
                            k[0] = ord(k[0])
                        data.append(k[0:-1])
                    data = np.array(data)
                    target = []
                    for k in a:
                        # target.append(k[-1])
                        if (k[-1] == minority_label1):
                            target.append(1)
                        elif(k[-1] == minority_label2):
                            target.append(2)
                        elif(k[-1] == minority_label3):
                            target.append(3)
                        elif(k[-1] == minority_label4):
                            target.append(4)
                        else:
                            target.append(0)
                    X = np.array(data)
                    from sklearn.utils import check_array
                    X = check_array(X, dtype=np.float64)
                    y = np.array(target)
                    auc_all, f1_all, MCC, gmeans_all = [], [], [], []
                    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
                    fold_cnt = 0
                    best_auc = 0  # 初始化最优 AUC
                    best_model = None  # 用于保存最优模型
                    for p in range(10):
                        for train_index, test_index in sss.split(X, y):
                            X_train, X_test = X[train_index], X[test_index]
                            y_train, y_test = y[train_index], y[test_index]
                            # y_train = flip_labels(y_train, flip_rate=0.1)
                            model = Fraud_detection(base_estimator=DecisionTreeClassifier(),
                                         n_estimators=ff,
                                         ar=1)
                            # model_filename = 'best_model.dill'
                            # with open(model_filename, 'rb') as file:
                            #     best_model = dill.load(file)
                            model.fit(X_train,y_train)
                            y_pred, y_pred_proba = model.predict(X_test)
                            # print(y_pred.shape)
                            # print(y_test.shape)
                            # y_pred = model.predict_proba(X_test)[:, 1]
                            # y_pred[np.isnan(y_pred)] = 0
                            # metric = ImBinaryMetric(y_test, y_pre)
                            mcc_score = matthews_corrcoef(y_test, y_pred)
                            auc = roc_auc_score(y_test, y_pred_proba,multi_class='ovr')
                            g_means = geometric_mean_score(y_test, y_pred,average='macro')
                            f1 = f1_score(y_test, y_pred,average='macro')
                            auc_all.append(auc)
                            f1_all.append(f1)
                            MCC.append(mcc_score)
                            gmeans_all.append(g_means)
                            if g_means > best_auc:
                                best_auc = g_means
                                best_model = model  # 保存当前最优模型
                            del model
                            model_filename = f'models/best_model.dill'  # 可以根据需要修改文件名
                            with open(model_filename, 'wb') as file:
                                dill.dump(best_model, file)

                    metrics = {'file_name': file_name,
                               'f1': f'{np.mean(f1_all):.3f} ± {np.std(f1_all):.3f}',
                               'MCC': f'{np.mean(MCC):.3f} ± {np.std(MCC):.3f}',
                               'AUC': f'{np.mean(auc_all):.3f} ± {np.std(auc_all):.3f}',
                               'Gean': f'{np.mean(gmeans_all):.3f} ± {np.std(gmeans_all):.3f}'}
                    new_row_df = pd.DataFrame([metrics])
                    df = pd.concat([df, new_row_df], ignore_index=True)

            df.to_excel(f'test.xlsx', index=False)

    return

if __name__ == '__main__':
    main()
