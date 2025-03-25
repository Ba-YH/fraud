import sys
import os
import csv

from imbens.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier

import numpy as np
import warnings

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.metrics import geometric_mean_score
import random
from imbens.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imbens.ensemble import SMOTEBaggingClassifier
from imbens.ensemble import SMOTEBoostClassifier
from imbens.ensemble import RUSBoostClassifier
from main import AHSE
import pandas as pd
from flip_labels import flip_labels

np.random.seed(42)
random.seed(42)


def main():
    algs = ['EASE']
    # ar_list = [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4]
    ar_list = [10]
    for ff in ar_list:
        df = pd.DataFrame(columns=['baseline', 'f1', 'MCC','AUC'])
        folder_path = 'process_data'  # 文件夹路径
        for file_name in os.listdir(folder_path):
            for baseline in ['SMOTEBagging', 'SMOTEBoost', 'RUSBoost', 'BalancedBagging', 'BalancedRandomForest', 'Adaboost']:
                print(f"baseline: {baseline}\n")
                if file_name.endswith('.csv'):  # 筛选出以 .data 结尾的文件
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # print(file_name)
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
                    for p in range(10):
                        for train_index, test_index in sss.split(X, y):
                            X_train, X_test = X[train_index], X[test_index]
                            y_train, y_test = y[train_index], y[test_index]
                            # y_train = flip_labels(y_train, flip_rate=0.1)
                            if baseline == 'SMOTEBagging':
                                model = SMOTEBaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=ff)
                            elif baseline == 'SMOTEBoost':
                                model = SMOTEBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=ff)
                            elif baseline == 'RUSBoost':
                                model = RUSBoostClassifier(DecisionTreeClassifier(), n_estimators=ff,
                                                           algorithm='SAMME.R')
                            elif baseline == 'BalancedBagging':
                                model = BalancedBaggingClassifier(DecisionTreeClassifier(), n_estimators=ff,
                                                                  sampling_strategy='auto',
                                                                  replacement=False)
                            elif baseline == 'BalancedRandomForest':
                                model = BalancedRandomForestClassifier(n_estimators=ff)
                            elif baseline == 'Adaboost':
                                model = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=ff, learning_rate=1.0)
                            model.fit(X_train,y_train)
                            y_pred= model.predict(X_test)
                            y_pred_proba = model.predict_proba(X_test)
                            # print(y_pred_proba)
                            y_pred[np.isnan(y_pred)] = 0
                            # metric = ImBinaryMetric(y_test, y_pre)
                            mcc_score = matthews_corrcoef(y_test, y_pred)
                            auc = roc_auc_score(y_test, y_pred_proba,multi_class='ovr')
                            g_means = geometric_mean_score(y_test, y_pred,average='macro')
                            f1 = f1_score(y_test, y_pred,average='macro')
                            auc_all.append(auc)
                            f1_all.append(f1)
                            MCC.append(mcc_score)
                            gmeans_all.append(g_means)
                            del model

                    metrics = {'baseline': file_name,
                               'f1': f'{np.mean(f1_all):.3f} ',
                               'MCC': f'{np.mean(MCC):.3f} ',
                               'AUC': f'{np.mean(auc_all):.3f} ',
                               'Gean': f'{np.mean(gmeans_all):.3f}'}
                    new_row_df = pd.DataFrame([metrics])
                    df = pd.concat([df, new_row_df], ignore_index=True)

        df.to_excel(f'model_test.xlsx', index=False)

    return


if __name__ == '__main__':
    main()

