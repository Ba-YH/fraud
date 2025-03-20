import numpy as np
import pandas as pd
import csv
import jieba
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse
import joblib
import os

stop_words=[]
def read_data():
    global stop_words
    with open("stop_words.txt", "r", encoding="UTF-8") as f:
        stop_words = set(word.strip() for word in f.readlines())
def preprocess_text(text):
    words = jieba.lcut(text)
    return [word for word in words if word not in stop_words]


# 使用处理数据时保存的模型在线处理输入数据
def process_text(words):
    word2vec_model = Word2Vec.load("models/word2vec.model")
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")

    if not words:
        return np.zeros(word2vec_model.vector_size)

    # 生成句子向量
    vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if not vectors:
        sentence_vector = np.zeros(word2vec_model.vector_size)
    else:
        sentence_vector = np.mean(vectors, axis=0)

    # 标准化和PCA
    scaled_vector = scaler.transform([sentence_vector])
    pca_vector = pca.transform(scaled_vector)
    return pca_vector.flatten()


def process_files(args):
    # 先读取停用词
    read_data()
    dfs=[]
    for number in args.file_numbers:
        df = pd.read_csv(f"../cleaned_data/label0{number}-last.csv",encoding="UTF-8")
        dfs.append(df)

    # 合并到一个新的dataframe
    df = pd.concat(dfs,axis=0,ignore_index=True)
    data = df['content'].apply(preprocess_text)

    if args.use_pretrained:
        model = KeyedVectors.load(f"pretrained_models/{args.pretrained_model}")
        model.wv=model
    else:
        model = Word2Vec(
            sentences=data.tolist(),
            vector_size=args.vector_size,
            window=args.window,
            min_count=1,
            workers=4,
            seed=42,
        )

    def get_sentence_vector(words):
        vectors = [model.wv[word] for word in words if word in model.wv]
        if len(vectors) == 0:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)

    X = np.array([get_sentence_vector(words) for words in data])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(args.n_components)
    X_pca = pca.fit_transform(X_scaled)


    file_path="../process_data/label"
    if args.use_pretrained:
        file_path+=''.join(str(num) for num in args.file_numbers)
        file_path+=f"-{args.pretrained_model}-{args.n_components}"
    else:
        for name, value in vars(args).items():
            if name == "file_numbers":
                for num in value:
                    file_path += f"{num}"
            else:
                file_path += f"-{value}"
    file_path+=".csv"

    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = [f"V{i}" for i in range(X_pca.shape[1])]
        header.append("label")
        writer.writerow(header)

        # 逐行写入特征和对应的标签
        for features, label in zip(X_pca, df['label']):
            writer.writerow([*features, label])

    print(f"{file_path}写入完毕")

    # 保存模型
    joblib.dump(scaler, "../models/scaler.pkl")
    joblib.dump(pca, "../models/pca.pkl")
    model.save(f"../models/word2vec.model")
    print("模型保存完毕")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_numbers",type=list,default=[0,1,2,3,4])

    # Word2Vec参数
    parser.add_argument("--vector_size",'-vs',type=int,default=100)
    parser.add_argument("--window",type=int,default=5)
    parser.add_argument("--n_components",'-nc',type=float,default=0.99)

    # 预训练参数
    parser.add_argument("--use_pretrained",'-up',default=False)
    parser.add_argument("--pretrained_models",type=str,default="sgns.wiki.bigram-char")

    args = parser.parse_args()
    process_files(args)