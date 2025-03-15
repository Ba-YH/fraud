# 加载并保存预训练词向量模型
from gensim.models import KeyedVectors

for file in ["sgns.wiki.bigram-char"]:
    model = KeyedVectors.load_word2vec_format(
        file,
        binary=False,
        encoding='GBK',
        unicode_errors='ignore'
    )
    model.save(f"pretrained_models/{file}")
    print(f"Loading and saving KeyedVectors model {file}")