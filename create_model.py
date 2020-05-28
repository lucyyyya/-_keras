import pandas as pd
import numpy as np
import jieba
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential

from gensim.models import KeyedVectors

maxlen = 20
max_words = 100000
# ----------------导入数据--------------------
df = pd.read_csv('train_data_fix.txt', sep='\t')
df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
df['text'] = df.content.apply(lambda x: " ".join(jieba.cut(x)))
df = df[['text', 'sentiment']]


# 把分词之后的评论信息，转换成为一系列的数字组成的序列。
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df.text)
sequences = tokenizer.texts_to_sequences(df.text)

# -----------------切割，不过好像都挺短的--------------
data = pad_sequences(sequences, maxlen=maxlen)

# -------------保存词典-----------------
word_index = tokenizer.word_index

labels = np.array(df.sentiment)

# ---------序号随机化，但保持数据和标记之间的一致性。-----------
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# -------划分训练集和验证集（8：2）------------
training_samples = int(len(indices) * .8)
validation_samples = len(indices) - training_samples

X_train = data[:training_samples]
y_train = labels[:training_samples]
X_valid = data[training_samples: training_samples + validation_samples]
y_valid = labels[training_samples: training_samples + validation_samples]

# ------------词嵌入-------------
# 读入词嵌入预训练模型数据。
zh_model = KeyedVectors.load_word2vec_format('zh.vec')
# 向量长度保存,建立随机矩阵,数据在-1到1之间
embedding_dim = len(zh_model[next(iter(zh_model.vocab))])
embedding_matrix = np.random.rand(max_words, embedding_dim)
embedding_matrix = (embedding_matrix - 0.5) * 2

# 词汇是否训练过
for word, i in word_index.items():
    if i < max_words:
        try:
            embedding_vector = zh_model.get_vector(word)
            embedding_matrix[i] = embedding_vector
        except:
            pass

# ----------------模型构造-------------------
units = 32

model = Sequential()
model.add(Embedding(max_words, embedding_dim))
model.add(LSTM(units))
model.add(Dense(1, activation='sigmoid'))
# 输出为一个0到1中间的数值
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

# 保存输出为 history
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_valid, y_valid))
# 保存模型
model.save("mymodel.h5")

