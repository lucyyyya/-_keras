import jieba
import pandas as pd
from sklearn import metrics
from keras.models import load_model
from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer


max_len = 10
max_words = 100000

df = pd.read_table('validation_data_demo.txt', sep='\t')
df['cutted'] = df.text.apply(lambda x: " ".join(jieba.cut(x)))

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df.cutted)
sequences = tokenizer.texts_to_sequences(df.cutted)
test_seq_mat = sequence.pad_sequences(sequences,maxlen=max_len)

model=load_model('mymodel.h5')
y_pre=model.predict(test_seq_mat)
df['sentiment']=y_pre
df["predict"] = df["sentiment"].apply(lambda x: 1 if x>0.5 else 0)
df=df[['text','sentiment','label','predict']]
print(df)
print("准确率")
print(metrics.accuracy_score(df.label, df.predict))
print(metrics.classification_report(df.label,df.predict))
