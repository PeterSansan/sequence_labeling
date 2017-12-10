
# coding: utf-8

# # 1、Bilstm中文分词实验
# ```
# 数据集:
#     训练集：msr_training_p3.txt（已打好标签）
#     测试集：msr_test_gold_p3.txt（已打好标签）
# 运行环境：
#     python3+Tensorflow 1.4+Keras 2.1.2
# ```

# In[14]:


# -*- coding:utf-8 -*-
# 第一版中文分词程序：单向lstm
import re
import copy
import tensorflow as tf
import random
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.utils import np_utils
import keras.preprocessing.text as T
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from util_function import split_data_label,calculate_evaluation,calculate_evaluation_batch,data_label_to_word,cws_pre,cws_pre_batch
from keras.callbacks import EarlyStopping

# 限制显存占比
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 1.0
sess = tf.Session(config=config)
set_session(sess)

# 越参数
word_size = 128  # 词向量的维度
maxlen = 128     # 句子最长的词数
epochs = 150     # 训练次数
TRAIN_PERC = 0.9 # 训练集占所有数据的比重
batch_size = 1 # 批处理数


# ## 1、加载数据

# In[15]:


s = [] 
with open('msr_train_p3.txt','r') as inputs:
    for line in inputs:
        line = line.strip()
        s.append(line)
t = [] 
with open('msr_test_gold_p3.txt','r') as inputs:
    for line in inputs:
        line = line.strip()
        t.append(line)
        
print("训练集句子数：",len(s))
print("测试集句子数：",len(t))


# In[16]:


train_x = [] #生成训练样本
train_y = []
test_x = [] #生成训练样本
test_y = []

# 训练集汉字与标签分开
train_x,train_y = split_data_label(s)
# 测试集汉字与标签分开
test_x,test_y = split_data_label(t)


# ## 2、数据清洗

# #### 2.1 超过maxlen长度的句子暂时滤掉

# In[17]:


train = pd.DataFrame(index=range(len(train_x)))
train['train_x'] = train_x
train['train_y'] = train_y
train = train[train['train_x'].apply(len) <= maxlen]  # 如果大于maxlen的句子滤掉
train.index = range(len(train))
print('过滤后剩余的训练集句子数 = ',len(train))

test= pd.DataFrame(index=range(len(test_x)))
test['test_x'] = test_x
test['test_y'] = test_y
test = test[test['test_x'].apply(len) <= maxlen]  # 如果大于maxlen的句子滤掉
test.index = range(len(test))
print('过滤后剩余的测试集句子数 = ',len(test))

train_x = list(train['train_x'])
train_y = list(train['train_y'])

test_x = list(test['test_x'])
test_y = list(test['test_y'])


# #### 2.2 重构标准的测试数据，与预测的做对比

# In[18]:


test_reorganization = data_label_to_word(test_x,test_y,0)
test_x_origin = copy.deepcopy(test_x)   # 一维数据
#for sen in test_x:
#    test_x_origin.extend(sen)


# #### 2.3 数据Token化，汉字转为数字序号

# In[19]:


for i,line in enumerate(train_x):
    str_tmp = ''
    for char in line:
        str_tmp+=char+' '
    train_x[i] = str_tmp
    
for i,line in enumerate(train_y):
    str_tmp = ''
    for char in line:
        str_tmp+=char+' '
    train_y[i] = str_tmp

for i,line in enumerate(test_x):
    str_tmp = ''
    for char in line:
        str_tmp+=char+' '
    test_x[i] = str_tmp
    
for i,line in enumerate(test_y):
    str_tmp = ''
    for char in line:
        str_tmp+=char+' '
    test_y[i] = str_tmp


# In[20]:


# token 序列化
tokenizer_x = Tokenizer(num_words=None)
tokenizer_x.fit_on_texts(np.concatenate((train_x,test_x),axis=0))
word_index = tokenizer_x.word_index # 词_索引,字典
index_word = dict(zip(word_index.values(), word_index.keys())) # 从下标1开始

print("汉字个数：",len(word_index))

tokenizer_y = Tokenizer(num_words=None)
tokenizer_y.fit_on_texts(['s b m e'])
label_index = tokenizer_y.word_index

train_x = tokenizer_x.texts_to_sequences(train_x)
train_y = tokenizer_y.texts_to_sequences(train_y) 

test_x = tokenizer_x.texts_to_sequences(test_x)
test_y = tokenizer_y.texts_to_sequences(test_y)

print(label_index)


# In[9]:


# 上面的标签序列从1开始，改为从0开始
for i,line in enumerate(train_y):
    for j,num in enumerate(line):
        train_y[i][j] = num-1
for i,line in enumerate(test_y):
    for j,num in enumerate(line):
        test_y[i][j] = num-1


# In[10]:


# 记录测试集每个句子的长度
len_test = []
for i in test_y:
    len_test.append(len(i))


# #### 2.4 为不够长的句子填充特征值，使得句子长度一致

# In[11]:


# 训练集填充，一个句子的字数小于maxlen，后面填充0
train_x = sequence.pad_sequences(train_x, maxlen=maxlen,padding='post')
train_y = sequence.pad_sequences(train_y, maxlen=maxlen,padding='post',value=4.)
train_y = to_categorical(train_y, num_classes=5)
train_y = train_y.reshape(-1,maxlen,5)

# 测试集要填充，否则没法用to_categoriecal函数，如果怀疑这样的准确度，可以后面再验证。
test_x = sequence.pad_sequences(test_x, maxlen=maxlen,padding='post')
test_y = sequence.pad_sequences(test_y, maxlen=maxlen,padding='post',value=4.)
test_y = to_categorical(test_y, num_classes=5)
test_y = test_y.reshape(-1,maxlen,5)


# #### 2.5 按比例分配训练集与验证集，还未做交叉验证

# In[12]:


#print(train_x.shape)
#print(train_y.shape)
#print(test_x.shape)
#print(test_y.shape)


# In[13]:


# 打散数据
indices = np.arange(train_x.shape[0])
np.random.shuffle(indices)
train_x = train_x[indices]
train_y = train_y[indices]

# 取部分数据
#train_x=train_x[:10000]
#train_y=train_y[:10000]


# 分开训练集与验证集
len_train = int(train_x.shape[0]*TRAIN_PERC)
len_develop = int(train_x.shape[0]-len_train)

print('训练集句子数 = ',len_train)
print('验证集句子数 = ',len_develop)

develop_x = train_x[len_train:]
develop_y = train_y[len_train:]

train_x = train_x[:len_train]
train_y = train_y[:len_train]

#train_y = train_y.reshape((-1,maxlen,5))
#develop_y = develop_y.reshape((-1,maxlen,5))


# ## 3、模型设计

# In[ ]:


#设计模型
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional,Dropout
from keras.models import Model
from keras.optimizers import SGD,RMSprop,Adagrad,Adadelta,Adamax,Adam

# 函数式模型
sequence = Input(shape=(maxlen,), dtype='int32')
embedded = Embedding(len(word_index), word_size, input_length=maxlen, mask_zero=False)(sequence)
#embedded = Embedding(len(word_index)+1, word_size, input_length=maxlen, mask_zero=True)(sequence)
blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
blstm = Dropout(0.5)(blstm)
blstm = Bidirectional(LSTM(32, return_sequences=True), merge_mode='sum')(blstm)
#output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
output = Dense(5, activation='softmax')(blstm)
model = Model(inputs=sequence, outputs=output)
#op = Adam(lr=0.0015, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#op = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()
# 设置early stopping
early = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, mode='auto')
history = model.fit(train_x, train_y, 
                    validation_data=(develop_x,develop_y),
                    batch_size=batch_size, 
                    epochs=epochs,
                    callbacks=[early],
                    verbose=2)


# ## 四、预测

# #### 4.1 求出测试集的发射概率矩阵

# In[ ]:


# 这里的测试集是有填充的，后面可以把填充的去掉试一试
score, acc = model.evaluate(test_x, test_y,
                            batch_size=batch_size,verbose=2)

print('Test score:',score)
print('Test accuracy:',acc)


# In[ ]:


b = model.predict(test_x)   # 发射概率
b = np.log(b)
B = []
# 按真实句子的长度截断
for i,sens in enumerate(b):
    B.append(sens[:len_test[i]])
del b
# 这下面的代码是错的，并不能简单地取最大概率的那个标签，因为可能造成不合理的现象。
#with sess.as_default():
#    acc = tf.argmax(test_y,2)
#    pre = tf.argmax(pre,2)
#    acc = acc.eval()
#    pre = pre.eval()


# #### 4.2 维特比算法解码，得到合理化的标签序列 

# In[ ]:


pre_y = cws_pre_batch(B) # 通过维特比算法求出合理的标签


# #### 4.3 重构分词后的句子

# In[ ]:


pre_reorganization = data_label_to_word(test_x_origin,pre_y,0)


# #### 4.4 计算测试集的准确率、召回率、F1值

# In[ ]:


# 计算评估指标
F,P,R = calculate_evaluation_batch(test_reorganization,pre_reorganization)
print(F,P,R)

