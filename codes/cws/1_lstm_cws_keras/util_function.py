
# coding: utf-8

# # 函数库

# In[1]:


# coding: utf-8

import re
import numpy as np
import copy


# ### 一、把汉字与标签分开
# ```
#   输入：['中/b','国/e']
#   输出：
#      data = ['中','国']
#      label = ['b','e']
# ```   

# In[2]:


def get_xy(s):  # 把汉字与标签分开
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        return list(s[:,0]), list(s[:,1])
    
def split_data_label(ss):
    data = [] 
    label = []
    for i in ss:  
        x = get_xy(i)
        if x:
            data.append(x[0])
            label.append(x[1])
    return data,label


# ### 二、准确率，召回率，F1的计算
# ```
# 输入：
#     sen1 = ['我', '从来', '没有','说', '不', '喜欢', '你']  # 正确的
#     sen2 = ['我', '从', '来','没','有','说','不','喜欢', '你'] # 预测的
# 输出：
#    (0.6250000000000001, 0.7142857142857143, 0.5555555555555556)
#     
# ```

# In[3]:


def calculate_evaluation(sen1,sen2):
    # 1、计算正确的词数
    TP = 0
    i = 0 # 句子0词的索引
    j = 0 # 句子1词的索引
    
    ii = 0 # 句子0字的索引
    jj = 0 # 句子1字的索引
    
    
    while i < len(sen1):
        if sen1[i] == sen2[j] and ii == jj :  # 找到的情况下
            ii+=len(sen1[i])
            jj+=len(sen2[j])
            i+=1
            j+=1
            TP+=1
        elif sen1[i] != sen2[j] and ii ==jj:
            ii+=len(sen1[i])
            jj+=len(sen2[j])
        else:
            if ii>jj:  # sen1词长于sen2 [我 喜欢 广州]  [我 喜 欢 广州]
                j+=1
                jj += len(sen2[j])
            else:  
                i+=1
                ii += len(sen1[i])
            if ii==jj:
                i+=1
                j+=1
        #print("i = %d,j = %d, ii = %d, jj = %d, TP = %d sen1[i] = %s, sen2[j] = %s"%(i,j,ii,jj,TP,sen1[i],sen2[j]))
    # 2、计算准确率、召回率、F1值
    m = len(sen2)  # 注意再错了
    n = len(sen1)
    
    precision = TP/m
    recall = TP/n
    f1 = 2*TP/(m+n)
    return f1,precision,recall,TP,m,n


# In[4]:


#sen1 = ['我', '喜欢', '广州']
#sen2 = ['我', '喜','欢','广州']
#print(sen2[2])
#calculate_evaluation(sen1,sen2)
#sen1 = ['他', '来到', '中国', '，', '成为', '第一个', '访', '华', '的', '大', '船主', '。']
#sen2 = ['他来', '到中', '国，', '成为', '第一', '个访', '华的', '大船', '主', '。']
#calculate_evaluation(sen1,sen2)


# ### 三、批量计算准确率、召回率、F1值

# In[5]:


def calculate_evaluation_batch(data1,data2):
    M = 0
    N = 0
    TP_S = 0
    for i,sen in enumerate(data1):
        f1,precision,recall,TP,m,n = calculate_evaluation(sen,data2[i])
        TP_S += TP
        M += m
        N += n
    F1 = 2*TP_S/(M+N)
    P = TP_S/M
    R = TP_S/N
    return F1,P,R


# ### 四、数据标签转为词的集合
# ```
# 输入：
#     data_x = [['我','从','来','没','有','说','不','喜','欢','你'],[...]]  |   data_x = ['我','从',...] 
#     data_y = [['s','b','e','b','e','s','s','b','e','s'],[...]]                  |   data_y = ['s','b',...]
#     op = [0|1]m= # 0表示两维数组  # 1表示一维数组
# 输出：
#     sen_reorganization = ['我', '从来', '没有','说', '不', '喜欢', '你']
# ```

# In[6]:


def data_label_to_word(data_x,data_y,op):
    sen_reorganization = []
    for i,sens in enumerate(data_x):
        sen_tmp = []
        for j,char in enumerate(sens):
            if data_y[i][j] == 's':
                sen_tmp.append(char)
            elif data_y[i][j] == 'b':
                new_word = ''
                new_word+=char
            elif data_y[i][j] == 'm':
                new_word+=char
            else:
                new_word+=char
                sen_tmp.append(new_word)
        sen_reorganization.append(sen_tmp)
    return sen_reorganization


# ### 五、模型中文分词预测函数
# ```
# 输入：
#     预测出每个类的概率，两维，一个句子(已取对数)，如test_x = [ [-1.23 -1.2 -23.3 -0.4 -0],[-2.3 -2.1 -1.2 -2.2 -0.2]];
# 输出：
#     ['b','e','s',...]
# ```

# In[7]:


def cws_pre(data_x):
    obs = data_x # 输入为发射概率矩阵
    dict_c = {0:'s',1:'b',2:'m',3:'e',4:'x'}
    #转移概率，单纯用了等概率,后面要对数据进行统计
    ee = 1e-300
    A = np.array([[0.5,0.5,ee,ee], # 一般情况下的状态转移矩阵
        [ee,ee,0.5,0.5],
        [ee,ee,0.5,0.5],
        [0.5,0.5,ee,ee]])
    
    for i,x in enumerate(A):   # 对数化
        for j,y in enumerate(x):
            A[i][j] = np.log(y)   
    # 维特比算法

    max_p = [0.5,0.5,1e-323,1e-323] # [max] 存储当前最大的概率 ，分别为每条路上的路径, 初始化为初始概率
    max_p = [np.log(x) for x in max_p]
    pathx = []    # 存储路径，一共有4条路径

    #　第一个字判断

    for i in range(4):
        max_p[i] = max_p[i]+obs[0][i]
        pathx.append([i])
    # 后面的字
    for i in range(len(obs)-2): 
        max_p_new = np.zeros(4) # 暂时存储最大概率
        pathx_new = [[0],[0],[0],[0]] #暂时存储路径
        for j in range(4): # 扫描当前的状态
            pro_max_tmp = -10000  # 很小的值就可以
            for k in range(4): # 扫描前面的4个最有可能的路径
                pro =  max_p[k] + A[k][j] + obs[i+1][j] 
                if pro > pro_max_tmp:
                    pro_max_tmp = pro   
                    path_tmp = copy.deepcopy(pathx[k])
                    path_tmp.append(j)
            max_p_new[j] = pro_max_tmp# 更新最大概率
            pathx_new[j] = path_tmp# 更新路径
        max_p = max_p_new
        pathx = pathx_new
    # 最后一个字的特殊处理，只有两种选择，即S与E
    # 最后一个是S的情况,两种情况，s -> s, e -> s
    p = np.array([-1000.0,-1000.0,-1000.0,-1000.0])
    if max_p[0] >= max_p[3]:
        p[0] = max_p[0]
        pathx[0].append(0)
    else:
        p[0] = max_p[3]
        pathx[3].append(0)
        pathx[0] = pathx[3]
        
    if max_p[1] >= max_p[2]:
        p[3] = max_p[1]
        pathx[1].append(3)
        pathx[3] = pathx[1]
    else:
        p[3] = max_p[2]
        pathx[2].append(3)
        pathx[3] = pathx[2]
    # 最后留下概率最大的路径
    pro_max_tmp = -10000
    num = np.argmax(p)
    pathx = pathx[num]
    pathx = [dict_c[x] for x in pathx]
    return pathx


# ### 五、批量数据预测函数
# ```
# 输入：
#     预测出每个类的概率，三维，多个句子（已取对数），如test_x = [[[-1.23 -1.2 -23.3 -0.4 -0],[-2.3 -2.1 -1.2 -2.2 -0.2],[[...]]];
# 输出：
#     [['s','b','e'],[...]]
# ```

# In[8]:


def cws_pre_batch(data):
    labels = []
    for i,sen in enumerate(data):
        path = cws_pre(sen)
        labels.append(path)
    return labels


# In[9]:


'''
xx = 1e-200
a = np.array([[xx,xx,0.999,xx,xx],
              [xx,xx,0.999,xx,xx],
              [xx,xx,0.999,xx,xx],
              [xx,ee,ee,0.99,xx],
              [0.99,xx,xx,xx,xx]])

for i,x in enumerate(a):
    for j,y in enumerate(x):
        a[i][j] = np.log(y)

result = cws_pre(a)
print(result)
'''

