# 1.基础实验

## 1.1 LSTM中文分词

### 一、问题与解决：
**现有方案：**模型采用`Embedding+BiLSTM`，前后向的输出相加（别一种是串联），数据滤掉了过长的句子，并保留了标点符号，标点符号与汉字同级对待

- [x] **1、 比较同一模型同一参数下Python2与Python3的效率问题**
【模型】`Embedding+BiLstm(64)+softmax`
【参数】`batch_size = 1024`	 
【实验结果】
```	
	batch_size = 1024 ,GPU占用限制一半
	python2 :  57s/ 57s/ 57s 
	python3 : 58s/ 57s/ 57s

	batch_size - 1024,GPU不限制
	python2 : 58s/ 57s/ 57s 
	python3 : 58s/ 57s/ 57s
```
**以下部分均取用如下模型**
![@双向LSTM | center](http://ogtxggxo6.bkt.clouddn.com/model_1.png)
- [x] **2.、【viterbi】预测过程中为什么要使用viterbi算法的解决**
 答：采用viterbi的目的是去除安全不合理的预测结果，如`我/s 喜/b 欢/e 红/b 色/e`，如果不加viterbi可能出现`我/e 喜/e 欢/e 红/e 色/e`这种错得离谱的答案，因为`e`标签是不能出现在一个词的词首的，这种标签不可能通过语法重构成词，所以很有必要用viterbi算法。
 
- [x] **3、 Keras中使用TimeDistributed的区别**
答：官网的解释，该包装器可以把一个层应用到输入的每一个时间步上，把输入第一维做为时序，后面所有的层都是作用在每个时序上的。
【模型】同上
【参数】
【实验结果】
 模型|没有TimeDi|TimeDi|
:--:|:--:|:--:|
每轮时间|24 s|24 s|
停止轮|55|51
F1|0.913|0.910
【结论】从结果来看，如果放在最后好像对结果没什么影响

- [ ] **4、目前的方案是把超过maxlen的句子去掉，不够长的句了补到maxlen，是否可以更加动态,目前TF中四种RNN函数中都有`sequence_length`这个参数，可以动态计算**
 答：用CudnnLSTM替换了LSTM函数后，但不支持
- [x] **5.目前汉字个数为5111个，比较少，如果出现  没有见过的词会怎么样。**
	答：目前的程序会自动把不认识的字过滤掉，主要是texts_to_sequences的功能，所以要保证所有字都认识
- [ ] **6、 Embedding函数中mask_zero设为True的影响，文档中对变长数据有效，目前的设置为False**
- [ ] 
- [x] **7、【batch的影响】结果详细见12月6日汇报的doc文件**
- [x] **8、【评估标准】**
$$
P = \frac{TP}{TP+FP} = \frac{TP}{M}
$$
$$
R = \frac{TP}{TP+FN}=\frac{TP}{N}
$$
$$
F1 = \frac{2PR}{P+R} = \frac{2TP}{2TP+FP+FN}=\frac{2TP}{M+N}
$$
- [ ] **9、cudnnLstm的使用，比较效率精度**
- [ ] **10、GRU核换掉LSTM核，比较效率精度**
- [ ] **11、全面编写TF，比较效率精度**
- [x] **12、用TF最新加的CudnnLSTM代替LSTM**
【使用模型】同上
【参数】batch_size = 2048
【实验结果】

 模型      |     LSTM |  CudnnLSTM   
 :------: | :-------:| :------: 
  时间     |   27s      |   7s  |
  【结论】使用CudnnLSTM快了3~4倍，但把
参考代码：
- [4. 基于双向LSTM的seq2seq字标注](http://spaces.ac.cn/archives/3924/comment-page-1#comments)
- [深度学习将会变革NLP中的中文分词](https://www.leiphone.com/news/201608/IWvc75oJglAIsDvJ.html)：这篇文章lstm模型的输入是前面n个字+当前字+后面n个字，输出为标签；并按2n+1的时间段做梯度更新；这个类似用lstm做手写识别的例子。最后输出的单个标签。


