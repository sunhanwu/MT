import numpy as np
import copy
import torch.nn as nn
import torch
import torch.nn.functional as F

def aggreVocab(filePaths, outputPath):
    """
    提取原始数据的词表,并且保存为字典文件
    :param filePaths: 输入文件的位置, 支持单个路径或者一个路径列表
    :param outputPath: 保存文件的位置
    :return: None
    """
    # 判断filePaths类型，如果不是列表就构造成列表
    if isinstance(filePaths, str):
        filePaths = [filePaths]
    # vocabList 存储单个词
    vocabList = []
    # 遍历每个文件
    for file in filePaths:
        print('read {}'.format(file))
        with open(file, 'r') as f:
            # 遍历每一行
            for line in f.readlines():
                # 遍历每个词
                words = line.strip().split(' ')
                for word in words:
                    vocabList.append(word)
    # vocabSet存储经过去重的词表
    vocabSet = set(vocabList)
    # vocabDict 存储 word:index
    vocabDict = {'<START>': 0, '<END>': 1, 'UNK': 2, 'PAD': 3}
    for index, word in enumerate(vocabSet):
        # +3的原因是前面已经有三个值了
        vocabDict[word] = index + 4
    np.save(outputPath, vocabDict)

def clones(module, N):
    """
    module克隆函数
    :param module: 被克隆的module
    :param N: 克隆的次数
    :return: ModuleList
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    """
    由于transformer的Decoder是采用的自回归的机制，预测t的时候只能看见1到t-1部分的词，不能看见t+1之后的词
    所以需要将t+1后面的词给mask掉
    本函数根据size产生对应的mask矩阵
    :param size:
    :return:
    """
    attention_shape = (1, size, size)
    # 产生一个上三角矩阵(除了上三角位置的数据保留其他所有位置的数置为0)，k=1表示对角线也为0
    subsequent_mask = np.triu(np.ones(attention_shape), k=1).astype('uint8')
    # 通过==0的操作，下三角全置为1
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    """
    attention计算
    :param query: Q
    :param key: K
    :param value: V
    :param mask: mask矩阵
    :param dropout: dropout层，不是一个比例
    :return:
    """
    # 这里d_k=embedding_dim / head_num
    # query.shape=(batch_size, head_num, seq_len, d_k)
    # key.shape=(batch_size, head_num, seq_len, d_k)
    # value.shape=(batch_size, head_num, seq_len, d_k)
    embedding_dim = query.size(-1)
    # 这一步实现Q和K的attention计算，Q和K都是四维，只需要计算最后两维即可
    # Q.dot(K^T).shape = (batch_size, head_num, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(embedding_dim)
    if mask is not None:
        # 对于mask为0的位置，给填充一个特别小的数，使用softmax计算概率的时候基本为0
        scores = scores.masked_fill(mask==0, -1e9)
    # 对scores的最后一位进行softmax计算
    attention_weight = F.softmax(scores, dim=-1)
    # attention_weight.shape=(batch_size, head_num, seq_len, seq_len)
    if dropout is not None:
        # 此处的
        attention_weight = dropout(attention_weight)
    # attention_weight.dot(value).shape=(batch_size, head_num, seq_len, d_k)
    return torch.matmul(attention_weight, value), attention_weight

def collate_fn(data):
    """
    对batch级别进行padding
    :param data: 要padding的数据
    :return: 处理好的X, y, lengths before padding
    """
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_X = [x[0] for x in data]
    data_y = [x[1] for x in data]
    length_X = len(data_X)
    length_y = len(data_y)
    data_X = torch.nn.utils.rnn.pad_sequence(data_X, batch_first=True, padding_value=3)
    data_y = torch.nn.utils.rnn.pad_sequence(data_y, batch_first=True, padding_value=3)
    return data_X, data_y, torch.tensor(length_X), torch



if __name__ == '__main__':
    aggreVocab(['../data/train.zh.tok', '../data/valid.zh.tok', '../data/test.zh.tok'], '../data/vocab_zh.npy')
    aggreVocab(['../data/train.en.tok', '../data/valid.en.tok', '../data/test.en.tok'], '../data/vocab_en.npy')
