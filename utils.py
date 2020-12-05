import numpy as np
import copy
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sys
sys.path.append('.')

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
    # vocabCount = collections.Counter(vocabList).most_common(2)
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

def batch_subsequent_mask(batch:list):
    """
    调用subsequent_mask函数，将batch转换为一个三维的mask
    :param batch: 一维的tensor，表示每个seq的长度
    :return: 三维tensor，(batch_size, size, size)
    """
    mask = []
    for item in batch:
        mask.append(subsequent_mask(item).squeeze(0))
    return torch.tensor(mask)

def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & torch.tensor(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask



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
    # mask.shape=(batch_size, 1, 1, seq_len)
    embedding_dim = query.size(-1)
    # 这一步实现Q和K的attention计算，Q和K都是四维，只需要计算最后两维即可
    # scores.shape = Q.dot(K^T).shape = (batch_size, head_num, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(embedding_dim)
    # scores.shape=(batch_size, head_num, seq_len, seq_len)
    if mask is not None:
        # 对于mask为0的位置，给填充一个特别小的数，使用softmax计算概率的时候基本为0
        scores = scores.masked_fill(mask==0, -1e9)
        # scores.shape=(batch_size, head_num, seq_len, seq_len)
    # 对scores的最后一位进行softmax计算
    attention_weight = F.softmax(scores, dim=-1)
    # attention_weight.shape=(batch_size, head_num, seq_len, seq_len)
    if dropout is not None:
        # 此处的
        attention_weight = dropout(attention_weight)
    # attention_weight.dot(value).shape=(batch_size, head_num, seq_len, d_k)
    return torch.matmul(attention_weight, value), attention_weight

def collate_fn(data, padding_length):
    """
    对batch级别进行padding
    :param data: 要padding的数据
    :param padding_length: padding的长度
    :return: 处理好的X, y, lengths before padding
    """
    pass
    # data_X = [x[0] for x in data]
    # data_y = [x[1] for x in data]
    # length_X = [len(x) for x in data_X]
    # length_y = [len(x) for x in data_y]
    # data_X = [torch.cat()]
    # data_X= F.pad(data_X, )
    # return data_X, data_y, torch.tensor(length_X), torch.tensor(length_y)

def compute_bleu(translate, reference, references_lens):
    """
    计算翻译句子的的BLEU值
    :param translate: transformer翻译的句子
    :param reference: 标准译文
    :return: BLEU值
    """
    # 定义平滑函数
    translate = translate.tolist()
    reference = reference.tolist()
    smooth = SmoothingFunction()
    references_lens = references_lens.tolist()
    blue_score = []
    for translate_sentence, reference_sentence, references_len in zip(translate, reference, references_lens):
        if 1 in translate_sentence:
            index = translate_sentence.index(1)
        else:
            index = len(translate_sentence)
        blue_score.append(sentence_bleu([reference_sentence[:references_len]], translate_sentence[:index], weights=(0.3, 0.4, 0.3, 0.0), smoothing_function=smooth.method1))
    return blue_score


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

if __name__ == '__main__':
    aggreVocab(['../data/train.en.tok', '../data/valid.en.tok', '../data/test.en.tok'], '../data/vocab_en_freq_2.npy')
    aggreVocab(['../data/train.zh.tok', '../data/valid.zh.tok', '../data/test.zh.tok'], '../data/vocab_zh_freq_2.npy')
