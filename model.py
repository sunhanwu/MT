"""
下面代码中实现的transformer模型参考知乎文章https://zhuanlan.zhihu.com/p/118601295
"""
import torch
from torch import nn
from dataloader import Data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from utils import *
import copy
import argparse
from tqdm import tqdm

class TranslateEn2Zh(nn.Module):
    """
    英文翻译为中文的模型, 通用的Encoder和Decoder框架
    """
    def __init__(self, encoder, decoder, src_embedding, dst_embedding, generator):
        """
        构造函数, 使用Encoder和Decoder通用框架实现一个Transformer模型
        :param encoder: 编码器, 本例中使用Transformer的Encoder
        :param decoder: 解码器, 本例中使用Transformer的Decoder
        :param src_embedding: 源语言的embedding
        :param dst_embedding: 目标语言的embedding
        :param generator: 将Decoder输入的隐状态输入一个全连和softmax用于输出概率
        """
        super(TranslateEn2Zh, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.dst_embedding = dst_embedding
        self.generator = generator

    def forward(self, english_seq, chinese_seq, english_mask, chinese_mask):
        """
        前向传播函数
        :param english_seq: 英文序列
        :param chinese_seq: 中文序列
        :param english_mask: ???
        :param chinese_mask: ???
        :return:
        """
        memory = self.encode(english_seq, english_mask)
        output = self.decode(memory, english_mask, chinese_seq, chinese_mask)
        return output

    def encode(self, english_seq, english_mask):
        """
        Transformer的编码器
        :param english_seq: 英文序列
        :param english_mask:
        :return:
        """
        return self.encoder(self.src_embedding(english_seq), english_mask)

    def decode(self, memory, english_mask, chinese_seq ,chinses_mask):
        """
        Transformer的解码器
        :param memory: 应该是encoder编码后的输出
        :param english_mask: 英文序列mask
        :param chinese_seq: 中文序列
        :param chinses_mask: 中文序列mask
        :return:
        """
        return self.decoder(self.dst_embedding(chinese_seq), memory, english_mask, chinses_mask)


class Generator(nn.Module):
    """
    根据Decoder输出的隐藏状态输出一个词
    """
    def __init__(self, decoder_dim, vocab_len):
        """
        generator的构造函数
        :param decoder_dim: decoder的输出的维度
        :param vocab_len: 词典大小
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(decoder_dim, vocab_len)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输出x，这里是decoder的输出
        :return:
        """
        proj = self.proj(x)
        # 做一次softmax, 返回的是softmax的log值
        # log_softmax + NLLLoss 效果类似与 softmax + CrossEntropyLoss
        return F.log_softmax(proj, dim=-1)

class TransformerEncoder(nn.Module):
    """
    Transformer的Encoder部分
    由6个EncoderLayer堆叠而成，而每个EncoderLayer又包含一个self-attention层和全连层
    """
    def __init__(self, encode_layer, N):
        """
        构造函数
        :param encode_layer: encode_layer, 包含一个self-attention层和一个全连层
        :param N: encoder_layer 重复的次数，transformer中为6
        """
        super(TransformerEncoder, self).__init__()
        # 6层encode_layer
        self.layers = clones(encode_layer, 6)
        # 再加一层Norm层
        self.norm = nn.LayerNorm(encode_layer.size)

    def forward(self, x, mask):
        """
        前向函数
        :param x: 待编码的数据
        :param mask:
        :return: Transformer的Encoder编码后的数据
        """
        for layer in self.layers:
            x = layer(x, mask)
        # 最后加一层Normalization层
        return self.norm(x)

class EncodeLayer(nn.Module):
    """
    transformer中Encoder部分的encode layer，一共6个encode layer组成一个Encoder
    一个encode layer包含两个子层, 每个子层包括self-attention、feed_forward等操作
    """
    def __init__(self, size, self_attention, feed_forward, dropout):
        """
        构造函数
        :param size:
        :param self_attention:
        :param feed_forward:
        :param dropout:
        :return:
        """
        super(EncodeLayer, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        # 两个子层
        self.sublayer = clones(SubLayer(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        前向函数
        :param x: 输入
        :param mask:
        :return:
        """
        # 进行self attention
        # self-attention需要四个输入， 分别是Query，Key，Value和最后的Mask
        # lambda 表达式中的x不是EncodeLayer的输入x，而是一个形式参数，可以是y或者其他任何名称，最终输入到lambda中的应该是SubEncodeLayer层中的self.norm(x)
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, mask))
        # 进行feed_forward
        return self.sublayer[1](x, self.feed_forward)


class SubLayer(nn.Module):
    """
    Encode Layer或者Decode Layer的一个子层(通用结构)
    这里会构造LayerNorm 和 Dropout，但是Self-Attention 和 Dense 不在这里构造，作为参数传入
    """
    def __init__(self, size, dropout):
        """
        构造函数
        :param size:
        :param dropout:
        """
        super(SubLayer, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, function_layer):
        """
        前向函数
        :param x: 传入数据
        :param sublayer: 功能层，Encoder中可以为self-attention 或者 feed_forward中的一个
        :return:
        """
        # x + 对应那个残差操作
        # 这个dropout和原文不太一样，加在这里是为了防止过拟合吧
        # layer normalization的位置也和原文不太一样，原文是放在最后，这里是放在最前面并且在最后一层再加一层layer normalization
        # 为了方便SubEncodeLayer层的复用，self-attention和feed_forward作为参数<function_layer>
        return x + self.dropout(function_layer(self.norm(x)))

class TransformerDecoder(nn.Module):
    """
    Transformer的Decoder端，包含6个DecodeLayer, 每个DecoderLayer又包含三个子层，分别是self-attention层，attention层和feed_foraward层
    """
    def __init__(self, decode_layer, N):
        """
        构造函数
        :param layer: DecodeLayer
        :param N: Transformer中的Deocder包含6个Deocde层，所以N为6
        """
        super(TransformerDecoder, self).__init__()
        self.layers = clones(decode_layer, N)
        self.norm = nn.LayerNorm(decode_layer.size)

    def forward(self, x, memory, src_mask, dst_mask):
        """
        前向传播函数
        :param x: 自回归输入，一开始只有一个起始符
        :param memory: 应该是Encoder编码的信息
        :param src_mask: padding mask
        :param dst_mask: 防止标签泄露的mask
        :return:
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, dst_mask)
        return self.norm(x)

class DecodeLayer(nn.Module):
    """
    Transformer的Decoder的一个Decode层
    包含三个子层，分别对应self-attention, attention(与Encoder编码的memory做attention), feed_forward
    """
    def __init__(self, size, self_attenton, attention, feed_forward, dropout):
        """
        构造函数
        :param size:
        :param self_attenton: self attention层
        :param attention: 与Encoder编码的memory做attention
        :param feed_forward: 全连层
        :param dropout: 防止过拟合加的dropout层
        """
        super(DecodeLayer, self).__init__()
        self.size = size
        self.self_attention = self_attenton
        self.attention = attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayer(size, dropout), 3)

    def forward(self, x, memory, src_mask, dst_mask):
        """
        前向传播函数
        :param x: 输出
        :param memory: Encoder编码的memory
        :param src_mask: 源端的padding mask
        :param dst_mask: 防止标签泄露的mask
        :return:
        """
        m = memory
        # 这里是self-attention子层，对于self-attention来说，Query, Key和Value都是等于x
        # lambda表达式中的x只是一个形式参数，不是输入x
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, dst_mask))
        # 第二个子层是attention层，与memory做attention, 此时的Query为x，Key为m，Value也为m
        x = self.sublayer[1](x, lambda x: self.attention(x, m, m, src_mask))
        # 第三个子层是一个全连层
        output = self.sublayer[2](x, self.feed_forward)
        return output


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制的实现
    """
    def __init__(self, head_num, multi_output_dim, dropout=0.1):
        """
        构造函数
        :param head_num: head的数目
        :param d_model: Multi-Head输出的维度，就等于embedding_dim
        :param dropout: dropout率
        """
        super(MultiHeadAttention, self).__init__()
        assert multi_output_dim % head_num == 0
        self.d_k = multi_output_dim // head_num
        self.head_num = head_num
        # 下面的四个线性层的前三个相当于对Q，K, V分别乘以一个权重矩阵，最后一个对计算完的整体结果加一个权重矩阵
        self.linears = clones(nn.Linear(multi_output_dim, multi_output_dim), 4)
        self.attention = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        前向传播函数
        :param query: Q
        :param key: K
        :param value: V
        :param mask:
        :return:
        """
        # query.shape=key.shape=value.shape=(batch_size, seq_len, embedding_dim)
        # 注意这里的embedding_dim = multi_output_dim，为了保证后面计算的一致性
        if mask is not None:
            # 扩维，原本mask.shap = (batch_size, 1, seq_len)
            # 扩充之后变为(batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1)
            # 由于广播机制，所有head的mask都一样
        batch_size = query.size(0)
        # 首先使用线性变换，然后将multi_output_dim分配给head_num个头，每个头为 d_k = multi_output_dim / head_num
        # 经过线性变换后，query，key，value等的维度不变，还是(batch_size, seq_len, embedding_dim)
        # 在经过view操作后，query.shape=(batch_size, seq_len, head_num, d_k)
        # 再经过transpose操作后，query.shape=(batch_size, head_num, seq_len, d_k)和attention函数要求的维度一致
        query, key, value = [l(x).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attention = attention(query, key, value, mask=mask, dropout=self.dropout)
        # attention 函数操作完之后x.shape=(batch_size ,head_num, seq_len, d_k)
        # self.attention.shape=(batch_size, head_num ,seq_len, seq_len)

        # 下面将multi head的最后一个维度d_k拼接在一起，然后再使用一个线性变换，最终embendding_dim维度不变
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head_num * self.d_k)
        # 此时的x.shape=(batch_size, seq_len, embedding_dim)
        return self.linears[-1](x)

class PositionWiseFeedForward(nn.Module):
    """
    全连层，有两个线性变化和之间的ReLU组成
    """
    def __init__(self, embedding_dim, hidden_num, dropout=0.1):
        """
        构造函数
        :param embedding_dim: embedding_dim 也等于 multi_output_dim
        :param hidden_num: 中间隐藏单元的个数
        :param dropout: dropout
        """
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_num)
        self.linear2 = nn.Linear(hidden_num, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入
        :return:
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class Embeddings(nn.Module):
    """
    原始输出的seq是每个word在vocab中的index的序列，需要embedding序列
    """
    def __init__(self, embedding_dim, vocab_len):
        """
        构造函数
        :param embedding_dim: embedding的维度
        :param vocab_len: 词表vocab的长度
        """
        self.lut = nn.Embedding(vocab_len, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # 这里没看懂为啥要乘以embedding_dim的平方根
        return self.lut(x) * torch.sqrt(self.embedding_dim)


class PositionalEncoding(nn.Module):
    """
    位置编码
    """
    def __init__(self, embedding_dim, dropout, max_len=5000):
        super(PositionalEncoding, self)
        self.dropout = nn.Dropout(p=dropout)

        # 在log空间中计算位置编码
        positionalEncode = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(torch.log(10000.0) / embedding_dim))
        positionalEncode[:, 0::2] = torch.sin(position * div_term)
        positionalEncode[:, 1::2] = torch.cos(position * div_term)
        positionalEncode = positionalEncode.unsqueeze(0)
        # 创建一个buffer，将pe保存下来
        self.register_buffer('pe', positionalEncode)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入
        :return:
        """
        x = x + torch.tensor(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

if __name__ == '__main__':
    # 设置命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100, help='minibatch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--embedding_dim', type=int, default=512, help='number of word embedding')
    parser.add_argument('--gpu', type=int, default=0, help='GPU NO, only support 1 or 2')
    parser.add_argument('--head_num', type=int, default=8, help="Multi head number")
    parser.add_argument('--hidden_num', type=int, default=2048, help="hidden neural number")
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout rate")
    args = parser.parse_args()

    # 指定device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 准备数据读取数据
    trainData = Data(mode='train')
    validData = Data(mode='valid')
    testData = Data(mode='test')

    # 准备dataloader, 使用tqdm封装打印进度条, 方便查看运行进度
    trainDataLoader = tqdm(DataLoader(trainData, batch_size=args.batch_size, shuffle=True))
    validDataLoader = tqdm(DataLoader(validData, batch_size=args.batch_size, shuffle=True))
    testDataLoader = tqdm(DataLoader(testData, batch_size=args.batch_size, shuffle=True))

    # 对于Vocab来说，三种模式下的vocabZh和vocabEn都一样
    zhVocabLen = trainData.getVocabZhLen()
    enVocabLen = trainData.getVocabEnLen()

    # 实例化attention
    cp = copy.deepcopy
    attention = MultiHeadAttention(head_num=args.head_num, multi_output_dim=args.embedding_dim).to(device)
    feedForward = PositionWiseFeedForward(embedding_dim=args.embedding_dim, hidden_num=args.hidden_num, dropout=args.dropout).to(device)
    position = PositionalEncoding(embedding_dim=args.embedding_dim, dropout=args.dropout).to(device)

    # 构建Encoder
    encodeLayer = EncodeLayer(args.embedding_dim, cp(attention), cp(feedForward), dropout=args.dropout).to(device)
    transformerEncoder = TransformerEncoder(encode_layer=encodeLayer, N=6).to(device)

    # 构建Decoder
    decodeLayer = DecodeLayer(args.embedding_dim, cp(attention), cp(attention), cp(feedForward), args.dropout).to(device)
    transformerDecoder = TransformerDecoder(decode_layer=decodeLayer, N=6).to(device)

    # 构建srd_embedding
    src_embedding = nn.Sequential(Embeddings(args.embedding_dim, enVocabLen), cp(position)).to(device)
    dst_embedding = nn.Sequential(Embeddings(args.embedding_dim, zhVocabLen), cp(position)).to(device)

    # 构建generator
    generator = Generator(decoder_dim=args.embedding_dim, vocab_len=zhVocabLen).to(device)

    # 构建transformer 机器翻译模型
    translateEn2Zh = TranslateEn2Zh(encoder=transformerEncoder, decoder=transformerDecoder, src_embedding=src_embedding,\
                                    dst_embedding=dst_embedding, generator=generator).to(device)

    # 随机初始化
    for param in translateEn2Zh.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform(param)

    # 开始训练
    for epoch in range(args.num_epochs):
        for batchIndex, (englishSeq, chineseSeq, lengths) in enumerate(trainDataLoader):
            englishSeq = englishSeq.to(device)
            chineseSeq = chineseSeq.to(device)
            src_mask = subsequent_mask(lengths)
            out = translateEn2Zh.forward(english_seq=englishSeq, english_mask=, chinese_seq=chineseSeq, chinese_mask=)




