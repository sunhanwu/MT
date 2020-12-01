import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class Data(Dataset):
    """
    机器翻译模型的数据集
    """
    def __init__(self, mode='train'):
        """
        构造函数
        :param mode: 指定是训练还是测试
        """
        assert mode in ['train', 'valid', 'test'], "mode must in ['train', 'valid', 'test']"
        self.mdoe = mode
        # 读取词表文件
        self.vocab_zh = np.load('../data/vocab_zh.npy').item()
        self.vocab_en = np.load('../data/vocab_en.npy').item()
        # 读取数据文件
        self.data_zh = self._getData('../data/{}.zh.tok'.format(mode), language='zh')
        self.data_en = self._getData('../data/{}.en.tok'.format(mode), language='en')

    def _getData(self, filepath, language):
        """
        读取数据文件，并将其转化为词表对应的index
        :param filePath: 数据文件的路径
        :return: 二维列表，第一维表示行的数目，第二维是每个单词
        """
        with open(filepath, 'r') as f:
            data = [x.strip() for x in f.readlines()]
        # 将数据通过词表转换为对应的索引
        dataSet = []
        for line in data:
            # 根据指定的语言将数据
            if language == 'en':
                dataSet.append([self.vocab_en.get(x, 2) for x in line.split(' ')])
            elif language == 'zh':
                dataSet.append([self.vocab_zh.get(x, 2) for x in line.split(' ')])
        return dataSet

    def getVocabZhLen(self):
        return len(self.vocab_zh)

    def getVocabEnLen(self):
        return len(self.vocab_en)

    def __getitem__(self, index):
        """
        按照index返回对应的en 和 zh
        :param index: 索引
        :return: index对应的训练数据X和对应的label
        """
        en = torch.tensor(self.data_en[index])
        zh = torch.tensor(self.data_zh[index])
        return en, zh

    def __len__(self):
        """
        返回所有的数据长度
        :return:
        """
        return len(self.data_en)
