# Transformer机器翻译

[toc]

## 组队信息

1.  姓名：孙汉武	学号：202028018670086	学院：网络空间安全学院
2. 姓名：

## 文件说明

1. 代码文件

   > 以下代码中标\*的表示适当参考了[知乎文章](https://zhuanlan.zhihu.com/p/118601295)，未标\*号均为自行实现

   1. dataloader.py: 数据加载器，实现Dataset类，用于加载数据，构造函数中可指定mode参数，根据mode参数的不同可以对应生成train, valid, test数据集
   2. utils.py: 工具函数集，包含一众工具函数如下(详细API请看代码注释)：
      1. aggreVocab:  汇总训练数据并生成词表
      2. clones*:  module克隆函数
      3. subsequent_mask*: 产生指定size的下三角矩阵
      4. make_std_mask: 产生encoder需要的标准mask
      5. attention*:  注意力计算函数，使用$Q^T\cdot K$的计算方式
      6. compute_bleu：计算机翻译文和参考译文之间的BLEU值
   3. model.py：实现Transformer及各个组件，并且是训练和测试的入口函数(详细API请见代码注释)
      1. TranlateEn2Zh*:  利用通用Encoder-Decoder框架实现的Transformer模型
      2. Generator*: 从Decoder生成的隐藏状态中输出一个词
      3. TransformerEncoder*: 编码器
      4. EncodeLayer*: 编码层，六个编码层组成一个编码器
      5. TransformerDecoder*:  解码器
      6. DeocdeLayer*:  解码层，六个解码层组成一个解码器
      7. SubLayer*: 通用子层，根据传输的功能函数不同实现不同的操作，包括self-attention， attention和feed_forward等。一个编码层包含一个self-attention子层和一个feed_forward子层；一个解码层包括一个self-atttention子层，一个attention子层和一个feed_forward子层。
      8. Embedding*: 词向量嵌入
      9. PositionalEncoding*:  位置编码
      10. main：实现入口函数，根据命令行参数指定模型参数和运行模式(train/test)

2. 数据文件

   > 语料数据不介绍

   1. vocab_en.npy：英文词表文件(`np.load`打开)
   2. vocab_zh.npy：中文词表文件(`np.load`打开)
   3. test_zh_translate.txt:  模型在test集上的翻译结果
   4. log: 模型运行日志文件
   5. translate_params.pkl：最好效果的模型参数文件(`model.load_state_dict(torch.load`可以加载)

## 运行参数

为了方便调参，定义了如下命令行参数，在运行时可以指定

```bash
positional arguments:
  mode                  train or test

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        minibatch size
  --lr LR               learning rate
  --num_epochs NUM_EPOCHS
                        number of epochs
  --embedding_dim EMBEDDING_DIM
                        number of word embedding
  --gpu GPU             GPU No, only support 1 or 2
  --head_num HEAD_NUM   Multi head number
  --hidden_num HIDDEN_NUM
                        hidden neural number
  --dropout DROPOUT     dropout rate
  --padding PADDING     padding length
  --model_path MODEL_PATH
                        model path
```

## 实验结果

1. 

