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

> 模型的翻译结果和运行日志可以在`test_zh_translate.txt`和`log`中查看，这里只展示一部分

### 翻译效果

```
原文: <START> and also , the senate &apos;s other worry , the appearance of singularity -- well , it was a singularity back then .
机翻译文: <START>还有，参议院，也是负面新闻的焦点，这是一个案例。
参考译文: <START>而且，参议院担心的另一件事，这个头衔听起来很古怪--哦，它曾经在那段时光听起来古怪。


原文: <START> you have not changed the story .
机翻译文: <START>你也没有改变。
参考译文: <START>故事的剧情依然如旧。


原文: <START> well , the things i constantly hear are : too many chemicals , pesticides , hormones , monoculture , we don &apos;t want giant fields of the same thing , that &apos;s wrong .
机翻译文: <START>我一直在听的东西是：许多化学，化学，化学，化学，我们不太熟悉，是哪个是非常不可能的，过大的问题，
参考译文: <START>我经常听到的理由是：（转基因食物）太多化学成分了，太多农药、荷尔蒙，单一化的生产，确实我们不希望数百亩的地都是生产同样的作物，那样


原文: <START> imagine you can have everybody make a small donation for one pixel .
机翻译文: <START>想象你可以给每一个人提供一个小的捐赠钞票。
参考译文: <START>想想你能让每个人捐出一个像素的钱。


原文: <START> except for around the two towers , there is 32 inches of steel paralleling the bridge .
机翻译文: <START>除了在这两个构式中，有32尊名为诈骗的史瓦西半径。
参考译文: <START>除了在两座塔周围，桥边有32英寸的钢索，平行于桥。
```

### BLEU指标

`translate_params.pkl`在以下超参数下训练：

+ lr=0.001
+ batch_size=128
+ embedding_dim=256
+ head_num=8
+ epoch_num=10

最终的test集上的BLEU值为**17.26%**



