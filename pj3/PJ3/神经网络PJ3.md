# 神经网络PJ3

赵心怡 19307110452

2022.5.29

## 1. noc图像字幕标注介绍

当前的image captioning模型各方面表现都已不错，但最大的问题是，它通常建立在image-caption对上，导致了仅仅能够捕捉领域内的目标，且无法拓展到现实中很多novel scene和out-of-domain的图片上

为了生成novel objects，目前来说需要存在两个困难：

1.如何促进词汇拓展
2.如何学习网络，使得能够很好的融入识别到的目标到caption中

在查找了关于图像字幕生成的相关论文，可以发现目前的主要研究方向是寻找基于模板的字幕模型，即NeuralBabyTalk、多任务模型如DCC，和新颖的采样算法如Statistical machine translation。

我尝试了NeuralBabyTalk模型的效果，由于服务器中间一直连不上，我另外在别的服务器上尝试训练了预训练+微调的Blip模型，并比较了两种模型的效果和优劣，讨论了得分波动和差异较大的原因。

## 2.NeuralBabyTalk

#### 方法概述

NeuralBabyTalk的贡献为提出了一种图像描述框架，以图中检测到的实体为基础而生成自然语言。并使得传统的槽填充方法（通常更好地基于图像）与现代神经描述方法（通常更加自然）协调一致。主要框架是生成一个句子模板，其中预留的槽会与图中区域相关联，并且这些槽由检测出来的物体标签来填充。

![image-20220526220642457](C:\Users\13374\AppData\Roaming\Typora\typora-user-images\image-20220526220642457.png)

公式可以分成两个级联的目标：1. 最大化生成句子“模板”的概率 2. 最大化依据grounding区域和目标识别信息得到的visual words的概率；

使用RNN生成Caption的模板。此RNN由LSTM层组成，CNN输出的feature maps作为其输入。

使用目标检测框架在grounding区域上，可以识别区域内的物体类别，例如，“狗”。对词语进行变换使其适合当前文本上下文，比如单复数、形态等。单复数用二分类器，fine-grained用多分类器来进行。

#### 具体实现：

**Detection model**: Faster R-CNN

**Region feature:** ResNet101

**预处理**： 将576\*576的大小的图片随机裁切为512\*512作为CNN的输入，对coco的数据集中caption的长度缩减，不超过16个词。

**Language Model:** attention model 两层的LSTM

**LSTM hidden layer:** 1024

**attention layer hidden size:** 512

**input word embedding**: 512

**Batch size：**20

**优化器**： Adam

**学习率：**5e-4,每三个epoch衰减为原来的0.8倍

**Epoch:** 10



**因为139机器进程过多，一直会崩溃，在训练过程中还重启了两次，所以真正成功训练的epoch数有点小，如果允许本来计划跑80个epoch 的。**

#### DCC split annotations

DCC annotations split 是 MSCOCO训练集的一个子集(表示为保留的MSCOCO训练集)，它排除了至少描述八个MSCOCO对象中的一个的所有图像-句子对。为了确保被排除的对象至少与一些被包含的对象相似，选择了以下词：“瓶子”、“公共汽车”、“沙发”、“微波炉”、“披萨”、“球拍”、“手提箱”和“斑马”。

随机选择50%的MSCOCO验证集进行验证，并留出剩下的50%进行测试。使用验证集来确定所有模型超参数，并在测试集上显示所有结果。根据MSCOCO数据集中提供的五个实际标题注释来标记每个图像中的视觉概念。如果任何一个实际标题提到一个物体，相应的图像被认为是该物体的证明例子。

进行数据清洗后，删除了captions_split_set_bottle_val_val_train2014的数据，我们只需要做out of domain的评价。

**Metric：**在实验中使用了BLEU4  、Meteor  、CIDEr  、SPICE  、F1分数5种指标对模型进行评价。

#### 效果

NBT在各个指标上的平均得分，所有指标都以百分数的形式表示。

|        |  bus |   bottle | couch | microwave | pizza |   racket | suitcase |    zebra |
| :----- | ---: | -------: | ----: | --------: | ----: | -------: | -------: | -------: |
| Bleu_4 | 23.5 |     25.5 |  28.4 |  **30.4** |  21.4 |     27.3 |     16.6 |     23.2 |
| METEOR | 20.7 |     21.7 |  23.1 |      23.0 |  21.8 | **25.7** |     18.8 |     23.8 |
| CIDEr  | 47.3 | **77.0** |  64.7 |      48.0 |  49.4 |     30.1 |     55.7 |     35.9 |
| SPICE  | 14.6 |     14.7 |  14.9 |      15.1 |  15.3 | **18.3** |     12.0 |     18.0 |
| F1     | 78.3 |     20.8 |  39.4 |      63.7 |  85.8 |     24.3 |     63.7 | **92.2** |

从结果上看到，模型平均的CIDER等其他值不是很高，但是它的F1却很高。猜测这样的原因是模型的字幕标注表现一般，因为NeuralBabyTalk的目标检测框架中对于新颖词汇的关注度很高，而忽视了其他词语的匹配度，所以TP的比例更高，但总体文本语义相似度较低。

使用matplotlib对数据进行可视化分析。

前三个epoch的loss曲线：

<img src="C:\Users\13374\AppData\Roaming\Typora\typora-user-images\image-20220529161913748.png" alt="image-20220529161913748" style="zoom: 67%;" />

cider曲线：

<img src="C:\Users\13374\AppData\Roaming\Typora\typora-user-images\image-20220529174352462.png" alt="image-20220529174352462" style="zoom: 80%;" />

训练结果和paper中给出的结果比较：

|              | BLEU4 | METEOR | CIDEr | SPICE |
| ------------ | ----- | ------ | ----- | ----- |
| My Model     | 26.9  | 22.3   | 74.8  | 15.4  |
| Paper Result | 30.8  | 25.5   | 95.2  | 18.4  |

从结果中可以看到我们的训练模型和paper的结果还有一定差距首先是机器的性能限制，我们的batch-size设置只有20，训练速度较慢，从训练的loss看也能看出模型还未收敛，epoch设置太小，但是因为机器一直崩溃的原因，没有办法进行更多的训练的测试，而且从F1的很高的结果来看NBT的模型对于新颖词有很高的记忆能力。

## 3. BLIP model

#### **方法概述:**

 BLIP 是一个统一的视觉语言预训练（vision-language pre-training, VLP）框架，从有噪声的图像文本对中学习。

![](D:\童年在人间我的fdu\大三下\神网\pj3\blip.png)

研究者将一个视觉 transformer 用作图像编码器，该编码器将输入图像分解为 patch，然后将这些 patch 编码为序列嵌入，并使用一个额外的[CLS] token 表征全局图像特征。

研究者在预训练过程中共同优化了三个目标，分别是两个基于理解的目标和一个基于生成的目标。每个图像文本对需要三个前向传播通过文本 transformer，其中激活不同的功能以计算图像文本对比损失（ITC）图像文本匹配损失（ITM）和语言建模损失（LM）。

编码器使用双向自注意力为当前输入 token 构建表征，同时解码器使用因果自注意力预测接下来的 token。

作者同时提出一种提升文本语料库质量的新方法——CapFilt。引入了一个为给定 web 图像生成标注的标注器（captioner）和一个是消除有噪声图像文本对的过滤器（filter）。![image-20220529175939736](C:\Users\13374\AppData\Roaming\Typora\typora-user-images\image-20220529175939736.png)

研究者将过滤后的图像文本对于人工注释对相结合以生成一个新的数据集，并用它预训练新模型。

可以发现，该模型的主要强大在于预训练和数据增强过程，并在之后的数据集上进行进一步微调。

#### **具体实现**：

在训练中我们使用BLIP w/ ViT-B and CapFilt-L模型作为初始训练结果，选择基础大小的Vit模型，在三张3090卡上进行分布式训练。

为了防止服务器在运行过程中突然断开，我们使用tmux命令来将会话与窗口可以"解绑"，即使窗口突然关闭，进程也能继续进行。

具体的数据预处理过程和环境配置在Readme中介绍。

**batch size：**6

**learning rate：**1e-5初始化的cosine lr schedule

**image size**:384

**weight decay**: 0.05

**优化器：**AdamW

**epoch**：10

原版代码中没有计算F1的过程， 而且evaluation过程也只能读取一个文件，因此我们重新写了eval_F1文件将F1分数的逻辑和其他，可以读取所有类别的文件对每个进行评估。

F1和tp,f,fnp的关系如下:

![image-20220530144639855](C:\Users\13374\AppData\Roaming\Typora\typora-user-images\image-20220530144639855.png)

#### 结果：

训练结束后，我们用beam size=3(默认)对模型进行评估，在各个指标的平均得分如下表所示

|        |    bottle |      bus |    couch | microwave | pizza | racket | suitcase | zebra |
| :----- | --------: | -------: | -------: | --------: | ----: | -----: | -------: | ----: |
| Bleu_4 |      32.1 |     27.5 | **33.6** |      32.7 |  25.8 |   29.8 |     28.3 |  18.1 |
| METEOR |      26.6 |       24 | **28.5** |      27.1 |  23.2 |   27.4 |     24.7 |  19.3 |
| CIDEr  | **104.4** |       71 |     85.1 |      76.3 |  67.5 |   51.8 |     82.9 |  39.7 |
| SPICE  |      18.9 |     15.7 | **21.2** |      16.7 |  16.5 |   19.6 |     17.4 |  11.4 |
| F1     |         0 | **69.8** |     3.25 |      15.5 |   4.1 |   14.0 |        0 | 32.76 |

训练结果的平均得分和NBT进行对比：

|      | Bleu4    | Meteor   | CIDEr     | SPICE    |
| ---- | -------- | -------- | --------- | -------- |
| BLIP | **31.7** | **26.4** | **104.4** | **18.9** |
| NBT* | 30.8     | 25.5     | 95.2      | 18.4     |

可以看见，BLIP全方位指标上都明显优于NeuralBabyTalk论文中的结果。但是F1却很小，远不如NBT，甚至在有的组别上F1为0。而且不同组别的分数波动变化很大。



## 4.讨论

对于BLIP的F1很小的原因， 我猜测是BLIP的模型虽然加入了大量模糊图像的预训练，但是其模型设计并没有对Novel Object强调，导致结果虽然总体很优秀，但是对novel object的识别能力很有限。

我们以含有bottle词语的validation图片为例，在测试中BLIP模型对BOTTLE测试的F1 score为0，我们为了研究为什么会发生这种情况，抽取一些图片进行测试，观察blip给出的结果

image_id: 226171

<img src="D:\童年在人间我的fdu\大三下\神网\pj3\coco\images\COCO_val2014_000000226171.jpg" alt="COCO_val2014_000000226171" style="zoom: 67%;" />

| **BLIP** | Desktop computer sitting on top of a wooden desk.            |
| -------- | ------------------------------------------------------------ |
| GT       | A computer monitor sitting above a computer keyboard.        |
|          | A sunny office desk with computer, laptop and phone          |
|          | a bottle a cup a keyboard a laptop and a monitor             |
|          | A monitor, keyboard, coffee cup, and plastic bottle sit on a table. |
|          | A desk with a computer, **computer keyboard,** mug, empty **plastic bottle,** **red pen**, and a laptop. |

可以看到模型主要关注图片的主体部分，而忽略了一些零碎的细节，导致识别出的物体个数很少，只看到了computer和desk，但是对识别出的物体进行了细致的描述，包括材质和性能。猜测产生这种结果的原因是预训练的图片往往物体较少，但对于物体的特征描述较为复杂，如果有时间调整预训练的数据集，那么模型在NOC物体上的字幕标注还会更加准确。

在另一种情况中，图片的物体较少，但是也没有识别出bottle

image_id:343248

<img src="D:\童年在人间我的fdu\大三下\神网\pj3\coco\images\COCO_val2014_000000343248.jpg" style="zoom:80%;" />

| **BLIP** | a man pouring a glass of wine into a wine glass              |
| -------- | ------------------------------------------------------------ |
| GT       | Someone holding a **bottle** of wine and a glass in his hands, with several other bottles and containers on the table behind him |
|          | A man prepares to pour something from a **bottle** into a glass. |
|          | The man is pouring wine into the glass.                      |
|          | A man pouring a drink in to a thin tall glass.               |
|          | A man pours wine from the **bottle** to the glass.           |

图中虽然有瓶子，但是模型用更具体的glass of wine来代替了。我觉得这种改动没有什么问题，其实是识别出酒瓶的，但是F1会很低。



对于CIDEr得分在某些数据上波动很大的原因我们选择了含有zebra的图片来测试，测试结果显示zebra的CIDEr得分只有39，我们随机抽取图片来比较和gt的区别。

image_id:1818

<img src="D:\童年在人间我的fdu\大三下\神网\pj3\coco\images\COCO_val2014_000000001818.jpg" style="zoom:67%;" />

| **BLIP** | a baby zebra standing next to an adult zebra                 |
| -------- | ------------------------------------------------------------ |
| **GT**   | A zebra in the grass who is **cleaning** himself.            |
|          | A baby giraffe **drinking** milk from it's mother in a field. |
|          | A baby zebra is **suckling** milk from its mother.           |
|          | a baby zebra **nursing** from an adult zebra                 |
|          | Baby zebra **sucking** milk from its mothers teat.           |

zebra是这几个词中的唯一动物，和场景和其他的物体可能有许多的复杂交互描述，而模型很难概括这种复杂的动作。Ground truth中的行为的描述包括了复杂的动作如suck, nurse等，但是模型只能用stand来笼统概括。词语上概括的不到位导致词语匹配得分较低。



## 5.总结

通过两种模型的测试，我们发现NBT模型的对于NovelObject的效果更好，但是在总体的得分，尤其是CIDEr得分上远不如BLIP。BLIP使用了多模态方法提升模型效果，但它在单一模块上的表现不是很理想，如果可以用BLIP的预训练模型来从头训练COCO caption效果会不如其他模型。

经过微调，最终我们模型的CIDEr得分最高达到了104.4%。





## **运行环境**

NeuralBabyTalk: python 3.7 +torch0.4.1port2, GTX1080+cuda11.0

BLIP: python3.8 + torch1.10.0, GTX3090+cuda11.3

具体库的配置和代码复现详见README.md



## Reference

[1].Lu, Jiasen,Yang, Jianwei,Batra, Dhruv,Parikh, Devi.Neural Baby Talk.2018 *CoRR* , Vol. abs/1803.09845

[2].Li, Junnan, Li, Dongxu ,Xiong, Caiming, Hoi, Steven C. H. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation.2022.*CoRR* , Vol. abs/2201.12086

[3].Hendricks, Lisa Anne, Venugopalan, Subhashini, Rohrbach, Marcus, Mooney, Raymond J. , Saenko, Kate, Darrell, Trevor .Deep Compositional Captioning: Describing Novel Object Categories without Paired Training Data.2015 .*CoRR* , Vol. abs/1511.05284
