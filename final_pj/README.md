# 代码运行和实验结果


## 环境配置
python == 3.8
torch == 1.10.0+cu111




## 下载数据集
**原始数据文件**:将oracle_fs.zip放在同一文件夹下进行解压

**获取预处理的增强数据**: 可以从https://drive.google.com/file/d/1-XdOr5YUCuq0r94JqqR5u3ooznwyP1KB/view?usp=sharing下载



## 生成增强数据

* 若要自动生成数据增强文件，请将文件放在oralce_fs文件夹下或修改198行path变量,将路径指向原始训练集。

* 12，192-195行是所有的超参数，block_num代表FFD变换的控制点数，offset代表FF D每个点的变换最大范围，num代表单个训练数据生成的增强图片数，shot即要数据增强的训练集是1，3或5 shot。

* 代码运行后会在path路径下生成对应超参数的文件夹。

    ```shell
    python FFD_augmentor.py
    ```

    

## 训练代码

```shell
# EASY for wideresnet
CUDA_LAUNCH_BLOCKING=1 python -u main_new.py --dataset-path ../oracle_fs --dataset oracle --n-shots 1 --mixup --model wideresnet --feature-maps 16 --skip-epochs 90 --epochs 100 --rotations --preprocessing "PEME" --sample-aug 10

# EASY for resnet18
CUDA_LAUNCH_BLOCKING=1 python -u main_new.py --dataset-path ../oracle_fs --dataset oracle --n-shots 1 --mixup --model Resnet18 --feature-maps 16 --skip-epochs 90 --epochs 100 --rotations --preprocessing "PEME" --sample-aug 10

#EASY for resnet20
CUDA_LAUNCH_BLOCKING=1 python -u main_new.py --dataset-path ../oracle_fs --dataset oracle --n-shots 1 --mixup --model Resnet20 --feature-maps 16 --skip-epochs 90 --epochs 100 --rotations --cosine --preprocessing "PEME" --sample-aug 10

#EASY for baseline
CUDA_LAUNCH_BLOCKING=1 python -u main_new.py --dataset-path ../oracle_fs --dataset oracle --n-shots 1 --model Resnet18 --feature-maps 16 --skip-epochs 90 --epochs 100


#test On ffd data
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python -u main_new.py --dataset-path FFD_train_11_block5_30_3_shot --dataset oracleAug --model wideresnet --n-shots 3 --mixup  --feature-maps 16 --skip-epochs 95 --epochs 100 --rotations --preprocessing PEME
```



## 训练结果

**With preprocessing: mixup+rotation+PEME**

|        | ResNet18 | ResNet20 | WideResnet | Resnet12 |
| ------ | -------- | -------- | ---------- | -------- |
| 1 shot | 55.17    | 54.67    | 53.77      | 58.46    |
| 3 shot | 79.45    | 82.53    | 84.81      | 84.91    |
| 5 shot | 90.34    | 90.58    | 92.30      | 91.30    |



**With no preprocessing**

|        | ResNet18 | ResNet20 | WideResnet | Resnet12 |
| ------ | -------- | -------- | ---------- | -------- |
| 1 shot | 47.00    | 49.26    | 49.91      | 62.10    |
| 3 shot | 78.71    | 78.29    | 81.79      | 88.34    |
| 5 shot | 89.36    | 91.55    | 92.43      | 92.42    |



| Num of augmented sample | N shot | Accuracy |
| ----------------------- | ------ | -------- |
| 0                       | 1      | 52.94    |
| 5                       | 1      | 53.39    |
| 10                      | 1      | 53.52    |
| 20                      | 1      | 53.40    |
| 40                      | 1      | 53.40    |



### Experiment with FFD+EASY



#### MAX OFFSET VALUE

epoch=100测试 Resnet18

| FFD MinMax | N shot | Accuracy |
| ---------- | ------ | -------- |
| no         | 1      | 51.56    |
| 3          | 1      | 68.31    |
| 5          | 1      | 73.14    |
| 6          | 1      | 74.51    |
| 7          | 1      | 69.24    |
| 9          | 1      | 76.15    |
| no         | 3      | 78.45    |
| 3          | 3      | 81.89    |
| 5          | 3      | 93.42    |
| 7          | 3      | 93.99    |
| no         | 5      | 89.71    |
| 3          | 5      | 94.02    |
| 5          | 5      | 95.40    |
| 7          | 5      | 94.92    |



1-shot 20个数据增强 不同 block_num和Minmax组合的实验(所有实验都做了2次取平均)
| MinMax/block_num | 3 | 5 |7 | 
| ---------- | ------ | -------- | -------- | 
| 7          |   74.0   |   75.76  | 75.88|
| 9          |   75.16    |    76.15 | 76.44 |
| 11         |    76.34   |   78.32  | 77.44 |
| 13         |   76.3    |  77.8   | 77.32 |
| 15         |    77.76   |   77.4  | 76.32 |
| 17         |    77.3   |     |
 19         |    77.21   |     |

 1-shot 的数据增强实验(所有实验都做了2次取平均)
| combination/n | 10 | 20  | 30  | 40  |
| ---------- | ------ | -------- | -------- | -------- | 
| 3 - 15         |    75.85   |  77.76   |78.33 | 78.59
| 5 - 11         |   75.92    |   78.32  | 78.90 | 78.2 |
| 7 -11          |   75.10   |   77.44  | 77.16 | 76.58|

1-shot 30 个数据增强  5 Block Num  11 Offset
| combination/lr | 0.1 | 0.05  | 0.01  | 0.005（过拟合）  |  0.001(收敛过慢)  |
| ---------- | ------ | -------- | -------- | -------- |  -------- | 
| 5-shot          |   95.82  |   96.31  | 96.47 | 89.08 | 87.05 | 



#### **FFD blocknum**

| Block Num | N Shot | Acc   |
| --------- | ------ | ----- |
| 9_Block3  | 1      | 75.16 |
| 9_block5  | 1      | 76.15 |
| 9_block7  | 1      | 76.44 |
| 11-block3 | 1      | 76.34 |
| 13-block3 | 1      | 76.35 |
| 15-block3 | 1      | 76.80 |





#### **best result**

| FFD                | Model      | N SHOT | ACC   |
| ------------------ | ---------- | ------ | ----- |
| 11-block5-30sample | wideresnet | 1      | 77.75 |
| 11-block5-30sample | resnet12   | 1      | 76.79 |
| 11-block5-30sample | wideresnet | 3      | 93.42 |
| 11-block5-30sample | resnet12   | 3      | 92.89 |
| 11-block5-30sample | wideresnet | 5      | 97.59 |
| 11-block5-30sample | resnet12   | 5      | 95.38 |

