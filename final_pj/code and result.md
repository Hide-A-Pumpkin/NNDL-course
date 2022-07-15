# Code for final pj

#### 虚拟创建并启动环境

```shell
source /home/zxy/anaconda3/bin/activate
conda activate blip
       
```

#### Oracle_fs运行代码：

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
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1,2,3 python -u main_new.py --dataset-path FFD_train_11_block5_30_3_shot --dataset oracleAug --model wideresnet --n-shots 3 --mixup  --feature-maps 16 --skip-epochs 95 --epochs 100 --rotations --preprocessing PEME
```



#### HWOBC代码

```shell
# EASY for resnet18
CUDA_LAUNCH_BLOCKING=1 python -u main_new.py --dataset-path ../HWOBC_new --dataset hwobc --n-shots 1 --mixup --model Resnet18 --feature-maps 16 --skip-epochs 90 --epochs 100 --rotations --preprocessing "PEME"

# EASY with FFD
CUDA_LAUNCH_BLOCKING=1 python -u main_new.py --dataset-path ../HWOBC_FFD/HWOBC_FFD_train_11_block5_30_1_shot --dataset hwobcAug --n-shots 1 --mixup --model Resnet18 --feature-maps 16 --skip-epochs 90 --epochs 100 --rotations --preprocessing "PEME"

# ResNet18
CUDA_LAUNCH_BLOCKING=1 python -u baseline_res.py --dataset-path ../HWOBC_new --dataset hwobc --n-shots 1 --model Resnet18 --feature-maps 16 --skip-epochs 90 --epochs 100
```



| Model    | Baseline result |
| -------- | --------------- |
| ResNet18 | 15.33           |
| ResNet12 | 9.6             |
| ResNet20 | 4.3             |
| ResNet50 | 12.5            |





#### 部分结果：

**EASY With preprocessing: mixup+rotation+PEME**

|        | ResNet18 | ResNet20 | WideResnet | Resnet12 |
| ------ | -------- | -------- | ---------- | -------- |
| 1 shot | 55.17    | 54.67    | 53.77      | 58.46    |
| 3 shot | 79.45    | 82.53    | 84.81      | 84.91    |
| 5 shot | 90.34    | 90.58    | 92.30      | 91.30    |



**EASY With no preprocessing**

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



**FFD blocknum**

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
| 15-block3-30sample | wideresnet | 1      | 71.70 |
| 15-block3-30sample | resnet12   | 1      | 69.56 |
| 11-block5-30sample | wideresnet | 1      | 77.75 |
| 11-block5-30sample | resnet12   | 1      | 76.79 |
| 11-block5-30sample | wideresnet | 3      | 93.42 |
| 11-block5-30sample | resnet12   | 3      | 92.89 |
| 11-block5-30sample | wideresnet | 5      | 97.59 |
| 11-block5-30sample | resnet12   | 5      | 95.38 |







