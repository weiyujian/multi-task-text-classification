#多任务分类算法

## Requirements

- Python 2.7
- Tensorflow > 0.12
- Numpy

## Training
CUDA_VISIBLE_DEVICES=0 python train.py --model_version=xxx

## Evaluating

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --checkpoint_dir=./runs/model_version/checkpoints/
```



## 说明
适用于一个句子需要做多个分类的场景，类比到图像就是，手写数字识别中，一个图片既需要识别数字，可能还需要识别是否是奇数
在nlp中，一个问句在任务a里需要做个分类，在任务b里也需要做个分类
算法支持不定任务个数,即3个任务可以运行，4个也能运行

#语料格式
问句 \t 任务1_标签 \t 任务2_标签 ... \t 任务n_标签



