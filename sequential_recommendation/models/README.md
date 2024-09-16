# Models

## SASRec

- unofficial reoisitory: https://github.com/pmixer/SASRec.pytorch

### Description

SASRec is a self-attentive sequential recommendation model proposed in the paper [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781). It is designed to effectively capture sequential patterns in recommendation tasks.

Use SASRec for self-attentive sequential recommendation. For more references, check out the [SASRec: Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) paper.

The following is the model architecture used in this repository.

```txt
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
SASRecModule                                       [1024, 50, 128]           --
├─SASRec: 1-1                                      [1024, 50, 128]           --
│    └─TransformerEmbeddings: 2-1                  [1024, 50, 128]           --
│    │    └─IdEmbeddings: 3-1                      [1024, 50, 128]           --
│    │    │    └─Embedding: 4-1                    [1024, 50, 128]           10,704,896
│    │    └─Embedding: 3-2                         [1, 50, 128]              6,400
│    │    └─LayerNorm: 3-3                         [1024, 50, 128]           256
│    │    └─Dropout: 3-4                           [1024, 50, 128]           --
│    └─ModuleList: 2-2                             --                        --
│    │    └─TransformerEncoderBlock: 3-5           [1024, 50, 128]           --
│    │    │    └─LayerNorm: 4-2                    [1024, 50, 128]           256
│    │    │    └─MultiheadAttention: 4-3           [1024, 50, 128]           66,048
│    │    │    └─LayerNorm: 4-4                    [1024, 50, 128]           256
│    │    │    └─PointwiseFeedForward: 4-5         [1024, 50, 128]           131,712
│    │    └─TransformerEncoderBlock: 3-6           [1024, 50, 128]           --
│    │    │    └─LayerNorm: 4-6                    [1024, 50, 128]           256
│    │    │    └─MultiheadAttention: 4-7           [1024, 50, 128]           66,048
│    │    │    └─LayerNorm: 4-8                    [1024, 50, 128]           256
│    │    │    └─PointwiseFeedForward: 4-9         [1024, 50, 128]           131,712
│    │    └─TransformerEncoderBlock: 3-7           [1024, 50, 128]           --
│    │    │    └─LayerNorm: 4-10                   [1024, 50, 128]           256
│    │    │    └─MultiheadAttention: 4-11          [1024, 50, 128]           66,048
│    │    │    └─LayerNorm: 4-12                   [1024, 50, 128]           256
│    │    │    └─PointwiseFeedForward: 4-13        [1024, 50, 128]           131,712
│    └─TransformerEmbeddings: 2-3                  --                        (recursive)
│    │    └─IdEmbeddings: 3-8                      [1024, 1, 128]            (recursive)
│    │    │    └─Embedding: 4-14                   [1024, 1, 128]            (recursive)
│    │    └─IdEmbeddings: 3-9                      [1024, 1, 128]            (recursive)
│    │    │    └─Embedding: 4-15                   [1024, 1, 128]            (recursive)
====================================================================================================
Total params: 11,306,368
Trainable params: 11,306,368
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 33.29
====================================================================================================
Input size (MB): 0.43
Forward/backward pass size (MB): 1208.01
Params size (MB): 44.43
Estimated Total Size (MB): 1252.87
====================================================================================================
```

### gSASRec

- official repository: https://github.com/asash/gSASRec-pytorch

### CAFE

- official repository: https://github.com/JiachengLi1995/CAFE

## Results

Here are the results of the model:

| metrics | train  | val |
| ------------- | ------------- | ------------- |
| loss | 0.4142 | 0.4634 |
| accuracy | 0.8134 | 0.7985 |

These metrics were calculated per epoch.

Here are the loss and accuracy plots for the training and validation sets:

| metrics | train  | val |
| ------------- | ------------- | ------------- |
| loss | ![train_loss](../img/train_loss.png) | ![val_loss](../img/val_loss.png) |
| accuracy | ![train_accuracy](../img/train_accuracy.png) | ![val_accuracy](../img/val_accuracy.png) |
