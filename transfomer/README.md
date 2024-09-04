# Full-Scratch Transfomer(IMDB Classification)

## Description

## Dataset

Transfomer is trained by [IMDB Dataset](https://huggingface.co/datasets/stanfordnlp/imdb).

> Large Movie Review Dataset. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well.

This dataset label is binary 0/1,  0: negative and 1: positive.
Because this dataset is not imbalance, random model accuracy has 50%.

## Model

We use Transfomer Encoder. Model Summary is the following.

```txt
=========================================================================================================
Layer (type:depth-idx)                                  Output Shape              Param #
=========================================================================================================
TransformerForSequenceClassification                    [128, 1]                  --
├─TransformerEncoder: 1-1                               [128, 512, 128]           --
│    └─Embeddings: 2-1                                  [128, 512, 128]           --
│    │    └─Embedding: 3-1                              [128, 512, 128]           1,280,384
│    │    └─Embedding: 3-2                              [1, 512, 128]             65,664
│    │    └─LayerNorm: 3-3                              [128, 512, 128]           256
│    │    └─Dropout: 3-4                                [128, 512, 128]           --
│    └─ModuleList: 2-2                                  --                        --
│    │    └─TransformerEncoderBlock: 3-5                [128, 512, 128]           --
│    │    │    └─LayerNorm: 4-1                         [128, 512, 128]           256
│    │    │    └─MultiHeadSelfAttention: 4-2            [128, 512, 128]           66,048
│    │    │    └─LayerNorm: 4-3                         [128, 512, 128]           256
│    │    │    └─PointwiseFeedForward: 4-4              [128, 512, 128]           131,712
├─Classifier: 1-2                                       [128, 1]                  --
│    └─Sequential: 2-3                                  [128, 1]                  --
│    │    └─Dropout: 3-6                                [128, 128]                --
│    │    └─Linear: 3-7                                 [128, 1]                  129
=========================================================================================================
Total params: 1,544,705
Trainable params: 1,544,705
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 189.38
=========================================================================================================
Input size (MB): 0.52
Forward/backward pass size (MB): 872.94
Params size (MB): 6.18
Estimated Total Size (MB): 879.64
=========================================================================================================
```

### Results

Despite selecting hyperparameters somewhat arbitrarily, the accuracy was very high, ranging from 0.85 to 0.86. Although I stopped at 10 epochs, there were still signs of improvement.

<img src=./img/fig.png />

This accuracy is close to DistlliBERT.

- <https://www.kaggle.com/code/omarallam22/movie-sentiment-analysis#Modeling>
