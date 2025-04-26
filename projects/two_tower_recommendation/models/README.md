# Models

## Basic Two-Tower Model

```txt
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
TwoTower                                 [1024, 128]               --
├─UserEmbedding: 1-1                     [1024, 128]               --
│    └─IdEmbedding: 2-1                  [1024, 128]               --
│    │    └─Embedding: 3-1               [1024, 128]               23,323,904
│    └─Linear: 2-2                       [1024, 128]               16,512
├─ItemEmbedding: 1-2                     [1024, 128]               --
│    └─IdEmbedding: 2-3                  [1024, 128]               --
│    │    └─Embedding: 3-2               [1024, 128]               10,022,656
│    └─Linear: 2-4                       [1024, 128]               16,512
├─ItemEmbedding: 1-3                     [1024, 128]               (recursive)
│    └─IdEmbedding: 2-5                  [1024, 128]               (recursive)
│    │    └─Embedding: 3-3               [1024, 128]               (recursive)
│    └─Linear: 2-6                       [1024, 128]               (recursive)
==========================================================================================
Total params: 33,379,584
Trainable params: 33,379,584
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 44.46
==========================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 6.29
Params size (MB): 133.52
Estimated Total Size (MB): 139.83
==========================================================================================
```
