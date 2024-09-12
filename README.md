# ML-Sandbox

This repository provides implementations and experiments of various ML models.

## Directory Structure

```sh
.
├── sequential_recommendation/ # Sequential Recommendation
└── transfomer/ # Sentiment Analysis
```

## Problem Types

This repository deals with the following problem types:

1. Sequential Recommendation
1. Sentiment Analysis

### Sequential Recommendation

According to <https://paperswithcode.com/task/sequential-recommendation>, Sequential Recommendation is the following.

> Sequential recommendation is a sophisticated approach to providing personalized suggestions by analyzing users' historical interactions in a sequential manner. Unlike traditional recommendation systems, which consider items in isolation, sequential recommendation takes into account the temporal order of user actions. This method is particularly valuable in domains where the sequence of events matters, such as streaming services, e-commerce platforms, and social media.

This repository has the following models.

- SASRec: [Self-Attentive Sequential Recommendation](<https://arxiv.org/abs/1808.09781>)
- gSASRec: [gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling](https://arxiv.org/abs/2308.07192)

For more information on Sequential Recommendation, please refer to the [sequential_recommendation directory](https://github.com/haru-256/ml-sandbox/tree/main/sequential_recommendation).

### Sentiment Analysis

According to <https://en.wikipedia.org/wiki/Sentiment_analysis>, Sentiment Analysis is the following.

> Sentiment analysis (also known as opinion mining or emotion AI) is the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. Sentiment analysis is widely applied to voice of the customer materials such as reviews and survey responses, online and social media, and healthcare materials for applications that range from marketing to customer service to clinical medicine. With the rise of deep language models, such as RoBERTa, also more difficult data domains can be analyzed, e.g., news texts where authors typically express their opinion/sentiment less explicitly.[1]

This repository has the following models.

- Transformer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

For more information on Sentiment Analysis, please refer to the [transfomer directory](https://github.com/haru-256/ml-sandbox/tree/main/transfomer).
