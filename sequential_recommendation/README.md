# Sequential Recommendation

## Dataset

The dataset used in this project is the [Amazon Reviews 2023, Video Games](https://amazon-reviews-2023.github.io/) dataset. It contains a collection of reviews for video games from the year 2023. This dataset is a valuable resource for training and evaluating recommendation models, such as SASRec. It provides a diverse range of user preferences and opinions, allowing for comprehensive analysis and accurate recommendations.

## Models

This repository provides the following mddels.

1. SASRec: [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
1. gSASRec: [gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling](https://arxiv.org/abs/2308.07192)
1. CAFE: [Coarse-to-Fine Sparse Sequential Recommendation](https://arxiv.org/abs/2204.01839)

The concrete model architecture is [here](https://github.com/haru-256/ml-sandbox/tree/main/sequential_recommendation/models)

## Usage

To train the model, follow these steps:

1. Clone the repository:

```sh
git clone https://github.com/haru-256/ml-sandbox.git
```

2. Navigate to the Transformer directory:

```sh
cd ml-sandbox/transfomer
```

3. Install the required dependencies:

```sh
uv sync
```

4. Run the training command:

```sh
make train
```

This command will initiate the training process and start training the Transformer model using the IMDB Dataset. The training progress and accuracy will be displayed in the console.

Note: Make sure you have [uv](https://github.com/astral-sh/uv) and Make installed on your system before running the above commands.

## References

- [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- [gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling](https://arxiv.org/abs/2308.07192)
- [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)
