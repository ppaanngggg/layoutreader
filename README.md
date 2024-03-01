# LayoutReader

<p align="center">
        🤗 <a href="https://huggingface.co/hantian/layoutreader">Hugging Face</a>
</p>

## Why this repo?

The original [LayoutReader](https://github.com/microsoft/unilm/tree/master/layoutreader) is published by Microsoft Research. It is based on `LayoutLM`, and use a `seq2seq` architecture to predict the reading order of the words in a document. There are several problems with the original repo:
1. Because it doesn't use `transformers`, there are lots of experiments in the code, and the code is not well-organized. It's hard to train and deploy.
2. `seq2seq` is too slow in production, I want to get the all predictions in one pass.
3. The [pre-trained model]()'s input is English word-level, but it's not the real case. The real inputs should be the spans extracted by PDF parser or OCR.
4. I want a multilingual model. I notice only use the bbox is only a little bit worse than bbox+text, so I want to train a model only use bbox, ignore the text.

## What I did?

1. Use `LayoutLMv3ForTokenClassification` of `transformers` to train and eval.
2. Offer a script turn the original word-level dataset into span-level dataset.
3. Use a better post-processor to avoid duplicate predictions.
4. Offer a docker image with API service.

## Dataset

The original dataset can download from [ReadingBank](https://layoutlm.blob.core.windows.net/readingbank/dataset/ReadingBank.zip?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D). More details can be found in the original [repo](https://aka.ms/readingbank). 

### Build Dataset

```bash
python tools.py cache-dataset-spans --help
```

### Train

```bash
bash train.sh
```

### Eval

```bash
python eval.py --help
```

## Span-Level Results

1. `shuf` means whether the input order is shuffled.
2. `BlEU Idx` is the BLEU score of predicted tokens' orders.
3. `BLEU Token` is the BLEU score of final merged text.

I only train the `layout only` model. And test on the span-level dataset. So the `Heuristic Method` result is quite different from the original word-level result. I mainly focus on the `BLEU Token`, it's only 0.4 lower than the original word-level result. But the speed is much faster.

> only use the first part of test file

| Method                     | shuf | BLEU Idx | BLEU Token |
|----------------------------|------|----------|------------|
| Heuristic Method           | no   | 44.4     | 70.7       |
| LayoutReader (layout only) | no   | 95.3     | 97.8       |
| LayoutReader (layout only) | yes  | 95.0     | 97.6       |

## Word-Level Results

### My eval script

The `layout only` model is trained by myself using the original codes, and the `public model` is the pre-trained model. The `layout only` is nearly as good as the `public model`, and the `shuf` only has a little effect on the results.

> only use the first part of test file.

| Method                      | shuf | BLEU Idx | BLEU Token |
|-----------------------------|------|----------|------------|
| Heuristic Method            | no   | 78.3     | 79.4       |
| LayoutReader (layout only)  | no   | 98.0     | 98.2       |
| LayoutReader (layout only)  | yes  | 97.8     | 98.0       |
| LayoutReader (public model) | no   | 98.0     | 98.3       |

### Old eval script (copy from original paper)

* Evaluation results of the LayoutReader on the reading order detection task, where the source-side of training/testing
  data is in the left-to-right and top-to-bottom order

| Method                     | Encoder                | BLEU   | ARD  |
|----------------------------|------------------------|--------|------|
| Heuristic Method           | -                      | 0.6972 | 8.46 |
| LayoutReader (layout only) | LayoutLM (layout only) | 0.9732 | 2.31 |
| LayoutReader               | LayoutLM               | 0.9819 | 1.75 |

* Input order study with left-to-right and top-to-bottom inputs in evaluation, where r is the proportion of
  shuffled samples in training.

| Method                     | BLEU   | BLEU   | BLEU   | ARD    | ARD   | ARD  |
|----------------------------|--------|--------|--------|--------|-------|------|
|                            | r=100% | r=50%  | r=0%   | r=100% | r=50% | r=0% |
| LayoutReader (layout only) | 0.9701 | 0.9729 | 0.9732 | 2.85   | 2.61  | 2.31 |
| LayoutReader               | 0.9765 | 0.9788 | 0.9819 | 2.50   | 2.24  | 1.75 |

* Input order study with token-shuffled inputs in evaluation, where r is the proportion of shuffled samples in training.

| Method                     | BLEU   | BLEU   | BLEU   | ARD    | ARD   | ARD    |
|----------------------------|--------|--------|--------|--------|-------|--------|
|                            | r=100% | r=50%  | r=0%   | r=100% | r=50% | r=0%   |
| LayoutReader (layout only) | 0.9718 | 0.9714 | 0.1331 | 2.72   | 2.82  | 105.40 |
| LayoutReader               | 0.9772 | 0.9770 | 0.1783 | 2.48   | 2.46  | 72.94  |
