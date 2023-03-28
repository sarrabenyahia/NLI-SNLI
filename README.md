# Natural Language Inference with Transformer-based Models
This repository contains the code and documentation for my work on natural language inference using transformer-based models. The goal of this project was to build a model that can classify the relationship between two sentences: entailment, contradiction, or neutral.

## Dataset  :pencil:
The SNLI (Stanford Natural Language Inference) dataset is a collection of sentence pairs that have been labeled with one of three categories: "entailment", "contradiction", or "neutral". The goal of the dataset is to provide a benchmark for natural language inference (NLI) models to classify the relationship between two sentences.

The SNLI dataset consists of 570,000 sentence pairs in total, split into 549,367 for training, 9,842 for validation, and 9,824 for testing. The sentences in each pair are drawn from a range of genres, including fiction, government reports, and popular magazines.

The creation of the SNLI dataset involved human annotation of the sentence pairs, with annotators determining whether the relationship between the two sentences was "entailment" (one sentence logically entails the other), "contradiction" (the two sentences cannot both be true), or "neutral" (there is no logical relationship between the two sentences).

## Approach :bar_chart:
I experimented with transformer-based models, and more specifically the [cross-encoder/nli-roberta-base](https://huggingface.co/cross-encoder/nli-roberta-base/blame/main/README.md). I fine-tuned pre-trained models on the SNLI dataset using the Hugging Face Transformers library. I also tried different hyperparameters and optimization strategies to improve the performance of my models.

## Results :outbox_tray:
My best model achieved an accuracy of 87.3% on the SNLI test. The model also outperformed the state-of-the-art models on the SNLI dataset. I also analyzed the errors made by the model and identified some common patterns.

## Credits :eyes:
This work was done as part of a collaborative project with [Pse1234](https://github.com/Pse1234) in this [repository](https://github.com/Pse1234/NLI).
You can find our collective report in this [pdf](https://github.com/Pse1234/NLI/blob/main/Natural_Language_Inference.pdf)

## References
Bowman, S. R., et al. (2015). A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP).
Williams, A., et al. (2018). A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT).
