# Sentence Embeddings in NLI with Iterative Refinement Encoders

Aarne Talman, Anssi Yli-Jyrä and Jörg Tiedemann. 2019. [Sentence Embeddings in NLI with Iterative Refinement Encoders](https://www.cambridge.org/core/journals/natural-language-engineering/article/sentence-embeddings-in-nli-with-iterative-refinement-encoders/AC811644D52446E414333B20FEACE00F). Natural Language Engineering 25 (4). 467-482. \[[preprint](https://arxiv.org/pdf/1808.08762.pdf)\]

**Abstract:** Sentence-level representations are necessary for various NLP tasks. Recurrent neural networks have proven to be very effective in learning distributed representations and can be trained efficiently on natural language inference tasks. We build on top of one such model and propose a hierarchy of BiLSTM and max pooling layers that implements an iterative refinement strategy and yields state of the art results on the SciTail dataset as well as strong results for SNLI and MultiNLI. We can show that the sentence embeddings learned in this way can be utilized in a wide variety of transfer learning tasks, outperforming InferSent on 7 out of 10 and SkipThought on 8 out of 9 SentEval sentence embedding evaluation tasks. Furthermore, our model beats the InferSent model in 8 out of 10 recently published SentEval probing tasks designed to evaluate sentence embeddings' ability to capture some of the important linguistic properties of sentences.

## Key Results

**Key NLI results**

* SNLI: 86.6% (600D model)
* SciTail: 86.0% (600D model)

**SentEval results**

Results for the [SentEval](https://github.com/facebookresearch/SentEval) sentence embedding evaluation library.

|Model | MR | CR | SUBJ | MPQA | SST | TREC | MRPC | SICK-R | SICK-E | STS14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |  --- |
|InferSent | 81.1 | 86.3 | 92.4 | 90.2 | **84.6** |  88.2 | 76.2/83.1 | **0.884** | **86.3** | .70/.67 |
|SkipThought | 79.4 | 83.1 | **93.7** | 89.3 | 82.9 | 88.4 | - | 0.858 | 79.5 | .44/.45 |
|**600D HBMP** | 81.5 | 86.4 | 92.7 |  89.8 | 83.6 |  86.4 |  74.6/82.0 | 0.876 | 85.3 | .70/.66 |
|**1200D HBMP**| **81.7** | **87.0** | **93.7** | **90.3** | 84.0 | **88.8** | **76.7/83.4** | 0.876 | 84.7 |  **.71/.68** |

**SentEval probing task results**

|Model | SentLen | WC | TreeDepth | TopConst | BShift | Tense | SubjNum | ObjNum | SOMO | CoordInv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| InferSent |  71.7 | **87.3** |  41.6 |  70.5 |  65.1 |  86.7 |  80.7 |  80.3 | **62.1** |  66.8 |
|**600D HBMP** | **75.9** |  84.1 |  42.9 |  76.6 |  64.3 |  86.2 |  83.7 |  79.3 |  58.9 |  68.5|
|**1200D HBMP** |  75.0 |  85.3 | **43.8** | **77.2** | **65.6** | **88.0** | **87.0** | **81.8** |  59.0 | **70.8** |

## Instructions

To replicate the results of our paper, follow the steps below.

**Install dependencies**

The following dependencies are required (versions used in brackets):
* Python (3.5.3)
* Pytorch (0.3.1)
* Numpy (1.14.3)
* Torchtext (for preprocessing) (0.2.1)
* SpaCy (for tokenization) (2.0.11)

For SpaCy you need to download the English model

```console
python -m spacy download en
```

**Download and prepare the datasets**

```console
./download_data.sh
```
This will download the needed datasets and word embeddings, including:
* [GloVe 840B 300D](https://nlp.stanford.edu/projects/glove/)
* [SNLI](https://nlp.stanford.edu/projects/snli/)
* [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/)
* [SciTail](http://data.allenai.org/scitail/)
* [Breaking NLI](https://github.com/BIU-NLP/Breaking_NLI)

**Train and test HBMP**

Run the train_hbmp.sh script to reproduce the NLI results for the HBMP model

```console
./train_hbmp.sh
```

Default settings for the SNLI dataset are as follows:

```console
python3 train.py \
  --epochs 20 \
  --batch_size 64 \
  --corpus snli \
  --encoder_type HBMP \
  --activation leakyrelu \
  --optimizer adam \
  --word_embedding glove.840B.300d \
  --embed_dim 300 \
  --fc_dim 600 \
  --hidden_dim 600 \
  --layers 1 \
  --dropout 0.1 \
  --learning_rate 0.0005 \
  --lr_patience 1 \
  --lr_decay 0.99 \
  --lr_reduction_factor 0.2 \
  --weight_decay 0 \
  --early_stopping_patience 3 \
  --save_path results \
  --seed 1234
  ```
To rerproduce the results for the other datasets, change the --corpus option to one of the following breaking_nli, multinli_matched, multinli_mismatched, scitail, all_nli.


In our paper some of the results for [InferSent](https://github.com/facebookresearch/InferSent) model were obtained using our implementation of the model. To train the InferSent model with our implementation use the train_infersent.sh script. See the paper for more details. 

```console
python3 train.py \
  --epochs 20 \
  --batch_size 64 \
  --corpus snli \
  --encoder_type BiLSTMMaxPoolEncoder \
  --activation tanh \
  --optimizer sgd \
  --word_embedding glove.840B.300d \
  --embed_dim 300 \
  --fc_dim 512 \
  --hidden_dim 2048 \
  --layers 1 \
  --dropout 0 \
  --learning_rate 0.1 \
  --lr_patience 1 \
  --lr_decay 0.99 \
  --lr_reduction_factor 0.2 \
  --save_path results \
  --seed 1234
  ```

## References

Please cite our paper if you find this code useful.

[1] Aarne Talman, Anssi Yli-Jyrä and Jörg Tiedemann. 2019. [Sentence Embeddings in NLI with Iterative Refinement Encoders](https://www.cambridge.org/core/journals/natural-language-engineering/article/sentence-embeddings-in-nli-with-iterative-refinement-encoders/AC811644D52446E414333B20FEACE00F). Natural Language Engineering. 25 (4), 467-482.

```
@article{talman_yli-jyra_tiedemann_2019, 
  title={Sentence embeddings in {NLI} with iterative refinement encoders}, 
  author={Talman, Aarne and Yli-Jyr\"a, Anssi and Tiedemann, J\"org}, 
  journal={Natural Language Engineering}, 
  volume={25},
  number={4}, 
  publisher={Cambridge University Press}, 
  year={2019}, 
  pages={467–482}}
```
