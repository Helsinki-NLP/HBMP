# Natural Language Inference with Hierarchical BiLSTM Max Pooling Architecture (HMBM)

To replicate the results of our paper, follow the steps below.

**Install dependencies**

The code has been tested with the following versions:
* Pytorch (0.3.1)
* Numpy (1.14.3)
* Torchtext (for preprocessing) (0.2.1)
* Spacy (for tokenization) (2.0.11)

**Download and prepare the datasets**

```console
sh download_data.sh
```
This will download the needed datasets and word embeddings, including:
* GloVe 840B 300D word embeddings (https://nlp.stanford.edu/projects/glove/)
* SNLI (https://nlp.stanford.edu/projects/snli/)
* MultiNLI (https://www.nyu.edu/projects/bowman/multinli/)
* SciTail (http://data.allenai.org/scitail/)
* Breaking NLI (https://github.com/BIU-NLP/Breaking_NLI)

**Train and test HBMP**

Run the train_hbmp.sh script to reproduce the NLI results for the HBMP model

```console
sh train_hbmp.sh
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


In our paper some of the results for InferSent model were obtained using our implementation of the model. To train the InferSent model with our implementation use the train_infersent.sh script. See the paper for more details.

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

