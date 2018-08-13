#!/bin/bash

# Create folder structure
echo -e "\nCreating the folder structure...\n"
mkdir .data
mkdir .data/multinli
mkdir .data/snli
mkdir .data/scitail
mkdir .data/breaking_nli
mkdir .vector_cache
echo -e "Done!"

# Download and unzip GloVe word embeddings
echo -e "\nDownloading and unzipping Glove 840B 300D to .vector_cache\n"
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip -a glove.840B.300d.zip -d .vector_cache/
rm -f glove.840B.300d.zip
echo -e "\nDone!"

# Download and unzip NLI corpora

# MultiNLI:
echo -e "\nDownloading and unzipping MultiNLI to .data\n"
wget http://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
unzip -a multinli_1.0.zip -d .data/multinli/
rm -f multinli_1.0.zip
echo -e "\nDone!"

# SNLI
echo -e "\nDownloading and unzipping SNLI to .data\n"
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip -a snli_1.0.zip -d .data/snli/
rm -f snli_1.0.zip
echo -e "\nDone!"

# SciTail
echo -e "\nDownloading and unzipping SciTail to .data\n"
wget http://data.allenai.org.s3.amazonaws.com/downloads/SciTailV1.zip
unzip -a SciTailV1.zip -d .data/scitail/
rm -f SciTailV1.zip
echo -e "\nDone!"

# Create All NLI training set
echo -e "\nCreating All NLI training set\n"
cat .data/snli/snli_1.0/snli_1.0_train.jsonl .data/multinli/multinli_1.0/multinli_1.0_train.jsonl > .data/snli/snli_1.0/all_nli.jsonl
echo -e "\nAll NLI created"
wc -l .data/snli/snli_1.0/all_nli.jsonl
echo -e "\nDone!"

#Breaking NLI
echo -e "\nDownloading and unzipping Breaking NLI to .data\n"
wget http://u.cs.biu.ac.il/~nlp/wp-content/uploads/breaking_nli_dataset.zip
unzip -a breaking_nli_dataset.zip -d .data/breaking_nli/
mv .data/breaking_nli/data/dataset.jsonl .data/breaking_nli/data/breaking_test.jsonl
rm -f breaking_nli_dataset.zip 
cp .data/snli/snli_1.0/snli_1.0_train.jsonl .data/breaking_nli/data/breaking_train.jsonl
cp .data/snli/snli_1.0/snli_1.0_dev.jsonl .data/breaking_nli/data/breaking_dev.jsonl
# prepare training and dev for torchtext with empty field "category"
sed -i 's/{/{"category": " ", /g' .data/breaking_nli/data/breaking_train.jsonl
sed -i 's/{/{"category": " ", /g' .data/breaking_nli/data/breaking_dev.jsonl
echo -e "\nAll Done!"