from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import numpy as np
from torchtext import data
from torchtext import datasets
import logging
from embeddings import HBMP


PATH_SENTEVAL = ''
PATH_TO_DATA = 'data'
PATH_TO_GLOVE = 'glove/glove.840B.300d.txt'
MODEL_PATH = 'HBMP.model'

assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_GLOVE), \
    'Set MODEL and GloVe PATHs'


sys.path.insert(0, PATH_SENTEVAL)
import senteval


def prepare(params, samples):
    params.inputs.build_vocab(samples)
    params.inputs.vocab.load_vectors('glove.840B.300d')
    params.hbmp.word_embedding.weight.data = params.inputs.vocab.vectors

def batcher(params, batch):
    sentences = []
    for s in batch:
      sentence = params.inputs.preprocess(s)
      sentences.append(sentence)
    sentences = params.inputs.process(sentences, train=True, device=0)
    params.hbmp = params.hbmp.cuda()
    emb = params.hbmp.forward(sentences.cuda())
    embeddings = []
    for sent in emb:
      sent = sent.cpu()
      embeddings.append(sent.data.cpu().numpy())
    embeddings = np.vstack(embeddings)
    return embeddings


# params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'seed':1234}
# params_senteval['classifier'] = {'nhid': 600, 'optim': 'adam,lr=0.0005', 'batch_size': 64,
#                                  'tenacity': 5, 'epoch_size': 4, 'dropout': 0.2}

params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":

    params_senteval['hbmp'] = torch.load(MODEL_PATH)
    params_senteval['inputs'] = data.Field(lower=True, tokenize='spacy')

    se = senteval.engine.SE(params_senteval, batcher, prepare)

    # define transfer tasks
    transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
                      'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
                      'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']

    #['MR', 'CR', 'SUBJ', 'MPQA', 'STSBenchmark', 'SST2', 'SST5', 'TREC', 'MRPC', 
    #'SICKRelatedness', 'SICKEntailment', 'STS14']

    results = se.eval(transfer_tasks)
    print(results)