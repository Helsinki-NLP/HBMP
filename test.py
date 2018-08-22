import numpy as np
import sys
from argparse import ArgumentParser
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchtext import data
from torchtext import datasets
from corpora import MultiNLI, SciTail, StanfordNLI, AllNLI, BreakingNLI

parser = ArgumentParser(description='Helsinki NLI System')
parser.add_argument('--model_path',
                    type=str,
                    default='results/model.pt')
parser.add_argument("--corpus",
                    type=str,
                    choices=['snli', 'breaking_nli', 'all_nli', 'multinli_matched', 'multinli_mismatched', 'scitail'],
                    default='snli')
parser.add_argument('--batch_size',
                    type=int,
                    default=64)
parser.add_argument('--seed',
                    type=int,
                    default=1234)
parser.add_argument('--gpu',
                    type=int,
                    default=0)
parser.add_argument('--preserve_case',
                    action='store_false',
                    dest='lower')


def main():

    config = parser.parse_args()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)

    inputs = data.Field(lower=config.lower, tokenize='spacy')
    labels = data.Field(sequential=False, unk_token=None)
    category_field = data.Field(sequential=False)
    id_field = data.Field(sequential=False, unk_token=None)

    if config.corpus == 'multinli_matched':
        train, dev, test = MultiNLI.splits_matched(inputs, labels, id_field)
        id_field.build_vocab(train, dev, test)
        f = open(config.corpus+'_kaggle_test.csv', 'w+')
    elif config.corpus == 'multinli_mismatched':
        train, dev, test = MultiNLI.splits_mismatched(inputs, labels, id_field)
        id_field.build_vocab(train, dev, test)
        f = open(config.corpus+'_kaggle_test.csv', 'w+')
    elif config.corpus == 'scitail':
        train, dev, test = SciTail.splits(inputs, labels)
    elif config.corpus == 'all_nli':
        train, dev, test = AllNLI.splits(inputs, labels)
        id_field.build_vocab(train, dev, test)
    elif config.corpus == 'breaking_nli':
        train, dev, test = BreakingNLI.splits(inputs, labels, category_field)
        category_field.build_vocab(test)
    else:
        train, dev, test = StanfordNLI.splits(inputs, labels)

    inputs.build_vocab(train, dev, test)
    labels.build_vocab(train)

    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
        batch_size=config.batch_size,
        device=config.gpu)

    # Loss
    criterion = nn.CrossEntropyLoss()

    test_model = torch.load(config.model_path)

    # Switch model to evaluation mode
    test_model.eval()
    test_iter.init_epoch()

    # Calculate Accuracy
    n_test_correct = 0
    test_loss = 0
    test_losses = []

    if config.corpus == 'multinli_mismatched' or config.corpus == 'multinli_matched':
        f.write('pairID,gold_label\n')

    print('ID | PREMISE | HYPOTHESIS | PREDICTION | RESULT | GOLD LABEL')
    for test_batch_idx, test_batch in enumerate(test_iter):
        # Make predictions
        answer = test_model(test_batch)
        # Keep track of location. Start form first item of the batch
        uid = 1+test_batch_idx*config.batch_size
        # Print the premise
        for i in range(test_batch.batch_size):
            if config.corpus == 'scitail' or config.corpus == 'breaking_nli':
                print('{} |'.format(i+uid), end=' ')
            else:
                print('{} |'.format(id_field.vocab.itos[test_batch.pair_id[i].data[0]]), end=' ')
            for prem in test_batch.premise.transpose(0,1)[i]:
                x = prem.data[0]
                if not inputs.vocab.itos[x] == '<pad>':
                    print(inputs.vocab.itos[x], end=' ')
            print('|', end=' ')
            # Print the hypothesis
            for hypo in test_batch.hypothesis.transpose(0,1)[i]:
                y = hypo.data[0]
                if not inputs.vocab.itos[y] == '<pad>':
                    print(inputs.vocab.itos[y], end=' ')
            print('|', end=' ')
            # Compare the prediction with the gold label and print
            for j, label in enumerate(answer[i]):
                if label.data[0] == torch.max(answer[i]).data[0]:
                    if config.corpus == 'multinli_mismatched' or config.corpus == 'multinli_matched':
                        f.write('{},{}\n'.format(id_field.vocab.itos[test_batch.pair_id[i].data[0]], labels.vocab.itos[j]))
                    print(labels.vocab.itos[j], end=' ')
                    if j == test_batch.label[i].data[0]:
                        print('| CORRECT |', end=' ')
                        print(labels.vocab.itos[test_batch.label[i].data[0]], end=' ')
                    else:
                        print('| INCORRECT |', end=' ')
                        print(labels.vocab.itos[test_batch.label[i].data[0]], end=' ')
            if config.corpus == 'breaking_nli':
                print('| {}'.format(category_field.vocab.itos[test_batch.category[i].data[0]]))
            else:
                print('')


        # Calculate the accuracy
        n_test_correct += (torch.max(answer, 1)[1].view(test_batch.label.size()).data == \
            test_batch.label.data).sum()
        test_loss = criterion(answer, test_batch.label)
        test_losses.append(test_loss.data[0])


    if config.corpus == 'multinli_mismatched' or config.corpus == 'multinli_matched':
        f.close()
    test_acc = 100. * n_test_correct / len(test)

    print('\nLoss: {:.4f} / Accuracy: {:.4f}\n'.format(round(np.mean(test_losses), 2), test_acc))

if __name__ == '__main__':
    main()
