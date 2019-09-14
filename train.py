import os
import sys
import errno
import glob
import random
import numpy as np
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchtext import data
from torchtext import datasets
from classifier import NLIModel
from corpora import MultiNLI, SciTail, StanfordNLI, AllNLI, BreakingNLI
import pdb
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser(description='Helsinki NLI')
parser.add_argument("--corpus",
                    type=str,
                    choices=['snli', 'breaking_nli', 'multinli_matched', 'multinli_mismatched', 'scitail', 'all_nli'],
                    default='snli')
parser.add_argument('--epochs',
                    type=int,
                    default=20)
parser.add_argument('--batch_size',
                    type=int,
                    default=64)
parser.add_argument("--encoder_type",
                    type=str,
                    choices=['BiLSTMMaxPoolEncoder',
                             'LSTMEncoder',
                             'HBMP'],
                    default='HBMP')
parser.add_argument("--activation",
                    type=str,
                    choices=['tanh', 'relu', 'leakyrelu'],
                    default='relu')
parser.add_argument("--optimizer",
                    type=str,
                    choices=['rprop',
                             'adadelta',
                             'adagrad',
                             'rmsprop',
                             'adamax',
                             'asgd',
                             'adam',
                             'sgd'],
                    default='adam')
parser.add_argument('--embed_dim',
                    type=int,
                    default=300)
parser.add_argument('--fc_dim',
                    type=int,
                    default=600)
parser.add_argument('--hidden_dim',
                    type=int,
                    default=600)
parser.add_argument('--layers',
                    type=int,
                    default=1)
parser.add_argument('--dropout',
                    type=float,
                    default=0.1)
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.0005)
parser.add_argument('--lr_patience',
                    type=int,
                    default=1)
parser.add_argument('--lr_decay',
                    type=float,
                    default=0.99)
parser.add_argument('--lr_reduction_factor',
                    type=float,
                    default=0.2)
parser.add_argument('--weight_decay',
                    type=float,
                    default=0)
parser.add_argument('--gpu',
                    type=int,
                    default=0)
parser.add_argument('--preserve_case',
                    action='store_false',
                    dest='lower')
parser.add_argument('--word_embedding',
                    type=str,
                    default='glove.840B.300d')
parser.add_argument('--resume_snapshot',
                    type=str,
                    default='')
parser.add_argument('--early_stopping_patience',
                    type=int,
                    default=3)
parser.add_argument('--save_path',
                    type=str,
                    default='results')
parser.add_argument('--seed',
                    type=int,
                    default=1234)


def make_dirs(name):
    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def main():

    config = parser.parse_args()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)

    torch.cuda.device(config.gpu)

    inputs = data.Field(lower=config.lower, tokenize='spacy')
    labels = data.Field(sequential=False, unk_token=None)
    category_field = data.Field(sequential=False)
    id_field = data.Field(sequential=False, unk_token=None)


    if config.corpus == 'multinli_matched':
        train, dev, test = MultiNLI.splits_matched(inputs, labels, id_field)
        id_field.build_vocab(train, dev, test)
    elif config.corpus == 'multinli_mismatched':
        train, dev, test = MultiNLI.splits_mismatched(inputs, labels, id_field)
        id_field.build_vocab(train, dev, test)
    elif config.corpus == 'scitail':
        train, dev, test = SciTail.splits(inputs, labels)
    elif config.corpus == 'all_nli':
        train, dev, test = AllNLI.splits(inputs, labels, id_field)
        id_field.build_vocab(train, dev, test)
    elif config.corpus == 'breaking_nli':
        train, dev, test = BreakingNLI.splits(inputs, labels, category_field)
        category_field.build_vocab(test)
    else:
        train, dev, test = StanfordNLI.splits(inputs, labels)

    inputs.build_vocab(train, dev, test)
    labels.build_vocab(train)

    if config.word_embedding:
        pretrained_embedding = os.path.join(os.getcwd(), '.vector_cache/'+config.corpus+'_'+config.word_embedding+'.pt')
        if os.path.isfile(pretrained_embedding):
            inputs.vocab.vectors = torch.load(pretrained_embedding,
                           map_location=lambda storage, location: storage.cuda(config.gpu))
        else:
            print('Downloading pretrained {} word embeddings\n'.format(config.word_embedding))
            inputs.vocab.load_vectors(config.word_embedding)
            make_dirs(os.path.dirname(pretrained_embedding))
            torch.save(inputs.vocab.vectors, pretrained_embedding)


    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
                                                                 batch_size=config.batch_size,
                                                                 device=device)
    config.embed_size = len(inputs.vocab)
    config.out_dim = len(labels.vocab)
    config.cells = config.layers

    if config.encoder_type != 'LSTMEncoder':
        config.cells *= 2

    if config.resume_snapshot:
        model = torch.load(config.resume_snapshot,
                           map_location=lambda storage, location: storage.cuda(config.gpu))
    else:
        model = NLIModel(config)
        if config.word_embedding:
            model.sentence_embedding.word_embedding.weight.data = inputs.vocab.vectors
            model.cuda(device=config.gpu)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if config.optimizer == 'adadelta':
        optim_algorithm = optim.Adadelta
    elif config.optimizer == 'adagrad':
        optim_algorithm = optim.Adagrad
    elif config.optimizer == 'adam':
        optim_algorithm = optim.Adam
    elif config.optimizer == 'adamax':
        optim_algorithm = optim.Adamax
    elif config.optimizer == 'asgd':
        optim_algorithm = optim.ASGD
    elif config.optimizer == 'rmsprop':
        optim_algorithm = optim.RMSprop
    elif config.optimizer == 'rprop':
        optim_algorithm = optim.Rprop
    elif config.optimizer == 'sgd':
        optim_algorithm = optim.SGD
    else:
        raise Exception('Unknown optimization optimizer: "%s"' % config.optimizer)

    optimizer = optim_algorithm(model.parameters(),
                                lr=config.learning_rate,
                                weight_decay=config.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               'min',
                                               factor=config.lr_reduction_factor,
                                               patience=config.lr_patience,
                                               verbose=False,
                                               min_lr=1e-5)

    iterations = 0
    best_dev_acc = -1
    dev_accuracies = []
    best_dev_loss = 1
    early_stopping = 0
    stop_training = False
    train_iter.repeat = False
    make_dirs(config.save_path)

    # Print parameters and config
    print('\nConfig: {}\n'.format(sys.argv[1:]))
    print(config)

    # Print the model
    print('Model:\n')
    print(model)
    print('\n')
    params = sum([p.numel() for p in model.parameters()])
    print('Parameters: {}'.format(params))
    print('\nTraining started...\n')

    # Train for the number of epochs specified
    for epoch in range(config.epochs):
        if stop_training == True:
            break

        train_iter.init_epoch()
        n_correct = 0
        n_total = 0
        all_losses = []
        train_accuracies = []
        all_losses = []

        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * config.lr_decay if epoch>0\
        and config.optimizer == 'sgd' else optimizer.param_groups[0]['lr']
        print('\nEpoch: {:>02.0f}/{:<02.0f}'.format(epoch+1, config.epochs), end=' ')
        print('(Learning rate: {})'.format(optimizer.param_groups[0]['lr']))

        for batch_idx, batch in enumerate(train_iter):

            model.train()
            optimizer.zero_grad()
            iterations += 1
            answer = model(batch)
            # sys.exit()
            # Calculate accuracy

            n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
            n_total += batch.batch_size
            train_acc = 100. * n_correct/n_total
            train_accuracies.append(train_acc.item())

            # Calculate loss
            loss = criterion(answer, batch.label)
            all_losses.append(loss.item())

            # Backpropagate and update the learning rate
            loss.backward()
            optimizer.step()

            print('Progress: {:3.0f}% - Batch: {:>4.0f}/{:<4.0f} - Loss: {:6.2f}% - Accuracy: {:6.2f}%'.format(
                100. * (1+batch_idx) / len(train_iter),
                1+batch_idx, len(train_iter),
                round(100. * np.mean(all_losses), 2),
                round(np.mean(train_accuracies), 2)), end='\r')

            # Evaluate performance
            # if iterations % config.dev_every == 0:
            if 1+batch_idx == len(train_iter):
                # Switch model to evaluation mode
                model.eval()
                dev_iter.init_epoch()

                # Calculate Accuracy
                n_dev_correct = 0
                dev_loss = 0
                dev_losses = []

                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                    answer = model(dev_batch)
                    n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == \
                        dev_batch.label.data).sum()
                    dev_loss = criterion(answer, dev_batch.label)
                    dev_losses.append(dev_loss.item())

                dev_acc = 100. * n_dev_correct / len(dev)
                dev_acc=dev_acc.item()
                dev_accuracies.append(dev_acc)

                print('\nDev loss: {}% - Dev accuracy: {}%'.format(round(100.*np.mean(dev_losses), 2), round(dev_acc, 2)))

                # Update validation best accuracy if it is better than
                # already stored
                if dev_acc > best_dev_acc:

                    best_dev_acc = dev_acc
                    best_dev_epoch = 1+epoch
                    snapshot_prefix = os.path.join(config.save_path, 'best')
                    dev_snapshot_path = snapshot_prefix + \
                        '_{}_{}D_devacc_{}_epoch_{}.pt'.format(config.encoder_type, config.hidden_dim, round(dev_acc, 2), 1+epoch)

                    # save model, delete previous snapshot
                    torch.save(model, dev_snapshot_path)
                    for f in glob.glob(snapshot_prefix + '*'):
                        if f != dev_snapshot_path:
                            os.remove(f)

                # Check for early stopping
                if np.mean(dev_losses) < best_dev_loss:
                    best_dev_loss = np.mean(dev_losses)
                else:
                    early_stopping += 1

                if early_stopping > config.early_stopping_patience and config.optimizer != 'sgd':
                    stop_training = True
                    print('\nEarly stopping')

                if config.optimizer == 'sgd' and optimizer.param_groups[0]['lr'] < 1e-5:
                    stop_training = True
                    print('\nEarly stopping')

                # Update learning rate
                scheduler.step(round(np.mean(dev_losses), 2))
                dev_losses = []


            # If training has completed, calculate the test scores
            if stop_training == True or (1+epoch == config.epochs and 1+batch_idx == len(train_iter)):
                print('\nTraining completed after {} epocs.\n'.format(1+epoch))


                #Save the final model
                final_snapshot_prefix = os.path.join(config.save_path, 'final')
                final_snapshot_path = final_snapshot_prefix + \
                '_{}_{}D.pt'.format(config.encoder_type, config.hidden_dim)
                torch.save(model, final_snapshot_path)
                for f in glob.glob(final_snapshot_prefix + '*'):
                    if f != final_snapshot_path:
                        os.remove(f)

                # Evaluate the best dev model
                test_model = torch.load(dev_snapshot_path)
                # Switch model to evaluation mode
                test_model.eval()
                test_iter.init_epoch()

                # Calculate Accuracy
                n_test_correct = 0
                test_loss = 0
                test_losses = []

                for test_batch_idx, test_batch in enumerate(test_iter):
                    answer = test_model(test_batch)
                    n_test_correct += (torch.max(answer, 1)[1].view(test_batch.label.size()).data == \
                        test_batch.label.data).sum()
                    test_loss = criterion(answer, test_batch.label)
                    test_losses.append(test_loss.item())

                test_acc = 100. * n_test_correct / len(test)
                test_acc=test_acc.item()

                print('SUMMARY:')
                print('Encoder: {}'.format(config.encoder_type))
                if config.encoder_type == 'BiLSTMMaxPoolEncoder' or config.encoder_type == \
                'HBMP' or config.encoder_type == 'HAttentionBiLSTMEncoder':
                    print('Sentence embedding size: {}D'.format(2*config.hidden_dim))
                else:
                    print('Sentence embedding size: {}D'.format(config.hidden_dim))

                print('\nMean dev accuracy: {:6.2f}%\n'.format(round(np.mean(dev_accuracies)), 2))
                print('BEST MODEL:')
                print('Early stopping patience: {}'.format(config.early_stopping_patience))
                print('Epoch: {}'.format(best_dev_epoch))
                print('Dev accuracy: {:<6.2f}%'.format(round(best_dev_acc, 2)))
                print('Test loss: {:<.2f}%'.format(round(100. * np.mean(test_losses), 2)))
                print('Test accuracy: {:<5.2f}%\n'.format(round(test_acc, 2)))


if __name__ == '__main__':
    main()
