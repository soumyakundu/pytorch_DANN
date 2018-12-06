"""
Main script for models
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import models
from train import test, train, params
from util import utils
from sklearn.manifold import TSNE
import argparse, sys, os
import torch
from torch.autograd import Variable
import time
from torchsummary import summary

#import keras_model
#import keras
#import metrics

def visualizePerformance(feature_extractor, class_classifier, domain_classifier, src_test_dataloader,
                         tgt_test_dataloader, num_of_samples=None, imgName=None):
    """
    Evaluate the performance of dann and source only by visualization.

    :param feature_extractor: network used to extract feature from target samples
    :param class_classifier: network used to predict labels
    :param domain_classifier: network used to predict domain
    :param source_dataloader: test dataloader of source domain
    :param target_dataloader: test dataloader of target domain
    :param num_of_samples: the number of samples (from train and test respectively) for t-sne
    :param imgName: the name of saving image

    :return:
    """

    print("\n WARNING: I AM HERE \n")

    # Setup the network
    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()

    # Randomly select samples from source domain and target domain.
    if num_of_samples is None:
        num_of_samples = params.batch_size
    else:
        assert len(src_test_dataloader) * num_of_samples, \
            'The number of samples can not bigger than dataset.' # NOT PRECISELY COMPUTATION

    # Collect source data.
    s_images, s_labels, s_tags = [], [], []
    for batch in src_test_dataloader:
        images, labels = batch

        if params.use_gpu:
            s_images.append(images.cuda())
        else:
            s_images.append(images)
        s_labels.append(labels)

        s_tags.append(torch.zeros((labels.size()[0])).type(torch.LongTensor))

        if len(s_images * params.batch_size) > num_of_samples:
            break

    s_images, s_labels, s_tags = torch.cat(s_images)[:num_of_samples], \
                                 torch.cat(s_labels)[:num_of_samples], torch.cat(s_tags)[:num_of_samples]


    # Collect test data.
    t_images, t_labels, t_tags = [], [], []
    for batch in tgt_test_dataloader:
        images, labels = batch

        if params.use_gpu:
            t_images.append(images.cuda())
        else:
            t_images.append(images)
        t_labels.append(labels)

        t_tags.append(torch.ones((labels.size()[0])).type(torch.LongTensor))

        if len(t_images * params.batch_size) > num_of_samples:
            break

    t_images, t_labels, t_tags = torch.cat(t_images)[:num_of_samples], \
                                 torch.cat(t_labels)[:num_of_samples], torch.cat(t_tags)[:num_of_samples]

    # Compute the embedding of target domain.
    embedding1 = feature_extractor(s_images)
    embedding2 = feature_extractor(t_images)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)

    if params.use_gpu:
        dann_tsne = tsne.fit_transform(np.concatenate((embedding1.cpu().detach().numpy(),
                                                       embedding2.cpu().detach().numpy())))
    else:
        dann_tsne = tsne.fit_transform(np.concatenate((embedding1.detach().numpy(),
                                                   embedding2.detach().numpy())))


    utils.plot_embedding(dann_tsne, np.concatenate((s_labels, t_labels)),
                         np.concatenate((s_tags, t_tags)), 'Domain Adaptation', imgName)




def main(args):

    # Set global parameters.
    params.fig_mode = args.fig_mode
    params.epochs = args.max_epoch
    params.num_train = args.num_train
    params.num_test = args.num_test
    params.upsample = args.upsample
    params.training_mode = args.training_mode
    params.source_domain = args.source_domain
    params.target_domain = args.target_domain
    if params.embed_plot_epoch is None:
        params.embed_plot_epoch = args.embed_plot_epoch
    params.lr = args.lr

    if args.save_dir is not None:
        params.save_dir = args.save_dir

    print("Starting up")

    device = torch.device("cuda")

    # prepare the source data and target data

    src_train_dataloader = utils.get_train_loader(params.source_domain)
    src_test_dataloader = utils.get_test_loader(params.source_domain)
    tgt_train_dataloader = utils.get_train_loader(params.target_domain)
    tgt_test_dataloader = utils.get_test_loader(params.target_domain)

    # init models

    model_index = params.source_domain + '_' + params.target_domain
    feature_extractor = params.extractor_dict[model_index]
    class_classifier = params.class_dict[model_index]
    domain_classifier = params.domain_dict[model_index]
    critic = models.Critic()

    """
    kmodel1 = keras_model.getModelGivenModelOptionsAndWeightInits('/srv/scratch/soumyak/metadata/encode-roadmap.dnase_tf-chip.batch_256.params.npz')

    kmodel2 = keras.models.load_model('/srv/scratch/soumyak/outputs/final/human', custom_objects={"positive_accuracy": metrics.positive_accuracy,
                    "negative_accuracy": metrics.negative_accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall})
                    #"softMaxAxis1":kerasAC.activations.softMaxAxis1})

    kmodel2.save_weights('my_weights.h5')
    kmodel1.load_weights('my_weights.h5')

    weight_dict = dict()
    for layer in kmodel1.layers:
        if type(layer) is keras.layers.convolutional.Conv2D:
            weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (3, 2, 0, 1))
            weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
        elif type(layer) is keras.layers.Dense:
            weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (1, 0))
            weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
        elif type(layer) is keras.layers.normalization.BatchNormalization:
            weight_dict[layer.get_config()['name'] + '.weight'] = layer.get_weights()[0]
            weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
            weight_dict[layer.get_config()['name'] + '.running_mean'] = layer.get_weights()[2]
            weight_dict[layer.get_config()['name'] + '.running_var'] = layer.get_weights()[3]

    #feature_extractor.load_state_dict(torch.load('best_feature_extractor.pt'))
    #class_classifier.load_state_dict(torch.load('best_class_classifier.pt'))

    for key in feature_extractor.state_dict().keys():
        if not key.endswith('num_batches_tracked'):
            feature_extractor.state_dict()[key] = torch.from_numpy(weight_dict[key])

    for key in class_classifier.state_dict().keys():
        if not key.endswith('num_batches_tracked'):
            class_classifier.state_dict()[key] = torch.from_numpy(weight_dict[key])

    torch.save(feature_extractor.state_dict(), 'feature_extractor.pt')
    torch.save(class_classifier.state_dict(), 'class_classifier.pt')
    """

    feature_extractor.load_state_dict(torch.load('feature_extractor.pt'))
    class_classifier.load_state_dict(torch.load('class_classifier.pt'))

    print("Loaded weights")

    feature_extractor.to(device)
    class_classifier.to(device)
    domain_classifier.to(device)
    critic.to(device)

    class_criterion = nn.BCELoss()
    domain_criterion = nn.BCELoss()

    if args.training_mode == 'dann':
        optimizer = optim.Adam([{'params': feature_extractor.parameters()},
                               {'params': class_classifier.parameters()},
                               {'params': domain_classifier.parameters()}], lr= params.lr)
    elif args.training_mode == 'wdgrl':
        optimizer = optim.Adam([{'params': feature_extractor.parameters()},
                               {'params': class_classifier.parameters()},
                               {'params': critic.parameters()}], lr= params.lr)
    else:
        optimizer = optim.Adam([{'params': feature_extractor.parameters()},
                                {'params': class_classifier.parameters()}], lr=params.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    best_auprc = 0.0

    for epoch in range(params.epochs):
        print('Epoch: {}'.format(epoch))
        train.train(args.training_mode, feature_extractor, class_classifier, domain_classifier, critic, class_criterion, domain_criterion,
                    src_train_dataloader, tgt_train_dataloader, optimizer, epoch, device)
        auprc = test.test(feature_extractor, class_classifier, domain_classifier, src_test_dataloader, tgt_test_dataloader, best_auprc, 'target', device)
        if auprc > best_auprc:
            best_auprc = auprc
            print("Best AUPRC")
            torch.save(feature_extractor.state_dict(), 'best_feature_extractor_wdgrl.pt')
            torch.save(class_classifier.state_dict(), 'best_class_classifier_wdgrl.pt')
            torch.save(domain_classifier.state_dict(), 'best_domain_classifier_wdgrl.pt')
            print("Saved Best Model")

        # Plot embeddings periodically.
        if epoch % params.embed_plot_epoch == 0 and params.fig_mode is not None:
            visualizePerformance(feature_extractor, class_classifier, domain_classifier, src_test_dataloader,
                                 tgt_test_dataloader, imgName='embedding_' + str(epoch))



def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_domain', type= str, default= 'Human', help= 'Choose source domain.')
    parser.add_argument('--target_domain', type= str, default= 'Mouse', help = 'Choose target domain.')
    parser.add_argument('--fig_mode', type=str, default=None, help='Plot experiment figures.')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to save plotted images.')
    parser.add_argument('--training_mode', type=str, default='dann', help='Choose a mode to train the model.')
    parser.add_argument('--max_epoch', type=int, default=100, help='The max number of epochs.')
    parser.add_argument('--embed_plot_epoch', type= int, default=100, help= 'Epoch number of plotting embeddings.')
    parser.add_argument('--lr', type= float, default= 0.001, help= 'Learning rate.')
    parser.add_argument('--upsample', type=int, default=3, help='Set frequency of sampling positive training example.')
    parser.add_argument('--num_train', type=int, default=100000, help='Number of training examples per epoch.')
    parser.add_argument('--num_test', type=int, default=50000, help='Number of testing examples per epoch.')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
