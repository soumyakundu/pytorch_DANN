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
        num_of_samples = params.test_batch_size
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

        if len(s_images * params.test_batch_size) > num_of_samples:
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

        if len(t_images * params.test_batch_size) > num_of_samples:
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

    if params.training_mode == 'source':
        title = 'Source'
    elif params.training_mode == 'target':
        title = 'Target'
    elif params.training_mode == 'dann':
        title = 'Domain Adversarial Neural Network (DANN)'
    elif params.training_mode == 'wdgrl':
        title = 'Wasserstein Distance Guided Representation Learning (WDGRL)'
    elif params.training_mode == 'adda':
        title = 'Adversarial Discriminative Domain Adaptation (ADDA)'

    utils.plot_embedding(dann_tsne, np.concatenate((s_labels, t_labels)),
                         np.concatenate((s_tags, t_tags)), title, imgName)


def main(args):

    # Set global parameters.
    params.fig_mode = args.fig_mode
    params.epochs = args.max_epoch
    params.num_train = args.num_train
    params.num_test = args.num_test
    params.train_upsample = args.train_upsample
    params.test_upsample = args.test_upsample
    params.train_batch_size = args.train_batch_size
    params.test_batch_size = args.test_batch_size
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
    device2 = torch.device("cpu")

    # prepare the source data and target data

    #"""
    src_train_dataloader = utils.get_train_loader(params.source_domain)
    src_test_dataloader = utils.get_test_loader(params.source_domain, args.mode)
    tgt_train_dataloader = utils.get_train_loader(params.target_domain)
    tgt_test_dataloader = utils.get_test_loader(params.target_domain, args.mode)
    #"""

    # init models

    model_index = params.source_domain + '_' + params.target_domain
    feature_extractor = params.extractor_dict[model_index]
    class_classifier = params.class_dict[model_index]
    domain_classifier = params.domain_dict[model_index]
    critic = models.Critic()
    #feature_extractor2 = params.extractor_dict[model_index]
    #class_classifier2 = params.class_dict[model_index]

    """
    kmodel1 = keras_model.getModelGivenModelOptionsAndWeightInits('/srv/scratch/soumyak/metadata/encode-roadmap.dnase_tf-chip.batch_256.params.npz')

    #kmodel2 = keras.models.load_model('/srv/scratch/soumyak/outputs/human_liver_adult', custom_objects={"positive_accuracy": metrics.positive_accuracy,
    #                "negative_accuracy": metrics.negative_accuracy,
    #                "precision": metrics.precision,
    #                "recall": metrics.recall})

    #kmodel2.save_weights('my_weights.h5')
    #kmodel2.save('my_model.h5')
    #kmodel1.load_weights('my_weights.h5')

    weight_dict = dict()
    for layer in kmodel1.layers:
        if type(layer) is keras.layers.convolutional.Conv2D:
            weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(np.array(layer.get_weights()[0]), (3, 2, 0, 1))
            weight_dict[layer.get_config()['name'] + '.bias'] = np.array(layer.get_weights()[1])
        elif type(layer) is keras.layers.Dense:
            weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(np.array(layer.get_weights()[0]), (1, 0))
            weight_dict[layer.get_config()['name'] + '.bias'] = np.array(layer.get_weights()[1])
        elif type(layer) is keras.layers.normalization.BatchNormalization:
            weight_dict[layer.get_config()['name'] + '.weight'] = np.array(layer.get_weights()[0])
            weight_dict[layer.get_config()['name'] + '.bias'] = np.array(layer.get_weights()[1])
            weight_dict[layer.get_config()['name'] + '.running_mean'] = np.array(layer.get_weights()[2])
            weight_dict[layer.get_config()['name'] + '.running_var'] = np.array(layer.get_weights()[3])

    pyt_state_dict = feature_extractor.state_dict()
    for key in pyt_state_dict.keys():
        if not key.endswith('num_batches_tracked'):# and not key.endswith('running_mean') and not key.endswith('running_var'):# and not key.startswith('bn'):
            pyt_state_dict[key] = torch.from_numpy(weight_dict[key]).contiguous()
    feature_extractor.load_state_dict(pyt_state_dict)

    pyt_state_dict2 = class_classifier.state_dict()
    for key in pyt_state_dict2.keys():
        if not key.endswith('num_batches_tracked'):# and not key.endswith('running_mean') and not key.endswith('running_var'):# and not key.startswith('bn'):
            pyt_state_dict2[key] = torch.from_numpy(weight_dict[key]).contiguous()
    class_classifier.load_state_dict(pyt_state_dict2)

    torch.save(feature_extractor.state_dict(), 'best_feature_extractor_mouse.pt')
    torch.save(class_classifier.state_dict(), 'best_class_classifier_mouse.pt')
    #"""

    if args.training_mode == 'dann':
        if args.source_domain == 'Human':
            if args.mode == 'train':
                feature_extractor.load_state_dict(torch.load('best_feature_extractor.pt'))
                class_classifier.load_state_dict(torch.load('best_class_classifier.pt'))
            else:
                feature_extractor.load_state_dict(torch.load('best_feature_extractor_dann.pt'))
                class_classifier.load_state_dict(torch.load('best_class_classifier_dann.pt'))
                domain_classifier.load_state_dict(torch.load('best_domain_classifier.pt'))
        else:
            if args.mode == 'train':
                feature_extractor.load_state_dict(torch.load('best_feature_extractor_Mouse_source.pt'))
                class_classifier.load_state_dict(torch.load('best_class_classifier_Mouse_source.pt'))
            else:
                feature_extractor.load_state_dict(torch.load('best_feature_extractor_Mouse_dann.pt'))
                class_classifier.load_state_dict(torch.load('best_class_classifier_Mouse_dann.pt'))
                domain_classifier.load_state_dict(torch.load('best_domain_classifier_Mouse.pt'))

    elif args.training_mode == 'wdgrl':
        if args.source_domain == 'Human':
            if args.mode == 'train':
                feature_extractor.load_state_dict(torch.load('best_feature_extractor.pt'))
                class_classifier.load_state_dict(torch.load('best_class_classifier.pt'))
            else:
                feature_extractor.load_state_dict(torch.load('best_feature_extractor_wdgrl.pt'))
                class_classifier.load_state_dict(torch.load('best_class_classifier_wdgrl.pt'))
                domain_classifier.load_state_dict(torch.load('best_domain_classifier.pt'))
        else:
            if args.mode == 'train':
                feature_extractor.load_state_dict(torch.load('best_feature_extractor_Mouse_source.pt'))
                class_classifier.load_state_dict(torch.load('best_class_classifier_Mouse_source.pt'))
            else:
                feature_extractor.load_state_dict(torch.load('best_feature_extractor_Mouse_wdgrl.pt'))
                class_classifier.load_state_dict(torch.load('best_class_classifier_Mouse_wdgrl.pt'))
                domain_classifier.load_state_dict(torch.load('best_domain_classifier_Mouse.pt'))

    elif args.training_mode == 'adda':
        if args.source_domain == 'Human':
            if args.mode == 'train':
                feature_extractor.load_state_dict(torch.load('best_feature_extractor.pt'))
                class_classifier.load_state_dict(torch.load('best_class_classifier.pt'))
            else:
                feature_extractor.load_state_dict(torch.load('best_feature_extractor_adda.pt'))
                class_classifier.load_state_dict(torch.load('best_class_classifier_adda.pt'))
                domain_classifier.load_state_dict(torch.load('best_domain_classifier.pt'))
        else:
            if args.mode == 'train':
                feature_extractor.load_state_dict(torch.load('best_feature_extractor_Mouse_source.pt'))
                class_classifier.load_state_dict(torch.load('best_class_classifier_Mouse_source.pt'))
            else:
                feature_extractor.load_state_dict(torch.load('best_feature_extractor_Mouse_adda.pt'))
                class_classifier.load_state_dict(torch.load('best_class_classifier_Mouse_adda.pt'))
                domain_classifier.load_state_dict(torch.load('best_domain_classifier_Mouse.pt'))

    elif args.training_mode == 'source':
        if args.source_domain == 'Human':
            feature_extractor.load_state_dict(torch.load('best_feature_extractor.pt'))
            class_classifier.load_state_dict(torch.load('best_class_classifier.pt'))
        else:
            feature_extractor.load_state_dict(torch.load('best_feature_extractor_Mouse_source.pt'))
            class_classifier.load_state_dict(torch.load('best_class_classifier_Mouse_source.pt'))

    feature_extractor.load_state_dict(torch.load('best_feature_extractor_dann.pt'))
    class_classifier.load_state_dict(torch.load('best_class_classifier_dann.pt'))
    #domain_classifier.load_state_dict(torch.load('best_domain_classifier_Mouse.pt'))

    feature_extractor.to(device)
    class_classifier.to(device)
    domain_classifier.to(device)
    critic.to(device)
    #feature_extractor2.to(device)
    #class_classifier2.to(device)

    class_criterion = nn.BCELoss()
    domain_criterion = nn.BCELoss()

    best_auprc = 0.0

    if args.training_mode == 'dann':
        optimizer = optim.Adam([{'params': feature_extractor.parameters()},
                               {'params': class_classifier.parameters()},
                               {'params': domain_classifier.parameters()}], lr= params.lr)
        best_auprc = 0.3225
        args.objective = 'target'
    elif args.training_mode == 'wdgrl':
        optimizer = optim.Adam([{'params': feature_extractor.parameters()},
                               {'params': class_classifier.parameters()},
                               {'params': critic.parameters()}], lr= params.lr)
        best_auprc = 0.32
        args.objective = 'target'
    elif args.training_mode == 'adda':
        optimizer = optim.Adam([{'params': feature_extractor.parameters()},
                               {'params': class_classifier.parameters()},
                               {'params': critic.parameters()}], lr= params.lr)
        best_auprc = 0.32
        args.objective = 'target'
    else:
        optimizer = optim.Adam([{'params': feature_extractor.parameters()},
                                {'params': class_classifier.parameters()}], lr=params.lr)
        if args.source_domain == 'Human':
            best_auprc = 0.60
        else:
            best_auprc = 0.7762
        args.objective = 'source'

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(params.epochs):
        print('Epoch: {}'.format(epoch))
        if args.mode == 'train':
            train.train(args.training_mode, feature_extractor, class_classifier, domain_classifier, critic, class_criterion, domain_criterion, src_train_dataloader, tgt_train_dataloader, optimizer, epoch, device, None, None)
            auprc = test.test(feature_extractor, class_classifier, domain_classifier, src_test_dataloader, tgt_test_dataloader, best_auprc, args.objective, device, class_criterion)
        elif args.mode == 'test':
            auprc = test.test(feature_extractor, class_classifier, domain_classifier, src_test_dataloader, tgt_test_dataloader, best_auprc, args.objective, device, class_criterion)
        if auprc > best_auprc:
            best_auprc = auprc
            print("Best AUPRC")
            torch.save(feature_extractor.state_dict(), 'best_feature_extractor_' + args.source_domain + '_' + args.training_mode + '_exp.pt')
            torch.save(class_classifier.state_dict(), 'best_class_classifier_' + args.source_domain + '_' + args.training_mode + '_exp.pt')
            if args.training_mode == 'dann':
                torch.save(domain_classifier.state_dict(), 'best_domain_classifier_' + args.source_domain + '.pt')
            elif args.training_mode == 'wdgrl':
                torch.save(critic.state_dict(), 'best_critic_' + args.source_domain + '.pt')
            print("Saved Best Model")

        # Plot embeddings periodically.
        if epoch % params.embed_plot_epoch == 0 and params.fig_mode is not None:
            visualizePerformance(feature_extractor, class_classifier, domain_classifier, src_test_dataloader,
                                 tgt_test_dataloader, imgName = args.source_domain + '_' + args.training_mode + '_' + str(epoch) + '.2')

def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_domain', type= str, default= 'Human', help= 'Choose source domain.')
    parser.add_argument('--target_domain', type= str, default= 'Mouse', help = 'Choose target domain.')
    parser.add_argument('--fig_mode', type=str, default='save', help='Plot experiment figures.')
    parser.add_argument('--save_dir', type=str, default='/srv/scratch/soumyak/outputs/', help='Path to save plotted images.')
    parser.add_argument('--training_mode', type=str, default='source', help='Choose a mode to train the model.')
    parser.add_argument('--max_epoch', type=int, default=15, help='The max number of epochs.')
    parser.add_argument('--embed_plot_epoch', type= int, default=10, help= 'Epoch number of plotting embeddings.')
    parser.add_argument('--lr', type= float, default= 0.001, help= 'Learning rate.')
    parser.add_argument('--train_upsample', type=int, default=2, help='Set frequency of sampling positive training examples.')
    parser.add_argument('--test_upsample', type=int, default=0, help='Set frequency of sampling positive testing examples.')
    parser.add_argument('--num_train', type=int, default=100000, help='Number of training examples per epoch.')
    parser.add_argument('--num_test', type=int, default=100000, help='Number of testing examples per epoch.')
    parser.add_argument('--train_batch_size', type=int, default=500, help='Set training mini-batch size.')
    parser.add_argument('--test_batch_size', type=int, default=500, help='Set testing mini-batch size.')
    parser.add_argument('--objective', default='target', help='Select source or target as the objective to maximize its AUPRC.')
    parser.add_argument('--mode', default='train', help='Set train or test mode.')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
