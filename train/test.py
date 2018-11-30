"""
Test the model with target domain
"""
import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import average_precision_score

from train import params


def test(feature_extractor, class_classifier, domain_classifier, source_dataloader, target_dataloader):
    """
    Test the performance of the model
    :param feature_extractor: network used to extract feature from target samples
    :param class_classifier: network used to predict labels
    :param domain_classifier: network used to predict domain
    :param source_dataloader: test dataloader of source domain
    :param target_dataloader: test dataloader of target domain
    :return: None
    """
    # setup the network
    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()
    source_correct = 0.0
    target_correct = 0.0
    domain_correct = 0.0
    tgt_correct = 0.0
    src_correct = 0.0

    for batch_idx, sdata in enumerate(source_dataloader):
        # setup hyperparameters
        p = float(batch_idx) / len(source_dataloader)
        constant = 2. / (1. + np.exp(-10 * p)) - 1.

        input1, label1 = sdata

        label1 = torch.squeeze(label1)
        label1 = label1.type(torch.LongTensor)
        input1 = input1.type(torch.FloatTensor)

        if params.use_gpu:
            input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
            src_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor).cuda())
        else:
            input1, label1 = Variable(input1), Variable(label1)
            src_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor))

        output1 = class_classifier(feature_extractor(input1))
        pred1 = output1.data.max(1, keepdim = True)[1]
        source_correct += pred1.eq(label1.data.view_as(pred1)).cpu().sum()

        src_preds = domain_classifier(feature_extractor(input1), constant)
        src_preds = src_preds.data.max(1, keepdim= True)[1]
        src_correct += src_preds.eq(src_labels.data.view_as(src_preds)).cpu().sum()

        output1_np = output1.data.cpu().numpy()
        label1_np = label1.data.cpu().numpy()
        predicted_probs1 = np.asarray([i[1] for i in output1_np])
        auprc1 = average_precision_score(label1_np, predicted_probs1)

    for batch_idx, tdata in enumerate(target_dataloader):
        # setup hyperparameters
        p = float(batch_idx) / len(source_dataloader)
        constant = 2. / (1. + np.exp(-10 * p)) - 1

        input2, label2 = tdata

        label2 = torch.squeeze(label2)
        label2 = label2.type(torch.LongTensor)
        input2 = input2.type(torch.FloatTensor)

        if params.use_gpu:
            input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
            tgt_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor).cuda())
        else:
            input2, label2 = Variable(input2), Variable(label2)
            tgt_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor))

        output2 = class_classifier(feature_extractor(input2))
        pred2 = output2.data.max(1, keepdim=True)[1]
        target_correct += pred2.eq(label2.data.view_as(pred2)).cpu().sum()

        tgt_preds = domain_classifier(feature_extractor(input2), constant)
        tgt_preds = tgt_preds.data.max(1, keepdim=True)[1]
        tgt_correct += tgt_preds.eq(tgt_labels.data.view_as(tgt_preds)).cpu().sum()

        output2_np = output2.data.cpu().numpy()
        label2_np = label2.data.cpu().numpy()
        predicted_probs2 = np.asarray([i[1] for i in output2_np])
        auprc2 = average_precision_score(label2_np, predicted_probs2)

    domain_correct = tgt_correct + src_correct

    print('\nSource Accuracy: {}/{} ({:.4f}%)\nTarget Accuracy: {}/{} ({:.4f}%)\n'
          'Domain Accuracy: {}/{} ({:.4f}%)\nSource AUPRC: {:.4f}\n'
          'Target AUPRC: {:.4f}\n'.
        format(
        source_correct, len(source_dataloader.dataset), 100. * float(source_correct) / len(source_dataloader.dataset),
        target_correct, len(target_dataloader.dataset), 100. * float(target_correct) / len(target_dataloader.dataset),
        domain_correct, len(source_dataloader.dataset) + len(target_dataloader.dataset),
        100. * float(domain_correct) / (len(source_dataloader.dataset) + len(target_dataloader.dataset)),
        auprc1, auprc2
    ))
