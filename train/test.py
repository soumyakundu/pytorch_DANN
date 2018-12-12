"""
Test the model with target domain
"""
import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import average_precision_score
from train import params
from train.accuracy_metrics import *

def test(feature_extractor, class_classifier, domain_classifier, source_dataloader, target_dataloader, auprc, objective, device, class_criterion):

    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()

    source_correct = 0.0
    target_correct = 0.0
    domain_correct = 0.0
    src_correct = 0.0
    tgt_correct = 0.0
    source_labels = []
    source_preds = []
    target_labels = []
    target_preds = []
    source_pos = 0
    source_neg = 0
    source_TP = 0
    source_TN = 0
    target_pos = 0
    target_neg = 0
    target_TP = 0
    target_TN = 0

    source_loss = 0.0
    target_loss = 0.0

    for batch_idx, sdata in enumerate(source_dataloader):

        with torch.no_grad():

            p = float(batch_idx) / len(source_dataloader)
            constant = 2. / (1. + np.exp(-10 * p)) - 1.

            input1, label1 = sdata

            label1 = label1.to(device)
            input1 = input1.to(device)

            src_labels = torch.zeros(label1.size()).type(torch.FloatTensor)
            src_labels = src_labels.to(device)

            output1 = class_classifier(feature_extractor(input1))
            source_correct += torch.round(output1).eq(label1.data.view_as(output1)).cpu().sum()

            source_loss += class_criterion(output1, label1).item()

            src_preds = domain_classifier(feature_extractor(input1), constant)
            src_correct += torch.round(src_preds).eq(src_labels.data.view_as(src_preds)).cpu().sum()

            output1_np = output1.data.cpu().numpy()
            label1_np = label1.data.cpu().numpy()

            output1_np = [i[0] for i in output1_np]
            label1_np = [i[0] for i in label1_np]

            source_labels.extend(label1_np)
            source_preds.extend(output1_np)

            output1_np = np.round_(output1_np)
            label1_np = np.round_(label1_np)

            source_pos += np.sum(label1_np == 1)
            source_neg += np.sum(label1_np == 0)
            source_TP += np.sum(np.logical_and(output1_np == 1, label1_np == 1))
            source_TN += np.sum(np.logical_and(output1_np == 0, label1_np == 0))

            if (batch_idx + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(
                    batch_idx * len(input1), len(source_dataloader.dataset),
                    100. * batch_idx / len(source_dataloader), source_loss / 10
                ))
                source_loss = 0.0

    for batch_idx, tdata in enumerate(target_dataloader):

        with torch.no_grad():

            p = float(batch_idx) / len(source_dataloader)
            constant = 2. / (1. + np.exp(-10 * p)) - 1

            input2, label2 = tdata

            label2 = label2.to(device)
            input2 = input2.to(device)

            tgt_labels = torch.ones(label2.size()).type(torch.FloatTensor)
            tgt_labels = tgt_labels.to(device)

            output2 = class_classifier(feature_extractor(input2))
            target_correct += torch.round(output2).eq(label2.data.view_as(output2)).cpu().sum()

            target_loss += class_criterion(output2, label2).item()

            tgt_preds = domain_classifier(feature_extractor(input2), constant)
            tgt_correct += torch.round(tgt_preds).eq(tgt_labels.data.view_as(tgt_preds)).cpu().sum()

            output2_np = output2.data.cpu().numpy()
            label2_np = label2.data.cpu().numpy()

            output2_np = [i[0] for i in output2_np]
            label2_np = [i[0] for i in label2_np]

            target_labels.extend(label2_np)
            target_preds.extend(output2_np)

            output2_np = np.round_(output2_np)
            label2_np = np.round_(label2_np)

            target_pos += np.sum(label2_np == 1)
            target_neg += np.sum(label2_np == 0)
            target_TP += np.sum(np.logical_and(output2_np == 1, label2_np == 1))
            target_TN += np.sum(np.logical_and(output2_np == 0, label2_np == 0))

            if (batch_idx + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(
                    batch_idx * len(input2), len(target_dataloader.dataset),
                    100. * batch_idx / len(target_dataloader), target_loss / 10
                ))
                target_loss = 0.0

    domain_correct = tgt_correct + src_correct

    source_labels = np.array(source_labels)
    source_preds = np.array(source_preds)

    target_labels = np.array(target_labels)
    target_preds = np.array(target_preds)

    np.save('dann_mouse_source_labels.npy', source_labels)
    np.save('dann_mouse_target_labels.npy', target_labels)
    np.save('dann_mouse_source_preds.npy', source_preds)
    np.save('dann_mouse_target_preds.npy', target_preds)

    source_auprc = average_precision_score(source_labels, source_preds)
    target_auprc = average_precision_score(target_labels, target_preds)

    source_TPR = source_TP / source_pos
    source_TNR = source_TN / source_neg
    target_TPR = target_TP / target_pos
    target_TNR = target_TN / target_neg

    print('\nSource Accuracy: {}/{} ({:.4f}%)\nTarget Accuracy: {}/{} ({:.4f}%)\n'
          'Domain Accuracy: {}/{} ({:.4f}%)\nSource AUPRC: {:.4f}\n'
          'Target AUPRC: {:.4f}\nSource Positive Accuracy: {}/{} ({:.4f}%)\nSource Negative Accuracy: {}/{} ({:.4f}%)\n'
          'Target Positive Accuracy: {}/{} ({:.4f}%)\nTarget Negative Accuracy: {}/{} ({:.4f}%)\n'.
        format(
        source_correct, len(source_dataloader.dataset), 100. * float(source_correct) / len(source_dataloader.dataset),
        target_correct, len(target_dataloader.dataset), 100. * float(target_correct) / len(target_dataloader.dataset),
        domain_correct, len(source_dataloader.dataset) + len(target_dataloader.dataset),
        100. * float(domain_correct) / (len(source_dataloader.dataset) + len(target_dataloader.dataset)),
        source_auprc, target_auprc, source_TP, source_pos, 100 * source_TPR, source_TN, source_neg, 100 * source_TNR, target_TP, target_pos, 100 * target_TPR, target_TN, target_neg, 100 * target_TNR
    ))

    if objective == 'source':
        return source_auprc
    else:
        return target_auprc
