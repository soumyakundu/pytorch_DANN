import torch
from torch.autograd import Variable
import numpy as np
from train import params
from util import utils
import torch.optim as optim
from torch.autograd import grad


def gradient_penalty(critic, h_s, h_t, device):
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def train(training_mode, feature_extractor, class_classifier, domain_classifier, critic, class_criterion, domain_criterion,
          source_dataloader, target_dataloader, optimizer, epoch, device):

    feature_extractor.train()
    class_classifier.train()
    domain_classifier.train()

    start_steps = epoch * len(source_dataloader)
    total_steps = params.epochs * len(source_dataloader)

    for batch_idx, (sdata, tdata) in enumerate(zip(source_dataloader, target_dataloader)):

        if training_mode == 'dann':

            p = float(batch_idx + start_steps) / total_steps
            constant = 2. / (1. + np.exp(-params.gamma * p)) - 1

            input1, label1 = sdata
            input2, label2 = tdata

            label1 = label1.to(device)
            label2 = label2.to(device)
            input1 = input1.to(device)
            input2 = input2.to(device)

            source_labels = torch.zeros(label1.size()).type(torch.FloatTensor)
            target_labels = torch.ones(label2.size()).type(torch.FloatTensor)

            source_labels = source_labels.to(device)
            target_labels = target_labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                src_feature = feature_extractor(input1)
                tgt_feature = feature_extractor(input2)

                class_preds = class_classifier(src_feature)
                class_loss = class_criterion(class_preds, label1)

                tgt_preds = domain_classifier(tgt_feature, constant)
                src_preds = domain_classifier(src_feature, constant)

                tgt_loss = domain_criterion(tgt_preds, target_labels)
                src_loss = domain_criterion(src_preds, source_labels)
                domain_loss = tgt_loss + src_loss

                loss = class_loss + params.theta * domain_loss
                loss.backward()
                optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                    batch_idx * len(input2), len(target_dataloader.dataset),
                    100. * batch_idx / len(target_dataloader), loss.item(), class_loss.item(),
                    domain_loss.item()
                ))

        elif training_mode == 'source':

            input1, label1 = sdata

            label1 = label1.to(device)
            input1 = input1.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                src_feature = feature_extractor(input1)
                class_preds = class_classifier(src_feature)
                class_loss = class_criterion(class_preds, label1)

                class_loss.backward()
                optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(
                    batch_idx * len(input1), len(source_dataloader.dataset),
                    100. * batch_idx / len(source_dataloader), class_loss.item()
                ))

        elif training_mode == 'target':

            input2, label2 = tdata

            label2 = label2.to(device)
            input2 = input2.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                tgt_feature = feature_extractor(input2)
                class_preds = class_classifier(tgt_feature)
                class_loss = class_criterion(class_preds, label2)

                class_loss.backward()
                optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(
                    batch_idx * len(input2), len(target_dataloader.dataset),
                    100. * batch_idx / len(target_dataloader), class_loss.item()
                ))

        elif training_mode == 'wdgrl':

            input1, label1 = sdata
            input2, label2 = tdata

            label1 = label1.to(device)
            label2 = label2.to(device)
            input1 = input1.to(device)
            input2 = input2.to(device)

            source_labels = torch.zeros(label1.size()).type(torch.FloatTensor)
            target_labels = torch.ones(label2.size()).type(torch.FloatTensor)

            source_labels = source_labels.to(device)
            target_labels = target_labels.to(device)

            class_optim = optim.Adam([{'params': feature_extractor.parameters()},
                                      {'params': class_classifier.parameters()}], lr=params.lr)

            critic_optim = optim.Adam(critic.parameters(), lr=params.lr)

            set_requires_grad(feature_extractor, requires_grad=False)
            set_requires_grad(critic, requires_grad=True)

            with torch.no_grad():
                h_s = feature_extractor(input1)
                h_t = feature_extractor(input2)

            for i in range(5):
                gp = gradient_penalty(critic, h_s, h_t, device)

                critic_s = critic(h_s)
                critic_t = critic(h_t)
                wasserstein_distance = critic_s.mean() - critic_t.mean()

                critic_cost = -wasserstein_distance + 10*gp
                critic_optim.zero_grad()
                critic_cost.backward()
                critic_optim.step()

            set_requires_grad(feature_extractor, requires_grad=True)
            set_requires_grad(critic, requires_grad=False)

            for i in range(10):

                src_feature = feature_extractor(input1)
                tgt_feature = feature_extractor(input2)

                class_preds = class_classifier(src_feature)
                class_loss = class_criterion(class_preds, label1)
                wasserstein_distance = critic(src_feature).mean() - critic(tgt_feature).mean()

                loss = class_loss + 0.1 * wasserstein_distance
                class_optim.zero_grad()
                loss.backward()
                class_optim.step()

            if (batch_idx + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                    batch_idx * len(input2), len(target_dataloader.dataset),
                    100. * batch_idx / len(target_dataloader), loss.item() + critic_cost.item(), loss.item(),
                    critic_cost.item()
                ))
