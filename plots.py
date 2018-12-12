import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from itertools import cycle
from sklearn.utils.fixes import signature

human_source_labels = np.load('human_source_labels.npy')
human_source_preds = np.load('human_source_preds.npy')

human_target_labels = np.load('human_target_labels.npy')
human_target_preds = np.load('human_target_preds.npy')

human_source_auprc = average_precision_score(human_source_labels, human_source_preds)
human_target_auprc = average_precision_score(human_target_labels, human_target_preds)

human_source_auroc = roc_auc_score(human_source_labels, human_source_preds)
human_target_auroc = roc_auc_score(human_target_labels, human_target_preds)

print("Human Source AUPRC: " + str(human_source_auprc))
print("Human Target AUPRC: " + str(human_target_auprc))

print("Human Source AUROC: " + str(human_source_auroc))
print("Human Target AUROC: " + str(human_target_auroc))

dann_source_labels = np.load('dann_source_labels.npy')
dann_source_preds = np.load('dann_source_preds.npy')

dann_target_labels = np.load('dann_target_labels.npy')
dann_target_preds = np.load('dann_target_preds.npy')

dann_source_auprc = average_precision_score(dann_source_labels, dann_source_preds)
dann_target_auprc = average_precision_score(dann_target_labels, dann_target_preds)

dann_source_auroc = roc_auc_score(dann_source_labels, dann_source_preds)
dann_target_auroc = roc_auc_score(dann_target_labels, dann_target_preds)

print("DANN Source AUPRC: " + str(dann_source_auprc))
print("DANN Target AUPRC: " + str(dann_target_auprc))

print("DANN Source AUROC: " + str(dann_source_auroc))
print("DANN Target AUROC: " + str(dann_target_auroc))

wdgrl_source_labels = np.load('wdgrl_source_labels.npy')
wdgrl_source_preds = np.load('wdgrl_source_preds.npy')

wdgrl_target_labels = np.load('wdgrl_target_labels.npy')
wdgrl_target_preds = np.load('wdgrl_target_preds.npy')

wdgrl_source_auprc = average_precision_score(wdgrl_source_labels, wdgrl_source_preds)
wdgrl_target_auprc = average_precision_score(wdgrl_target_labels, wdgrl_target_preds)

wdgrl_source_auroc = roc_auc_score(wdgrl_source_labels, wdgrl_source_preds)
wdgrl_target_auroc = roc_auc_score(wdgrl_target_labels, wdgrl_target_preds)

print("WDGRL Source AUPRC: " + str(wdgrl_source_auprc))
print("WDGRL Target AUPRC: " + str(wdgrl_target_auprc))

print("WDGRL Source AUROC: " + str(wdgrl_source_auroc))
print("WDGRL Target AUROC: " + str(wdgrl_target_auroc))

mouse_source_labels = np.load('mouse_source_labels.npy')
mouse_source_preds = np.load('mouse_source_preds.npy')

mouse_target_labels = np.load('mouse_target_labels.npy')
mouse_target_preds = np.load('mouse_target_preds.npy')

mouse_source_auprc = average_precision_score(mouse_source_labels, mouse_source_preds)
mouse_target_auprc = average_precision_score(mouse_target_labels, mouse_target_preds)

mouse_source_auroc = roc_auc_score(mouse_source_labels, mouse_source_preds)
mouse_target_auroc = roc_auc_score(mouse_target_labels, mouse_target_preds)

print("Mouse Source AUPRC: " + str(mouse_source_auprc))
print("Mouse Target AUPRC: " + str(mouse_target_auprc))

print("Mouse Source AUROC: " + str(mouse_source_auroc))
print("Mouse Target AUROC: " + str(mouse_target_auroc))


dann_mouse_source_labels = np.load('dann_mouse_source_labels.npy')
dann_mouse_source_preds = np.load('dann_mouse_source_preds.npy')

dann_mouse_target_labels = np.load('dann_mouse_target_labels.npy')
dann_mouse_target_preds = np.load('dann_mouse_target_preds.npy')

dann_mouse_source_auprc = average_precision_score(dann_mouse_source_labels, dann_mouse_source_preds)
dann_mouse_target_auprc = average_precision_score(dann_mouse_target_labels, dann_mouse_target_preds)

dann_mouse_source_auroc = roc_auc_score(dann_mouse_source_labels, dann_mouse_source_preds)
dann_mouse_target_auroc = roc_auc_score(dann_mouse_target_labels, dann_mouse_target_preds)

print("DANN Mouse Source AUPRC: " + str(dann_mouse_source_auprc))
print("DANN Mouse Target AUPRC: " + str(dann_mouse_target_auprc))

print("DANN Mouse Source AUROC: " + str(dann_mouse_source_auroc))
print("DANN Mouse Target AUROC: " + str(dann_mouse_target_auroc))

wdgrl_mouse_source_labels = np.load('wdgrl_mouse_source_labels.npy')
wdgrl_mouse_source_preds = np.load('wdgrl_mouse_source_preds.npy')

wdgrl_mouse_target_labels = np.load('wdgrl_mouse_target_labels.npy')
wdgrl_mouse_target_preds = np.load('wdgrl_mouse_target_preds.npy')

wdgrl_mouse_source_auprc = average_precision_score(wdgrl_mouse_source_labels, wdgrl_mouse_source_preds)
wdgrl_mouse_target_auprc = average_precision_score(wdgrl_mouse_target_labels, wdgrl_mouse_target_preds)

wdgrl_mouse_source_auroc = roc_auc_score(wdgrl_mouse_source_labels, wdgrl_mouse_source_preds)
wdgrl_mouse_target_auroc = roc_auc_score(wdgrl_mouse_target_labels, wdgrl_mouse_target_preds)

print("WDGRL Mouse Source AUPRC: " + str(wdgrl_mouse_source_auprc))
print("WDGRL Mouse Target AUPRC: " + str(wdgrl_mouse_target_auprc))

print("WDGRL Mouse Source AUROC: " + str(wdgrl_mouse_source_auroc))
print("WDGRL Mouse Target AUROC: " + str(wdgrl_mouse_target_auroc))

"""
hs_precision, hs_recall, _ = precision_recall_curve(wdgrl_source_labels, wdgrl_source_preds)

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(hs_recall, hs_precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(hs_recall, hs_precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Human Trained WDGRL on Human: AUPRC={0:0.5f}'.format(
          wdgrl_source_auprc))

plt.savefig('wdgrl_source_prc')
plt.close()
"""
