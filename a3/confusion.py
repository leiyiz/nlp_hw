import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools

title = 'Confusion matrix'

if len(sys.argv) is not 2:
    print('usage: python3 confusion.py PREDICTOR_OUTPUT_FILE_PATH')
    exit()
    pass

path = sys.argv[1]
with open(path, 'r') as f:
    pred_lines = f.readlines()
with open('test-gold-labels.tsv', 'r') as f:
    true_lines = f.readlines()

true_labels, pred_labels = [], []
for line in pred_lines:
    for label in line.split('\t')[1].strip().split():
        pred_labels.append(label)

for line in true_lines:
    for label in line.split('\t')[1].strip().split():
        true_labels.append(label)

all_poss_labels = set(true_labels)

rec = {}
i = 0
for label in all_poss_labels:
    rec[label] = i
    i += 1

confusion_matrix = np.zeros((len(all_poss_labels), len(all_poss_labels)))
for i in range(len(true_labels)):
    confusion_matrix[rec[true_labels[i]]][rec[pred_labels[i]]] += 1


plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(all_poss_labels))
plt.xticks(tick_marks, all_poss_labels, rotation=45)
plt.yticks(tick_marks, all_poss_labels)

fmt = 'd'
thresh = confusion_matrix.max() / 2.
for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
    plt.text(j, i, format(confusion_matrix[i, j], fmt),
             horizontalalignment="center",
             color="white" if confusion_matrix[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
