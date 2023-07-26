import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
import gzip, time, random, math
from collections import OrderedDict
import re, os
import sklearn.model_selection
from numpy.random import default_rng
import torchmetrics
import linecache
import pytorch_lightning as pl
from operator import itemgetter
import zmq
import struct, glob
from scipy import interp
from itertools import cycle
import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, matthews_corrcoef, jaccard_score


IUPAC_VOCAB = OrderedDict([
    ("<pad>", 0),
    ("X", 1),
    ("<start>", 2),
    ("<stop>", 3),
    ("<mask>", 4),
    ("D", 5),
    ("E", 6),
    ("F", 7),
    ("G", 8),
    ("H", 9),
    ("I", 10),
    ("K", 11),
    ("L", 12),
    ("M", 13),
    ("N", 14),
    ("P", 15),
    ("Q", 16),
    ("R", 17),
    ("S", 18),
    ("T", 19),
    ("V", 20),
    ("W", 21),
    ("Y", 22),
    ("A", 23),
    ("C", 24),
    ])

int_to_aa = {value:key for key, value in IUPAC_VOCAB.items()}

PAD_IDX, UNK_IDX, START_IDX, STOP_IDX, MASK_IDX = 0, 1, 2, 3, 4
special_symbols = ['<pad>', 'X', '<start>', '<stop>', "<mask>"]


def readFasta(fastaFile):
    if fastaFile.split('.')[-1] == "gz":
        fh = gzip.open(fastaFile, 'rt')
    else:
        fh = open(fastaFile, 'r')
    name, seq = None, []
    for line in fh:
        line = line.rstrip()
        if line.startswith(">"):
            if name: yield (name, ''.join(seq))
            name, seq = line, []
        else:
            seq.append(line)
    if name: yield (name, ''.join(seq))


def parseCSV(csvFile):
    fh = open(csvFile, 'r')
    header = fh.readline()

    for line in fh:
        split_line = line.rstrip().split(',')
        seq = split_line[1]
        intensity = [float(j) for j in split_line[2:]]
        yield [seq, intensity]
    fh.close()


def tokenise(sequence: str):
    start_token = "<start>"
    stop_token = "<stop>"

    sequence = [re.sub(r"[UZOB*]", "X", x) for x in sequence]

    tokens = [x for x in sequence]
    tokens = [start_token] + tokens + [stop_token]

    tokenised = [IUPAC_VOCAB[x] for x in tokens]

    return np.array(tokenised, np.int32)


def pad_tokeised(tokenised, max_len, pad_idx=PAD_IDX):
    padded = np.full((max_len), pad_idx, np.int32)
    padded[:len(tokenised)] = tokenised
    return padded


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class oasDataset(Dataset):
    def __init__(self, oasFile, max_len, n_samples, task=None):
        self.max_len = max_len
        self.n_samples = n_samples
        self.oasFile = oasFile
        self.task = task
        self.rng = default_rng()
        self.mask_n = 0.15
        self.seq_data = self.setup()


    def setup(self):
        data = []

        local_rank = int(os.getenv("LOCAL_RANK", '0'))
        world_size = int(os.getenv("WORLD_SIZE", '0'))

        print("Rank %d/%d: Loading OAS dataset into memory." % (local_rank, world_size))
        seq_generator = readOAS(self.oasFile)

        i = 0
        for seq in seq_generator:
            if i >= self.n_samples:
                break

            if len(seq)+2 <= self.max_len and len(seq) >= 5:
                data.append(seq)
                i += 1
        print("Loaded %d OAS dataset samples." % (len(data)))

        return data


    def __getitem__(self, idx):
        seq = self.seq_data[idx]

        tokenised = tokenise(seq)
        masked = tokenised.copy()

        if self.task == "mlm":

            for i in range(len(masked[1:-1])):
                if self.rng.uniform() < 0.15: # Mask position at a 15% probability.
                    if self.rng.uniform() < 0.8: # Mask with a 80% prob.
                        masked[i] = MASK_IDX
                    elif self.rng.uniform() < 0.5: # Scramble pos at 10%.
                        new_token = self.rng.integers(5, len(IUPAC_VOCAB))
                        masked[i] = new_token

        masked = torch.LongTensor(pad_tokeised(masked, self.max_len))
        encoded = torch.LongTensor(pad_tokeised(tokenised, self.max_len))

        return (masked, encoded)

    def __len__(self):
        return self.n_samples


class oasDatasetIterable(Dataset):
    def __init__(self, max_len, n_samples, task=None):
        self.max_len = max_len
        self.n_samples = n_samples
        self.task = task
        self.rng = default_rng()
        self.mask_n = 0.15


        self.context = zmq.Context()
        print("Connecting to server...")
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5570")



    def __getitem__(self, idx):

        self.socket.send(struct.pack('i', self.max_len-2))

        seq = self.socket.recv().decode("utf-8")

        tokenised = tokenise(seq.rstrip())
        masked = tokenised.copy()

        if self.task == "mlm":

            for i in range(len(masked[1:-1])):
                if self.rng.uniform() < 0.15: # Mask position at a 15% probability.
                    if self.rng.uniform() < 0.8: # Mask with a 80% prob.
                        masked[i] = MASK_IDX
                    elif self.rng.uniform() < 0.5: # Scramble pos at 10%.
                        new_token = self.rng.integers(5, len(IUPAC_VOCAB))
                        masked[i] = new_token

        # if len(tokenised)+2 <= self.max_len and len(tokenised) >= 5:
        masked = torch.LongTensor(pad_tokeised(masked, self.max_len))
        encoded = torch.LongTensor(pad_tokeised(tokenised, self.max_len))

        # print(seq)
        return (masked, encoded)


    def __len__(self):
        return self.n_samples



class DSDataset(Dataset):
    def __init__(self, csvFile, max_len, n_samples, task=None, split='all', dataset="NNS", balance=None):
        self.max_len = max_len
        self.n_samples = n_samples
        self.csvFile = csvFile
        self.task = task
        self.rng = default_rng()
        self.n_classes = 3
        self.dataset = dataset

        self.data = []

        self.loadData()
        self.train_data, self.test_data = sklearn.model_selection.train_test_split(self.data, test_size=0.1, random_state=0, shuffle=True)

        if split == "train":
            self.data = self.train_data
            del self.test_data
        elif split == "test":
            self.data = self.test_data
            del self.train_data


        if balance != None:
            if balance == 'oversample':
                '''Oversample the under represented classes.'''
                oversampled = []
                weights = []
                total = len(self.data)
                class_data = np.array([x[1] for x in self.data], dtype=np.int)
                class_sum = np.count_nonzero(class_data, axis=0)
                for i in range(self.n_classes):
                    w = (1 / class_sum[i])*(total)/self.n_classes
                    if w == np.inf:
                        w = 0.
                    weights.append(w)


                weights = np.floor(weights)
                print(total, class_sum, weights)

                for x, y in self.data:
                    weight = int(weights[np.argmax(y)])
                    if weight > 0:
                        for i in range(weight):
                            oversampled.append([x, y])

                print(len(oversampled))

                self.data += oversampled



    def loadData(self):

        unique_seq_scores = {}
        print("Parsing csv data from file: %s" % (self.csvFile))
        c = 0
        for seq, targets in parseCSV(self.csvFile):

            ## Remove sequencing artefacts.
            if "PPPPPPP" in seq:
                continue
            if "KKKKKKK" in seq:
                continue
            if "GGGGGGG" in seq:
                continue

            if targets[0] >= 0.0:
                if seq in unique_seq_scores:
                    unique_seq_scores[seq].append(targets)
                else:
                    unique_seq_scores[seq] = [targets]
            c += 1

            if c >= self.n_samples:
                break

        for i, seq in enumerate(unique_seq_scores):
            targets = np.array(unique_seq_scores[seq])

            conc = np.max(targets, axis=0)

            tgt_condition = conc[8]


            peak_tgt = np.zeros(self.n_classes, dtype=np.int)

            if self.dataset == "NNS":
                if tgt_condition < 150.0:
                    peak_tgt[0] = 1
                if tgt_condition >= 150.0 and tgt_condition < 250.0:
                    peak_tgt[1] = 1
                if tgt_condition >= 250.0:
                    peak_tgt[2] = 1

            seq_composed = "MQVQLVQSGAEVKKPGESLKISCKGSGYSFTSYWIAWVRQMPGKGLEYMGLIYPGDSDTKYSPSFQGQVTISVDKSVSTAYLQWSSLKPSDSAVYFCAR%sGQGTLVTVSS" % (seq)
            self.data.append([seq_composed, peak_tgt])


    def getclassweights(self):
        weights = []
        total = len(self.data)

        class_data = np.array([x[1] for x in self.data], dtype=np.int)

        class_sum = np.count_nonzero(class_data, axis=0)
        print("Class counts: %s" % (class_sum))

        for i in range(self.n_classes):
            w = (1 / class_sum[i])*(total)/self.n_classes
            if w == np.inf:
                w = 0.
            weights.append(w)

        return weights


    def __getitem__(self, idx):
        seq, tgt = self.data[idx]
        tokenised = tokenise(seq)

        if len(tokenised) <= self.max_len and len(tokenised) >= 5:
            seq_encoded = torch.LongTensor(pad_tokeised(tokenised, self.max_len))
            tgt_encoded = torch.LongTensor(tgt)

            return (seq_encoded, tgt_encoded)
        else:
            ## If here, fetch the next item.
            return self.__getitem__(idx)

    def __len__(self):
        return len(self.data)


def token_to_seq(tokenised):
    seq = ""
    for tok in tokenised:
        try:
            if tok == MASK_IDX:
                seq += " "
            else:
                seq += int_to_aa[tok]
        except:
            seq += int_to_aa[UNK_IDX]
    return seq

def make_prediction_mlm(model, sequence, max_len=150, mask_pos=1):
    model.eval()
    with torch.no_grad():
        tokenised = tokenise(sequence)
        print(token_to_seq(tokenised))

        # mask some positions
        tokenised[mask_pos] = MASK_IDX
        # tokenised[30:40] = MASK_IDX

        print(token_to_seq(tokenised))

        src = torch.LongTensor([pad_tokeised(tokenised, max_len)])
        pred = model.forward(src).squeeze()
        pred = F.softmax(pred, dim=1)
        max_pred = torch.argmax(pred, dim=-1).detach().cpu().view(-1).numpy()

        print(token_to_seq(max_pred))


        ## For the mask position, print out AA probabilities.
        mask_probs = pred.detach().cpu().numpy()[mask_pos]
        aa_prob_list = []
        for i, p in enumerate(mask_probs):
            aa = int_to_aa[i]
            aa_prob_list.append([aa, p])

        aa_prob_list = sorted(aa_prob_list, key=itemgetter(1), reverse=True)

        for (aa, p) in aa_prob_list:
            print("%s = %f" % (aa, p))



def make_prediction_classify(model, sequence):
    max_len = 150
    model.eval()
    with torch.no_grad():
        tokenised = tokenise(sequence)
        print(token_to_seq(tokenised))

        # mask some positions
        # tokenised[23:25] = MASK_IDX

        # print(token_to_seq(tokenised))

        src = torch.LongTensor([pad_tokeised(tokenised, max_len)])
        # tgt = src.clone()
        pred = model.forward(src)

        print(torch.softmax(pred, dim=1))

        # max_pred = torch.argmax(pred, dim=-1).detach().cpu().view(-1).numpy()

        # print(token_to_seq(max_pred))




def plot_roc_auc(y_true, y_pred, n_classes, model_name, lw=2):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(f'figures/roc_{model_name}.pdf', dpi=300, bbox_inches='tight')


def compute_metrics(model, dataloader, model_name):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    preds = []
    labels = []

    # initialise metrics
    n_classes = 3
    metric_collection = torchmetrics.MetricCollection([
        torchmetrics.Accuracy(num_classes=n_classes),
        torchmetrics.F1Score(num_classes=n_classes, multiclass=True),
        torchmetrics.Precision(num_classes=n_classes, multiclass=True),
        torchmetrics.Recall(num_classes=n_classes, multiclass=True),
        torchmetrics.ConfusionMatrix(num_classes=n_classes),
        torchmetrics.StatScores(num_classes=n_classes, multiclass=True, reduce='macro', mdmc_reduce='global')
        ])

    model = model.to(device)

    model.eval()
    with torch.no_grad():
        for (src, tgt) in dataloader:
            src = src.to(device)
            pred = torch.softmax(model.forward(src), dim=1).cpu()

            metrics = metric_collection(pred, torch.max(tgt, axis=1)[1])

            preds.append(pred)
            labels.append(tgt)

    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    metrics = metric_collection.compute()

    pred_all = [1 * (x>=0.5) for x in preds]
    labels_all = [1 * (x>=0.5) for x in labels]
    cr = classification_report(labels_all, pred_all, zero_division=0)
    print(cr)


    csc = 0
    for cstats in metrics['StatScores']:
        print("Class %d: TP=%d, FP=%d, TN=%d, FN=%d, Support/count=%d" % (csc, cstats[0], cstats[1], cstats[2], cstats[3], cstats[4]))
        csc += 1


    # jaccard score
    jac = jaccard_score(labels_all, pred_all, average=None)
    print(f"Jaccard score: {jac}")

    # # MCC score
    # mcc = matthews_corrcoef(labels_all, preds)
    # print("MCC %f" % (mcc))


    # ROC AUC score
    rocauc = roc_auc_score(labels, preds, average=None)
    print(f"\nROC AUC: {rocauc}\n")

    macro_roc_auc_ovo = roc_auc_score(labels, preds, multi_class="ovo",
                                average="macro")
    weighted_roc_auc_ovo = roc_auc_score(labels, preds, multi_class="ovo",
                                average="weighted")
    macro_roc_auc_ovr = roc_auc_score(labels, preds, multi_class="ovr",
                                average="macro")
    weighted_roc_auc_ovr = roc_auc_score(labels, preds, multi_class="ovr",
                                average="weighted")
    print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
          "(weighted by prevalence)"
          .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
    print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
          "(weighted by prevalence)"
          .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

    plot_roc_auc(labels, preds, n_classes, model_name)



    return preds, labels

def measure_model_performance(model, train_set, test_set, model_name):

    print("Computing metrics on training set of %d samples." % (len(train_set)*train_set.batch_size))
    train_preds, train_labels = compute_metrics(model, train_set, f"{model_name}_train")

    print("Computing metrics on test set of %d samples." % (len(test_set)*test_set.batch_size))
    test_preds, test_labels = compute_metrics(model, test_set, f"{model_name}_test")

    # print(train_preds)


def load_unique_experimental_data(experimental_csv_file):
    print('Loading unique experimental data from: %s' % (experimental_csv_file))
    experimental_data = {}
    experimental_data_argmax = {}
    c = 0
    exp_idx = 7

    for seq, targets in parseCSV(experimental_csv_file):
        seq = seq
        if "PPPPPPP" in seq:
            continue
        if "KKKKKKK" in seq:
            continue
        if "GGGGGGG" in seq:
            continue

        if targets[0] >= 0.0:
            if seq in experimental_data:
                experimental_data[seq].append(targets)
            else:
                experimental_data[seq] = [targets]
        c += 1

    for i, seq in enumerate(experimental_data):
        targets = np.array(experimental_data[seq])
        conc = np.max(targets, axis=0)
        experimental_data_argmax[seq] = conc

    print('Loaded experimental data. %d uniques.' % (len(experimental_data_argmax)))

    return experimental_data_argmax




class ModelAccuracyDataset(Dataset):
    def __init__(self, csvFile, sequence_files, max_len):
        self.max_len = max_len
        self.csvFile = csvFile
        self.sequence_files = sequence_files
        self.n_classes = 3

        self.data = []

        self.loadData()


    def loadData(self):

        experimental_data_argmax = load_unique_experimental_data(self.csvFile)
        intensity_threshold = 200.0
        prediction_threshold = 0.5
        intensity_idx = 8

        sequence_files = glob.glob(self.sequence_files)

        for seq_file in sequence_files:
            for line in open(seq_file, "r"):
                seq, pred_score = line.rstrip().split(',')

                seq = seq+"W"

                if seq in experimental_data_argmax:
                    exp_score = experimental_data_argmax[seq][intensity_idx]

                    if exp_score > 0.0:
                        peak_tgt = np.zeros(self.n_classes, dtype=np.int)

                        if exp_score >= intensity_threshold:
                            peak_tgt[2] = 1
                        else:
                            peak_tgt[0] = 1

                        seq_composed = "MQVQLVQSGAEVKKPGESLKISCKGSGYSFTSYWIAWVRQMPGKGLEYMGLIYPGDSDTKYSPSFQGQVTISVDKSVSTAYLQWSSLKPSDSAVYFCAR%sGQGTLVTVSS" % (seq)
                        self.data.append([seq_composed, peak_tgt])



    def __getitem__(self, idx):
        seq, tgt = self.data[idx]
        tokenised = tokenise(seq)

        seq_encoded = torch.LongTensor(pad_tokeised(tokenised, self.max_len))
        tgt_encoded = torch.LongTensor(tgt)

        return (seq_encoded, tgt_encoded)


    def __len__(self):
        return len(self.data)




def calculate_model_accuracy(model, experimental_csv_file, sequence_files):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = model.to(device)


    dataset = ModelAccuracyDataset(experimental_csv_file, sequence_files, 150)
    dataloader = DataLoader(dataset, batch_size=512, num_workers=0, worker_init_fn=worker_init_fn)

    # initialise metrics
    n_classes = 3
    metric_collection = torchmetrics.MetricCollection([
        torchmetrics.Accuracy(num_classes=n_classes),
        torchmetrics.F1Score(num_classes=n_classes, multiclass=True),
        torchmetrics.Precision(num_classes=n_classes, multiclass=True),
        torchmetrics.Recall(num_classes=n_classes, multiclass=True),
        torchmetrics.ConfusionMatrix(num_classes=n_classes),
        torchmetrics.StatScores(num_classes=n_classes, multiclass=True, reduce='macro', mdmc_reduce='global')
        ])

    preds = []
    labels = []

    print(len(dataset))

    model.eval()
    with torch.no_grad():
        for (src, tgt) in dataloader:

            # print(src, tgt)
            src = src.to(device)
            pred = torch.softmax(model.forward(src), dim=1).cpu()

            metrics = metric_collection(pred, torch.max(tgt, axis=1)[1])

            preds.append(pred)
            labels.append(tgt)

    preds = torch.cat(preds, dim=0)

    # metrics = metric_collection.compute()
    # print(metrics)

    cr = classification_report(labels, preds)
    print(cr)

    csc = 0
    for cstats in metrics['StatScores']:
        print("Class %d: TP=%d, FP=%d, TN=%d, FN=%d, Support/count=%d" % (csc, cstats[0], cstats[1], cstats[2], cstats[3], cstats[4]))
        csc += 1


    cr = classification_report(y_true, y_pred)
    print(cr)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7,5))
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap=plt.cm.Blues, fmt='g')
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Non-hit', 'Hit'])
    ax.yaxis.set_ticklabels(['Non-hit', 'Hit'])
    plt.show(block=True)