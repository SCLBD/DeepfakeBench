import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn

def get_accracy(output, label):
    _, prediction = torch.max(output, 1)    # argmax
    correct = (prediction == label).sum().item()
    accuracy = correct / prediction.size(0)

    return accuracy


def get_prediction(output, label):
    prob = nn.functional.softmax(output, dim=1)[:, 1]
    prob = prob.view(prob.size(0), 1)
    label = label.view(label.size(0), 1)
    #print(prob.size(), label.size())
    datas = torch.cat((prob, label.float()), dim=1)
    return datas


def calculate_metrics(label, output):
    if output.size(1) == 2:
        prob = torch.softmax(output, dim=1)[:, 1]
    else:
        prob = output

    # Accuracy
    _, prediction = torch.max(output, 1)
    correct = (prediction == label).sum().item()
    accuracy = correct / prediction.size(0)

    # AUC and EER
    fpr, tpr, thresholds = metrics.roc_curve(label.squeeze().cpu().numpy(),
                                             prob.squeeze().cpu().numpy(),
                                             pos_label=1)
    if np.isnan(fpr[0]) or np.isnan(tpr[0]):
        auc, eer = -1, -1
    else:
        auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    # Average Precision
    y_true = label.cpu().detach().numpy()
    y_pred = prob.cpu().detach().numpy()
    ap = metrics.average_precision_score(y_true, y_pred)

    return auc, eer, accuracy, ap


# ------------ compute average metrics of batches---------------------
class Metrics_batches_mean():
    def __init__(self):
        self.tprs = []
        self.mean_fpr = np.linspace(0, 1, 100)
        self.aucs = []
        self.eers = []
        self.aps = []

        self.correct = 0
        self.total = 0

    def update(self, label, output):
        acc = self._update_acc(label, output)
        if output.size(1) == 2:
            prob = torch.softmax(output, dim=1)[:, 1]
        else:
            prob = output
        #label = 1-label
        #prob = torch.softmax(output, dim=1)[:, 1]
        auc, eer = self._update_auc(label, prob)
        ap = self._update_ap(label, prob)

        return acc, auc, eer, ap

    def _update_auc(self, lab, prob):
        fpr, tpr, thresholds = metrics.roc_curve(lab.squeeze().cpu().numpy(),
                                                 prob.squeeze().cpu().numpy(),
                                                 pos_label=1)
        if np.isnan(fpr[0]) or np.isnan(tpr[0]):
            return -1, -1

        auc = metrics.auc(fpr, tpr)
        interp_tpr = np.interp(self.mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        self.tprs.append(interp_tpr)
        self.aucs.append(auc)

        # return auc

        # EER
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        self.eers.append(eer)

        return auc, eer

    def _update_acc(self, lab, output):
        _, prediction = torch.max(output, 1)    # argmax
        correct = (prediction == lab).sum().item()
        accuracy = correct / prediction.size(0)
        # self.accs.append(accuracy)
        self.correct = self.correct+correct
        self.total = self.total+lab.size(0)
        return accuracy

    def _update_ap(self, label, prob):
        y_true = label.cpu().detach().numpy()
        y_pred = prob.cpu().detach().numpy()
        ap = metrics.average_precision_score(y_true,y_pred)
        self.aps.append(ap)

        return np.mean(ap)


    def get_mean_metrics(self):
        mean_acc, std_acc = self.correct/self.total, 0
        mean_auc, std_auc = self._mean_auc()
        mean_err, std_err = np.mean(self.eers), np.std(self.eers)
        mean_ap, std_ap = np.mean(self.aps), np.std(self.aps)

        return mean_acc, std_acc, mean_auc, std_auc, mean_err, std_err, mean_ap, std_ap

    def _mean_auc(self):
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.aucs)
        return mean_auc, std_auc

    def clear(self):
        self.tprs.clear()
        self.aucs.clear()
        # self.accs.clear()
        self.correct=0
        self.total=0
        self.eers.clear()
        self.aps.clear()


class Metrics_all():
    def __init__(self):
        self.probs = []
        self.labels = []

    def store(self, label, output):
        prob = torch.softmax(output, dim=1)[:, 1]
        self.labels.append(label.squeeze().cpu().numpy())
        self.probs.append(prob.squeeze().cpu().numpy())


    def get_metrics(self):
        y_pred = np.concatenate(self.probs)
        y_true = np.concatenate(self.labels)
        fpr, tpr, thresholds = metrics.roc_curve(y_true,y_pred,pos_label=1)
        auc = metrics.auc(fpr, tpr)

        # EER
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        ap = metrics.average_precision_score(y_true,y_pred)

        return 0, 0, auc, 0, eer, 0, ap, 0

    def clear(self):
        self.probs.clear()
        self.labels.clear()