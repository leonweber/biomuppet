import argparse
from plot_results_per_task import compute_ner_f1, get_label_vectors
from seqeval.metrics import classification_report as seqeval_classification_report
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scipy.stats import pearsonr
import numpy as np

def compute_ner_report(pred_file, true_file):
    all_labels_pred = []
    with open(pred_file) as f:
        labels_pred = [] 
        for line in f:
            line = line.strip()
            if not line:
                all_labels_pred.append(labels_pred)
                labels_pred = []
            else:
                label = line.split("\t")[-1].strip()
                # labels_pred.append(label.split("-")[0])
                labels_pred.append(label)

    all_labels_true = []
    with open(true_file) as f:
        labels_true = [] 
        for line in f:
            line = line.strip()
            if not line:
                all_labels_true.append(labels_true)
                labels_true = []
            else:
                label = line.split("\t")[-1].strip()
                # labels_true.append(label.split("-")[0])
                labels_true.append(label)
    return seqeval_classification_report(all_labels_true, all_labels_pred, digits=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--true_file", type=str, required=True)
    parser.add_argument("--type", type=str, required=True, choices=["ner_report", "f1", "accuracy", "pearson"])
    args = parser.parse_args()

    if args.type == "ner_report":
        print(compute_ner_report(args.pred_file, args.true_file))
    elif args.type == "f1":
        labels_pred, labels_true, encoder = get_label_vectors(args.pred_file, args.true_file)
        labels = [i for i, class_ in enumerate(encoder.classes_) if class_ in {'CPR:3', 'CPR:4', 'CPR:5', 'CPR:6', 'CPR:9'}]
        print(classification_report(labels_true, labels_pred, labels=labels, digits=4))
    elif args.type == "accuracy":
        labels_pred, labels_true, encoder = get_label_vectors(args.pred_file, args.true_file)
        print(accuracy_score(labels_true, labels_pred))
    elif args.type == "pearson":
        labels_pred, labels_true, encoder = get_label_vectors(args.pred_file, args.true_file)
        print(pearsonr(labels_pred, labels_true))