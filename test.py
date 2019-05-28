import yaml
import argparse

import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from tqdm import tqdm

from models import encoders

from data.datasets import insightface
from data.transform import Transforms

from utils.storage import load_weights


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_subset(container, subset_bounds):
    subset = []
    for bound in subset_bounds:
        subset += container[bound[0]: bound[1]]
    return subset


def get_roc(distances_with_gt, n_threshs=400):
    thresholds = np.linspace(0., 4., n_threshs)

    fp_rates = []
    tp_rates = []

    for threshold in thresholds:
        fp = 0
        tp = 0
        for distance_with_gt in distances_with_gt:
            predict_same = distance_with_gt['distance'] < threshold
            actual_same = distance_with_gt['label']

            if predict_same and actual_same:
                tp += 1
            elif predict_same and not actual_same:
                fp += 1

        fp_rates.append(float(fp) / len(distances_with_gt) * 2)
        tp_rates.append(float(tp) / len(distances_with_gt) * 2)

    return np.array(fp_rates), np.array(tp_rates)


def get_auc(fprs, tprs):
    sorted_fprs, sorted_tprs = zip(*sorted(zip(*(fprs, tprs))))
    sorted_fprs = list(sorted_fprs)
    sorted_tprs = list(sorted_tprs)
    if sorted_fprs[-1] != 1.0:
        sorted_fprs.append(1.0)
        sorted_tprs.append(sorted_tprs[-1])
    return np.trapz(sorted_tprs, sorted_fprs)


def get_distances(data_loader, model, flip_image=False, 
               distance=lambda x, y: 1. - F.cosine_similarity(x, y)):
    model.eval()
    distances_with_gt = []
    
    for i, (image1, image2, label) in enumerate(tqdm(data_loader)):
        image1 = image1.to(device)
        image2 = image2.to(device)
        
        embedding1 = model(image1)
        embedding2 = model(image2)
        if flip_image:
            image1_flipped = image1.flip(3)
            image2_flipped = image2.flip(3)
            
            embedding1_flipped = model(image1_flipped)
            embedding2_flipped = model(image2_flipped)
            
            embedding1 = (embedding1 + embedding1_flipped) * 0.5
            embedding2 = (embedding2 + embedding2_flipped) * 0.5

        scores = distance(embedding1, embedding2).data.cpu().numpy()

        for i, _ in enumerate(scores):
            distances_with_gt.append({'distance': scores[i], 'label': label[i]})

    return distances_with_gt


def get_optimal_thresh(distances_with_gt):
    pos_scores = []
    neg_scores = []
    for distance_with_gt in distances_with_gt:
        if distance_with_gt['label']:
            pos_scores.append(distance_with_gt['distance'])
        else:
            neg_scores.append(distance_with_gt['distance'])

    hist_pos, bins = np.histogram(np.array(pos_scores), 60)
    hist_neg, _ = np.histogram(np.array(neg_scores), bins)

    intersection_bins = []

    for i in range(1, len(hist_neg)):
        if hist_pos[i - 1] >= hist_neg[i - 1] and 0.05 < hist_pos[i] <= hist_neg[i]:
            intersection_bins.append(bins[i])

    if not intersection_bins:
        intersection_bins.append(0.5)

    return np.mean(intersection_bins)


def validate(data_loader, model, folds=10, flip_image=False):
    distances_with_gt = get_distances(data_loader, model, flip_image)
    num_pairs = len(distances_with_gt)

    subsets = []
    for i in range(folds):
        lower_bnd = i * num_pairs // 10
        upper_bnd = (i + 1) * num_pairs // 10
        subset_test = [(lower_bnd, upper_bnd)]
        subset_train = [(0, lower_bnd), (upper_bnd, num_pairs)]
        subsets.append({'test': subset_test, 'train': subset_train})

    same_scores = []
    diff_scores = []
    val_scores = []
    threshs = []
    mean_fpr = np.zeros(400)
    mean_tpr = np.zeros(400)

    for subset in tqdm(subsets):
        train_list = get_subset(distances_with_gt, subset['train'])
        optimal_thresh = get_optimal_thresh(train_list)
        threshs.append(optimal_thresh)

        test_list = get_subset(distances_with_gt, subset['test'])
        same_correct = 0
        diff_correct = 0
        pos_pairs_num = neg_pairs_num = len(test_list) // 2

        for distance_with_gt in test_list:
            if distance_with_gt['distance'] < optimal_thresh and distance_with_gt['label']:
                same_correct += 1
            elif distance_with_gt['distance'] >= optimal_thresh and not distance_with_gt['label']:
                diff_correct += 1

        same_scores.append(float(same_correct) / pos_pairs_num)
        diff_scores.append(float(diff_correct) / neg_pairs_num)
        val_scores.append(0.5 * (same_scores[-1] + diff_scores[-1]))

        fprs, tprs = get_roc(test_list, mean_fpr.shape[0])
        mean_fpr = mean_fpr + fprs
        mean_tpr = mean_tpr + tprs

    mean_fpr /= 10
    mean_tpr /= 10

    same_acc = np.mean(same_scores)
    diff_acc = np.mean(diff_scores)
    overall_acc = np.mean(val_scores)
    auc = get_auc(mean_fpr, mean_tpr)

    print('Same accuracy mean: {0:.4f}'.format(same_acc),
          'Diff accuracy mean: {0:.4f}'.format(diff_acc),
          'Accuracy mean: {0:.4f}'.format(overall_acc),
          'Accuracy/Val_accuracy std dev: {0:.4f}'.format(np.std(val_scores)),
          'AUC: {0:.4f}'.format(auc),
          'Estimated threshold: {0:.4f}'.format(np.mean(threshs)))

    return same_acc, diff_acc, overall_acc, auc


def main(config):
    model = getattr(encoders, config['model_name'])(out_features=config['features'],
                                                    device=device)
    
    if 'weights' in config['validataion']:
        model.load_state_dict(torch.load(config['validataion']['weights'])['state_dict'])
    else:
        load_weights(model, config['prefix'], 'model', config['validataion']['epoch'])
    
    transforms = Transforms(input_size=config['input_size'], train=False)
    
    for dataset in config['validataion']['dataset']:
        print(dataset.upper())
        data_loader = DataLoader(insightface.Test(folder=config['validataion']['folder'],
                                                  dataset=dataset,
                                                  transforms=transforms),
                                 batch_size=config['validataion']['batch_size'], 
                                 num_workers=1,
                                 shuffle=False)
    
        validate(data_loader, model, flip_image=config['validataion']['flip'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train code')
    parser.add_argument('--config', required=True, help='configuration file')
    args = parser.parse_args()
    
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    main(config)