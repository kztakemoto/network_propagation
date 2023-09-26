import pickle
import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def outputs(method, results, nb_proteins, nb_PD_pairs, args):
    # save results
    if method == "prince":
        filename = 'results_prince_alpha{}_maxiter{}_eps{}.pickle'.format(args.alpha, args.max_iter, args.eps)
    elif method == "minprop2":
        filename = 'results_minprop2_alphaP{}_alphaD{}_maxiter{}_eps{}.pickle'.format(args.alphaP, args.alphaD, args.max_iter, args.eps)
    else:
        raise ValueError("invalid method")

    with open(filename, 'wb') as f:
        pickle.dump(results, f)

    # get results
    rankings, roc_value_set = zip(*results)

    average_auc = np.mean(roc_value_set)
    print("Average AUC", average_auc)

    ### compute sensitivity and top rate (ROC-like curve)
    # top rate
    top_rate_set = np.linspace(1 / nb_proteins, 1, nb_proteins)
    # Compute sensitivity values
    cumulative_sens = np.cumsum(np.bincount(rankings, minlength=nb_proteins+1))
    sen_set = cumulative_sens[:nb_proteins] / nb_PD_pairs

    summarized_auc = auc(top_rate_set, sen_set)
    print("Summarized AUC", summarized_auc)

    # plot ROC-like curve
    plt.plot(top_rate_set, sen_set, label='ROC-like curve', linewidth=3)
    plt.xlabel('Top Rate')
    plt.ylabel('Sensitivity')
    plt.text(0.6, 0.2, f'Average AUC: {average_auc:.2%}', fontsize=10, bbox=dict(facecolor='white'))
    plt.text(0.6, 0.1, f'Summarized AUC: {summarized_auc:.2%}', fontsize=10, bbox=dict(facecolor='white'))
    plt.title('ROC-like Curve with AUC values')
    plt.grid(True)

    if method == "prince":
        filename = 'roc_prince_alpha{}_maxiter{}_eps{}.png'.format(args.alpha, args.max_iter, args.eps)
    elif method == "minprop2":
        filename = 'roc_minprop2_alphaP{}_alphaD{}_maxiter{}_eps{}.png'.format(args.alphaP, args.alphaD, args.max_iter, args.eps)
    else:
        raise ValueError("invalid method")

    plt.savefig(filename)
