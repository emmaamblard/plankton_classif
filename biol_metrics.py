import pandas as pd
import numpy as np
from sklearn import metrics

def classification_report(y_true, y_pred, y_prob, non_biol_classes=['detritus'], **kwargs):
    """
    Build a text report showing the main classification metrics.
    
    Args:
        y_true (1d array-like): Ground truth (correct) target values
        y_pred (1d array-like): Predicted target values
        y_prob (2d array-like): Predicted probabilities (output of the CNN)
        non_biol_classes (list of strings): Classes to exclude to compute
            statistics on biological classes only.
        **kwargs: Passed to sklearn.metrics.classification_report

    Returns:
        cr (pd.DataFrame): The classification report, as a DataFrame
    """

    # compute the classification report
    cr = metrics.classification_report(y_true=y_true, y_pred=y_pred,
             output_dict=True, **kwargs)
             
    # convert to DataFrame for printing and computation
    cr = pd.DataFrame(cr).transpose()
    
    # get only biological classes
    stats = ['accuracy', 'macro avg', 'weighted avg']
    biol_cr = cr[~cr.index.isin(non_biol_classes + stats)]
    
    # compute top-2 accuracy for all classes and add it to cr
    top2acc = metrics.top_k_accuracy_score(y_true, y_prob, k=2)
    cr.at['top-2 accuracy','f1-score'] = top2acc
    cr.at['top-2 accuracy', 'support'] = len(y_true)
    
    # compute top-3 accuracy for all classes and add it to cr
    top3acc = metrics.top_k_accuracy_score(y_true, y_prob, k=3)
    cr.at['top-3 accuracy','f1-score'] = top3acc
    cr.at['top-3 accuracy', 'support'] = len(y_true)
    
    # compute log_loss for all classes and add it to cr
    log_loss = metrics.log_loss(y_true, y_prob)
    cr.at['log loss','f1-score'] = log_loss
    cr.at['log loss', 'support'] = len(y_true)
    
    # reorder index to place top-2 accuracy, top-3 accuracy and log loss under accuracy
    new_index = cr.index[:-5].to_list() + ['top-2 accuracy', 'top-3 accuracy', 'log loss'] + cr.index[-5:-3].to_list()
    cr = cr.reindex(new_index)
    
    # compute stats for biological classes
    biol_macro_avg = biol_cr.apply(np.average)
    biol_weighted_avg = biol_cr.apply(np.average, weights=biol_cr.support)
    
    # reformat as DataFrame
    biol_stats = pd.concat([biol_macro_avg, biol_weighted_avg], axis=1)
    biol_stats.columns = ['biol macro avg', 'biol weighted avg']
    biol_stats = biol_stats.transpose()
    biol_stats.support = len(y_true)
    
    # add to the total classification report
    cr = pd.concat([cr, biol_stats])
    # and format it nicely
    cr['precision']['accuracy'] = np.float64('NaN')
    cr['recall']['accuracy'] = np.float64('NaN')
    cr['support']['accuracy'] = len(y_true)
    cr.support = cr.support.astype(int)
    cr = cr.round(4)
    
    return(cr)
