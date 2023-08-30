#--------------------------------------------------------------------------#
# Project: plankton_classif
# Script purpose: Generate classification reports.
# Date: 30/08/2023
# Author: Thelma Panaïotis
#--------------------------------------------------------------------------#

import glob
import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import argparse

parser = argparse.ArgumentParser(description='Script to generate classification reports.')
parser.add_argument('--predictions_path', type=str, help='Path to predictions')
args = parser.parse_args()

predictions_path = args.predictions_path

# Create dir to store reports
save_dir = 'perf'
os.makedirs(save_dir, exist_ok=True)

## Get detailed and group classes for all datasets
# Initiate empty dataframe
df_all = pd.DataFrame()

# List datasets
datasets = ['flowcam', 'ifcb', 'isiis', 'uvp6', 'zoocam', 'zooscan']

sheet_id = '1C57EPnnOljtFKWrkdvQj2-UNc0bg0PMHb6km0YhaGSw'

for dataset in datasets:
    
    # Generate sheet url
    sheet_name = dataset
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    
    # Reed sheet
    df = pd.read_csv(url, usecols=['taxon', 'level2', 'plankton'])
    
    # Add name of dataset
    df['dataset'] = dataset
    
    # Store with other datasets
    df_all = pd.concat([df_all, df]).reset_index(drop = True)

# Rename grouped column
df_all = df_all.rename(columns={'level2': 'grouped'})


# Models to be included in CR
# - RF on native features
# - Mob + MLP600
# - Eff S + MLP600
# - Mob + PCA + MLP600

# List files of predictions for deep models
files = glob.glob(os.path.join(predictions_path,'*.csv'))
files.sort()

# Get relevant files for each model
eff_files = [f for f in files if 'effnetv2s' in f]
mob600_files = [f for f in files if 'mobilenet600_' in f]
mob600_pca_files = [f for f in files if 'mobilenet_pca50_rf' in f]
nat_fr_files = [f for f in files if 'native_rf_p' in f]

# List with all relevant files
files = nat_fr_files + mob600_pca_files + mob600_files + eff_files

# For each file, find dataset and method
files_dict = {
    'path': [],
    'dataset': [],
    'method': [],
}

for file in files:
    files_dict['path'].append(file)
    dataset = [f for f in datasets if f in file][0]
    files_dict['dataset'].append(dataset)
    files_dict['method'].append('rf' if 'rf' in file else 'deep')


# Loop over datasets
for dataset in datasets:
    print(f'Processing {dataset}')
    
    # Get indexes of paths and methods for a given dataset
    ind = [i for i in range(len(files_dict['path'])) if files_dict['dataset'][i] == dataset]
    paths = np.array(files_dict['path'])[ind].tolist()
    
    # New dataframes for model
    cr_all = pd.DataFrame()
    cr_g_all = pd.DataFrame()
    
    for i, path in enumerate(paths):
        # Get model name
        model = os.path.split(path)[1].split('_', 1)[1].replace('_predictions.csv', '')
        print(f'Generating report for {model}')
    
        # Read file
        df = pd.read_csv(path, usecols=['label', 'predicted_label'])
        
        # Get labels and predictions
        y = df['label']
        y_pred = df['predicted_label']
        
        # Space fix for zooscan dataset
        if dataset == 'zooscan':
            y = [e.replace('\u202f', '') for e in y]
            y_pred = [e.replace('\u202f', '') for e in y_pred]
        
        # Get classes for this dataset
        df_ref = df_all[df_all['dataset'] == dataset].reset_index(drop = True).drop('dataset', axis = 1)
        
        # Plankton classes
        plankton_classes = list(set(df_ref[df_ref['plankton']]['taxon'].tolist()))
        plankton_classes_g = list(set(df_ref[df_ref['plankton']]['grouped'].tolist()))
        
        # Classification report
        # generate classification report as a dict
        cr = classification_report(y, y_pred, output_dict=True, zero_division = 0)
        # convert to dataframe
        cr = pd.DataFrame(cr).transpose()
        # drop unwanted columns
        cr = cr.drop(['support'], axis = 1)
        # rename columns with model name
        cr = cr.rename(columns={
            'precision': f'precision-{model}',
            'recall': f'recall-{model}',
            'f1-score': f'f1-{model}'
        })
        # move taxon from index to column
        cr = cr.rename_axis('taxon').reset_index()
        # drop averages
        cr = cr.drop(cr.tail(3).index)
        # store with other models
        if len(cr_all) == 0:
            cr_all = cr
        else:
            cr_all = pd.merge(cr_all, cr)
    
        # Generate dict for taxonomy match
        taxo_match = df_ref.set_index('taxon').to_dict('index')
        
        # Convert true classes to larger ecological classes
        y_g = np.array([taxo_match[t]['grouped'] for t in y])
        
        # Convert predicted classes to larger ecological classes
        y_pred_g = np.array([taxo_match[p]['grouped'] for p in y_pred])
        
        # Classification report grouped
        # generate classification report as a dict
        cr_g = classification_report(y_g, y_pred_g, output_dict=True, zero_division = 0)
        # convert to dataframe
        cr_g = pd.DataFrame(cr_g).transpose()
        # drop unwanted columns
        cr_g = cr_g.drop(['support'], axis = 1)
        # rename columns with model name
        cr_g = cr_g.rename(columns={
            'precision': f'precision-{model}',
            'recall': f'recall-{model}',
            'f1-score': f'f1-{model}'
        })
        # move taxon from index to column
        cr_g = cr_g.rename_axis('taxon').reset_index()
        # drop averages
        cr_g = cr_g.drop(cr_g.tail(3).index)
    
        # store with other models
        if len(cr_g_all) == 0:
            cr_g_all = cr_g
        else:
            cr_g_all = pd.merge(cr_g_all, cr_g)
    
    
    # Add grouped classes for detailed report
    cr_all = pd.merge(cr_all, df_ref[['taxon', 'grouped']])
    
    # Save classification reports
    print(f'Saving reports for {dataset}')
    cr_all.to_csv(os.path.join(save_dir, f'report_{dataset}_detailed.csv'), index=False)
    cr_g_all.to_csv(os.path.join(save_dir, f'report_{dataset}_grouped.csv'), index=False)

