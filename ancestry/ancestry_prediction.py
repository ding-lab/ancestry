import logging
import os
import re
import yaml

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def get_ancestry_to_color(ancestries):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'gray', 'black', 'pink', 'brown']
    ancestry_to_color = {}
    for i, k in enumerate(ancestries):
        color_index = i % len(colors)
        ancestry_to_color[k] = colors[color_index]

    return ancestry_to_color

def get_ancestry_vcf_df(vcf_fp, stop_on='#CHROM', keep_only=None):
    f = open(vcf_fp)

    line = ''
    header = ''
    while True:
        line = f.readline()
        if line[:6] == stop_on:
            break

    columns = line[1:].strip().replace('""', '').split('\t')[9:]
    
    index_data_tups = []
    already_seen = set()
    for i, line in enumerate(f):
        pieces = line.strip().split('\t')
        r_id = pieces[0] + ':' + pieces[1] + ':' + pieces[3] + ':' + pieces[4]

        if r_id not in already_seen:
            already_seen.add(r_id)
            index_data_tups.append((r_id, pieces[9:]))

    index_data_tups = sorted(index_data_tups, key=lambda x: x[0])
    index_data_tups = [(i, ls) for i, ls in index_data_tups
            if ':X:' not in i and ':chrX:' not in i]

    index, data = zip(*index_data_tups)
    data = np.asarray(data)

    df = pd.DataFrame(data=data, columns=columns, index=index)

    df.index.name = ''

    # transpose dataframe so samples are rows, mutations are columns
    df = df.transpose()
    
    # replace phased calls
#    df = df.replace(re.compile(r'^1\|0'), '0|1')

    sample_ids = list(df.index)
    
    f.close()
    
    return df, sample_ids

def get_ancestry_map(map_fp, super_population=True):
    f = open(map_fp)
    
    # dump header
    f.readline()
    
    ancestry_map = {}
    for line in f:
        if super_population:
            sample_id, _, ancestry, _ = line.strip().split('\t')
        else:
            sample_id, ancestry, _, _ = line.strip().split('\t')
        
        ancestry_map[sample_id] = ancestry
        
    
    return ancestry_map

def get_columns_to_drop(df, max_missingness=.05):
    to_drop = []
    for i, c in enumerate(df.columns):
        missing_count = len([x for x in df[c] if x == '.|.'])
        if missing_count / df.shape[0] > .05:
            to_drop.append(c)

    return to_drop

def create_dfs(thousand_genomes_vcf_fp, sample_vcf_fp, stats_dict=None):
    sample_df, sample_ids = get_ancestry_vcf_df(sample_vcf_fp)

    thousand_genomes_df, thousand_genomes_sample_ids = get_ancestry_vcf_df(thousand_genomes_vcf_fp,
            keep_only=sample_df.columns)

    if list(sample_df.columns) != list(thousand_genomes_df.columns):
        logging.warning(f'sample dataframe and thousand genomes dataframe do not have the same\
 variants: sample_df-{sample_df.shape}, thousand_genomes_df-{thousand_genomes_df.shape}. \
 trimming down.')
        thousand_genomes_df = thousand_genomes_df[sample_df.columns]
        logging.info(f'new thousand genomes shape is {thousand_genomes_df.shape}')
#         raise RuntimeError(f'sample dataframe and thousand genomes dataframe do not have the same\
#  variants: sample_df-{sample_df.shape}, thousand_genomes_df-{thousand_genomes_df.shape}')

    if stats_dict is not None:
        stats_dict['inputs_before_drop'] = {
            'thousand_genomes_num_samples': thousand_genomes_df.shape[0],
            'thousand_genomes_num_variants': thousand_genomes_df.shape[1],
            'run_num_samples': sample_df.shape[0],
            'run_num_variants': sample_df.shape[1]
        }

    to_drop = get_columns_to_drop(sample_df, max_missingness=.05)

    if stats_dict is not None:
        stats_dict['percent_of_variants_dropped'] = len(to_drop) / sample_df.shape[1]
        stats_dict['num_variants_dropped'] = len(to_drop)

    thousand_genomes_df = thousand_genomes_df.drop(to_drop, axis=1)
    sample_df = sample_df.drop(to_drop, axis=1)

    if stats_dict is not None:
        stats_dict['inputs_after_drop'] = {
            'thousand_genomes_num_samples': thousand_genomes_df.shape[0],
            'thousand_genomes_num_variants': thousand_genomes_df.shape[1],
            'run_num_samples': sample_df.shape[0],
            'run_num_variants': sample_df.shape[1]
        }

    return thousand_genomes_df, sample_df

def create_target_df(thousand_genomes_df, thousand_genomes_panel_fp, return_map=True,
        super_population=True):
    # read in ancestries for samples
    sample_id_to_ancestry = get_ancestry_map(thousand_genomes_panel_fp,
            super_population=super_population)

    # grab our target variable
    ancestries = [sample_id_to_ancestry[sample_id] for sample_id in thousand_genomes_df.index]

    target_df = pd.DataFrame.from_dict({
        'ancestry': ancestries
    })
    target_df.index = thousand_genomes_df.index

    return target_df, sample_id_to_ancestry

def plot_components(pcs, sample_ids, sample_id_to_ancestry, n=5,
        prefix='output', output_dir=os.getcwd()):
    labels = [f'PC{i}' for i in range(1, n + 1)]

    plotting_df = pd.DataFrame(data=pcs[:, :n], columns=labels)

    ancestry_to_color = get_ancestry_to_color(set(list(sample_id_to_ancestry.values())))

    colors = [ancestry_to_color[sample_id_to_ancestry[s_id]]
            for s_id in sample_ids]

    axs = pd.plotting.scatter_matrix(plotting_df, color=colors,
            figsize=(12,12), diagonal='kde')

    for subaxis in axs:
        for ax in subaxis:
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

    plt.savefig(os.path.join(output_dir, f'{prefix}.pdf'), dpi=300, figsize=(12,12))
    plt.savefig(os.path.join(output_dir, f'{prefix}.png'), dpi=300, figsize=(12,12))

def write_predictions_file(output_fp, sample_ids, predictions, probs, classes):
    out_f = open(output_fp, 'w')
    labels = ['probability_' + c for c in classes]
    out_f.write('sample_id\tpredicted_ancestry\t' + '\t'.join(labels) + '\n')
    for s_id, prediction, probabilities in zip(sample_ids, predictions, probs):
        out_f.write(f'{s_id}\t{prediction}\t' + '\t'.join([str(x)
            for x in probabilities]) + '\n')

    out_f.close()


def write_principle_components_file(sample_ids, pcs, output_fp):
    f = open(output_fp, 'w')
    for sample_id, vals in zip(sample_ids, pcs):
        f.write(sample_id + '\t' + '\t'.join([str(x) for x in vals]) + '\n')
    f.close()

def run_model(thousand_genomes_df, sample_df, target_df, sample_id_to_ancestry,
        test_size=.2, stats_dict=None, num_components=20, output_dir=os.getcwd()):
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
            thousand_genomes_df, target_df, test_size=test_size)

    if stats_dict is not None:
        stats_dict['model'] = {'test_split': test_size, 'num_pca_components': num_components}

    X_train, X_test = X_train_df.values, X_test_df.values
    y_train, y_test = (np.reshape(y_train_df.values, (y_train_df.shape[0],)),
           np.reshape(y_test_df.values, (y_test_df.shape[0],)))

    # create one hot encoder
    genotype_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    genotype_encoder.fit(X_train)

    # actually transform
    X_train = genotype_encoder.transform(X_train)

    # do pca
    pca = PCA(n_components=num_components)
    pca.fit(X_train)
    
    X_train_pcs = pca.transform(X_train)

    # write your thousand genomes training pcs file
    write_principle_components_file(X_train_df.index, X_train_pcs, os.path.join(output_dir, 'thousand_genomes.training.pcs'))

    #plot_components(X_train_pcs, X_train_df.index, sample_id_to_ancestry, n=5, prefix='pc.thousand_genomes.training',
    #        output_dir=output_dir)

    # train random forest model
    scaler = StandardScaler()
    scaler.fit(X_train_pcs)

    X_train_pcs = scaler.transform(X_train_pcs)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train_pcs, y_train)

    score = clf.score(X_train_pcs, y_train)
    if stats_dict is not None:
        stats_dict['model']['training_score'] = float(score)

    # test our new model
    X_test = genotype_encoder.transform(X_test)
    X_test_pcs = pca.transform(X_test)

    # write your thousand genomes test pcs file
    write_principle_components_file(X_test_df.index, X_test_pcs, os.path.join(output_dir, 'thousand_genomes.test.pcs'))

    #plot_components(X_test_pcs, X_test_df.index, sample_id_to_ancestry, n=5, prefix='pc.thousand_genomes.test',
    #        output_dir=output_dir)

    X_test_pcs = scaler.transform(X_test_pcs)
    score = clf.score(X_test_pcs, y_test)
    if stats_dict is not None:
        stats_dict['model']['test_score'] = float(score)

    # make predictions
    samples_test = sample_df.values
    samples_test = genotype_encoder.transform(samples_test)
    X_test_pcs = pca.transform(samples_test)

    # write your sample pcs file
    write_principle_components_file(sample_df.index, X_test_pcs, os.path.join(output_dir, 'samples.pcs'))

    X_test_pcs = scaler.transform(X_test_pcs)
    predictions = clf.predict(X_test_pcs)
    probs = clf.predict_proba(X_test_pcs)
    classes = clf.classes_

    sample_id_to_predicted_ancestry = {s_id:p for s_id, p in zip(sample_df.index, predictions)}
    #plot_components(X_test_pcs, sample_df.index, sample_id_to_predicted_ancestry, n=5,
    #        prefix='pc.samples', output_dir=output_dir)

    write_predictions_file(os.path.join(output_dir, 'predictions.tsv'),
            sample_df.index, predictions, probs, classes)

    if stats_dict is not None:
        with open(os.path.join(output_dir, 'stats.yaml'), 'w') as outfile:
            yaml.dump(stats_dict, outfile, default_flow_style=False)

def preform_ancestry_analysis(thousand_genomes_vcf_fp, sample_vcf_fp, thousand_genomes_panel_fp,
        output_dir=os.getcwd()):
    stats_dict = {}
    thousand_genomes_df, sample_df = create_dfs(thousand_genomes_vcf_fp, sample_vcf_fp,
            stats_dict=stats_dict)
    target_df, sample_id_to_ancestry = create_target_df(thousand_genomes_df,
            thousand_genomes_panel_fp, super_population=True)

    # make population directory
    super_population_dir = os.path.join(output_dir, 'super_population')
    sub_population_dir = os.path.join(output_dir, 'sub_population')
    if not os.path.isdir(super_population_dir):
        os.mkdir(super_population_dir)
    if not os.path.isdir(sub_population_dir):
        os.mkdir(sub_population_dir)

    run_model(thousand_genomes_df, sample_df, target_df, sample_id_to_ancestry,
            test_size=.2, stats_dict=stats_dict, num_components=20, output_dir=super_population_dir)

    # run for sub populations
    target_df, sample_id_to_ancestry = create_target_df(thousand_genomes_df,
            thousand_genomes_panel_fp, super_population=False)

    run_model(thousand_genomes_df, sample_df, target_df, sample_id_to_ancestry,
            test_size=.2, stats_dict=stats_dict, num_components=20, output_dir=sub_population_dir)
