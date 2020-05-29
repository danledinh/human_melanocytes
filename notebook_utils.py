# libraries
import pandas as pd
from pandas.api.types import CategoricalDtype, is_categorical_dtype
import numpy as np
import string
import types
import scanpy.api as sc
import anndata as ad
from plotnine import *
import plotnine
import scipy
from scipy import sparse, stats
from scipy.cluster import hierarchy
import glob
import more_itertools as mit
import tqdm
import pickle
import multiprocessing
import itertools
import sklearn
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, classification_report, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import typing
import random
from adjustText import adjust_text
import sys
import lifelines
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib as mp
import matplotlib.pyplot as plt


# classes
class SklearnWrapper:
    """
    class to handle sklearn function piped inline with pandas
    """
    def __init__(self, transform: typing.Callable):
        self.transform = transform

    def __call__(self, df):
        transformed = self.transform.fit_transform(df.values)
        return pd.DataFrame(transformed, columns=df.columns, index=df.index)

# functions
def imports():
    """
    print module names and versions 
    ref: https://stackoverflow.com/questions/20180543/how-to-check-version-of-python-modules
    
    input: none
    output: print to std out
    """
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            if val.__name__ not in ['builtins']:
                try:
                    print (f'{val.__name__}:', val.__version__)
                except:
                    pass
                    
def create_adata (pre_adata):
    """
    Creates adata obj from raw data (rows=gene_names, col=cell_id)
    
    Input: raw expression data in pd df
    Output: adata obj
    """

    
    print('Ingest raw data...')

    # pd df to np array
    array_adata = pre_adata.values

    # extract obs and var
    obs = pre_adata.columns.tolist()
    gene_names = pre_adata.index.tolist()
    var = pd.DataFrame({'gene_symbols':gene_names})
    
    # create ad obj
    adata = ad.AnnData(X=array_adata).T
    adata.X = sparse.csr_matrix(adata.X)
    adata.var_names = gene_names
    adata.obs_names = obs
    
    return adata
    
def append_anno (adata, anno, anno_dict):
    """
    Add annotations of choice from annotation file
    
    input = adata obj + dictionary of label and column name (with respect to annotation df) + anno pd df
    output = updated adata obj
    
    """

    
    print('Append annotations...')
    
    anno = anno
    anno_dict = anno_dict
    
    # append metadata of choice
    for key,value in anno_dict.items():
        adata.obs[value] = anno[key].values

def remove_ercc (adata):
    """
    Remove ercc spike-in
    
    Input: adata obj
    Output: updated adata obj
    """
    
    print('Remove ERCC genes...')
    
    gene_names = adata.var_names.tolist()
    ERCC_hits = list(filter(lambda x: 'ERCC' in x, gene_names))
    adata = adata[:, [x for x in gene_names if not (x in ERCC_hits)]]
    
    return adata

def technical_filters (adata, min_genes=500,min_counts=50000,min_cells=3):
    """
    remove cells/genes based on low quality
    
    input: adata
    output: updated adata obj 
    """

    print('Remove low-quality cells/genes...')

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
def prepare_dataframe(adata, var_names, groupby=None, use_raw=None, log=False, num_categories=7):
    """
    ### FROM scanpy ###
    
    Given the anndata object, prepares a data frame in which the row index are the categories
    defined by group by and the columns correspond to var_names.
    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix.
    var_names : `str` or list of `str`
        `var_names` should be a valid subset of  `adata.var_names`.
    groupby : `str` or `None`, optional (default: `None`)
        The key of the observation grouping to consider. It is expected that
        groupby is a categorical. If groupby is not a categorical observation,
        it would be subdivided into `num_categories`.
    log : `bool`, optional (default: `False`)
        Use the log of the values
    use_raw : `bool`, optional (default: `None`)
        Use `raw` attribute of `adata` if present.
    num_categories : `int`, optional (default: `7`)
        Only used if groupby observation is not categorical. This value
        determines the number of groups into which the groupby observation
        should be subdivided.
    Returns
    -------
    Tuple of `pandas.DataFrame` and list of categories.
    """
    from scipy.sparse import issparse
#     sanitize_anndata(adata)
    if use_raw is None and adata.raw is not None: use_raw = True
    if isinstance(var_names, str):
        var_names = [var_names]

    if groupby is not None:
        if groupby not in adata.obs_keys():
            raise ValueError('groupby has to be a valid observation. Given value: {}, '
                             'valid observations: {}'.format(groupby, adata.obs_keys()))

    if use_raw:
        matrix = adata.raw[:, var_names].X
    else:
        matrix = adata[:, var_names].X

    if issparse(matrix):
        matrix = matrix.toarray()
    if log:
        matrix = np.log1p(matrix)

    obs_tidy = pd.DataFrame(matrix, columns=var_names)
    if groupby is None:
        groupby = ''
        categorical = pd.Series(np.repeat('', len(obs_tidy))).astype('category')
    else:
        if not is_categorical_dtype(adata.obs[groupby]):
            # if the groupby column is not categorical, turn it into one
            # by subdividing into  `num_categories` categories
            categorical = pd.cut(adata.obs[groupby], num_categories)
        else:
            categorical = adata.obs[groupby]

    obs_tidy.set_index(categorical, groupby, inplace=True)
    categories = obs_tidy.index.categories

    return categories, obs_tidy

def regress(x, y, predictor, response, fit_intercept = False):
    """
    wrapper for linear regression
    
    input: 1D array of shape (-1, 1)
    output: fit and residuals
    """
    
    # Model initialization
    regression_model =  LinearRegression(fit_intercept=fit_intercept)
    # Fit the data(train the model)
    regression_model.fit(x, y)
    # Predict
    y_predicted = regression_model.predict(predictor).reshape((-1,1))
    y_residuals = response - y_predicted
    # coef
    m = regression_model.coef_[0][0]
    
    return y_predicted.flatten(), y_residuals.flatten(), m

def adata_DE_pairwise(input_adata, 
                      groupby, 
                      target_1, 
                      target_2, 
                      method = 'wilcoxon',
                      corr_method = 'benjamini-hochberg'
                     ):
    """
    Wrapper for scanpy DE tests. Two-sided.
    
    Input: adata, groupby variable, comparison labels, test, mutliple hypothesis procedure
    Output: dataframe of gene, log2fc, pval, adj_pval
    """
    
    n_genes=len(input_adata.var_names)
    sc.tl.rank_genes_groups(input_adata, 
                            groupby=groupby, 
                            groups=[target_1],
                            reference=target_2,
                            method=method,
                            n_genes = n_genes,
                            corr_method = corr_method
                           )
    genes = [x[0] for x in input_adata.uns['rank_genes_groups']['names']]
    log2change = [x[0] for x in input_adata.uns['rank_genes_groups']['logfoldchanges']]
    pvals = [x[0] for x in input_adata.uns['rank_genes_groups']['pvals']]
    pvals_adj = [x[0] for x in input_adata.uns['rank_genes_groups']['pvals_adj']]
    
    results_df = pd.DataFrame({
        'gene':genes,
        'log2change':log2change,
        'pvals':pvals,
        'pvals_adj':pvals_adj
    })
    
    return results_df

def fast_DE(input_adata, clusterOI, groupby, reference='rest', n_genes=10):
    """
    Wrapper for scanpy DE function
    
    Input: groupby label, cluster of interest label, referennce label, method name, number of genes
    outputed
    Output: gene list
    """
    sc.tl.rank_genes_groups(input_adata, 
                            groupby=groupby, 
                            groups=[clusterOI], 
                            method='wilcoxon',
                            reference=reference,
                            n_genes = n_genes)
    gene = [x[0] for x in input_adata.uns['rank_genes_groups']['names']]
    return gene

def index_max(x):
    """
    Returns index of max value in list.
    
    Input: List of values
    Output: Index of max value from input list
    """
    return x.index(np.max(x))

def heatmap_sc_plot(merged_adata, gene_order, xplot=8, yplot=6, font=7):
    """
    Heatmap plot function for dedifferentiation modules
    
    Input: adata, gene list
    Output: plot
    """
    groupby = 'source_label'
    type_order = ['normal_adult_interfollicular',
                    'normal_neonatal_interfollicular',
                    'normal_fetal_interfollicular',
                    'normal_fetal_follicular',
                    'cancer_adult_interfollicular',
                    'cancer_neonatal_interfollicular',
                    'cancer_fetal_interfollicular',
                    'cancer_fetal_follicular',     
                    ]
    n_cells = 100
    
    cat, exp_df = prepare_dataframe(merged_adata,
                         var_names = merged_adata.var_names.tolist(),
                         groupby = 'cell')
    exp_df = exp_df.rank(pct=True, method='dense', axis=1)
    exp_df = exp_df.loc[:,gene_order]
    exp_df[groupby] = merged_adata.obs[groupby]

    compiled_rows = pd.DataFrame()
    type_order_revised = []
    for x in type_order:
        df_slice = exp_df[exp_df[groupby] == x]
        df_nrow = len(df_slice)
        if df_nrow >= n_cells:
            df_sample = df_slice.sample(n_cells)
            num_cell = n_cells
        else:
            df_sample = df_slice
            num_cell = df_nrow
        idx_list = [x for x in range(len(df_sample))]
        random.shuffle(idx_list)
        df_sample['idx'] = idx_list
        df_sample[groupby] = f'{x} ({num_cell}/{df_nrow})'
        type_order_revised = type_order_revised + [f'{x} ({num_cell}/{df_nrow})']
        compiled_rows = compiled_rows.append(df_sample)

    compiled_rows_melt = pd.melt(compiled_rows, id_vars=[groupby,'idx'])
    compiled_rows_melt[groupby] = (compiled_rows_melt[groupby]
                                     .astype(str)
                                     .astype(CategoricalDtype(type_order_revised,
                                                              ordered=True
                                                             )
                                            )
                                    )
    compiled_rows_melt['variable'] = (compiled_rows_melt['variable']
                                     .astype(str)
                                     .astype(CategoricalDtype(gene_order,
                                                              ordered=True
                                                             )
                                            )
                                    )

    plotnine.options.figure_size = (xplot,yplot)
    plot = (ggplot(compiled_rows_melt)
          + theme_bw()
          + theme(axis_text_x = element_blank(),
                  axis_text_y = element_text(size = font),
                  strip_text_x = element_text(angle = 90, vjust = 0),
                  strip_background_x = element_rect(fill = 'white', color = 'white') )
          + geom_tile(aes('idx','variable',fill='value'))
          + facet_grid(f'~{groupby}', scales='free')
          + scale_fill_cmap(heatmap_cmap)
          + labs(x = '', y = ''))

    print(plot)
    
def heatmap_wilcoxon(merged_adata, gl, target_cancer=True, xplot=8, yplot=6, font=7, alpha = 0.05):
    """
    Bar plot function for dedifferentiation module wilcoxon test p-values
    
    Input: adata, gene list
    Output: plot and results dataframe
    """
    results_df = pd.DataFrame()
    for varval in varvals:
        if target_cancer == True:
            group1 = f'cancer_{varval}'
            group2 = f'normal_{varval}'
        else:
            group2 = f'cancer_{varval}'
            group1 = f'normal_{varval}'
        results_slice = adata_DE_pairwise(merged_adata[:, gl], 
                                      'source_label', 
                                      group1, # right group
                                      group2, # left group
                                     )
        results_slice['varval'] = varval
        results_slice['neglog10_pvals_adj'] = -np.log10(results_slice['pvals_adj'])
        results_df = results_df.append(results_slice)
        
    if target_cancer == False:
        results_df['log2change'] = -results_df['log2change']

    results_df['varval'] = (results_df['varval']
                            .astype(str)
                            .astype(CategoricalDtype(varvals, ordered = True))
                           )
    results_df['gene'] = (results_df['gene']
                        .astype(str)
                        .astype(CategoricalDtype(gl, ordered = True))
                       )    

    ylimval = np.max(np.abs(results_df['log2change']))
    label_df = results_df[results_df['neglog10_pvals_adj'] <= -np.log10(alpha)]

    plotnine.options.figure_size = (xplot,yplot)
    plot = (ggplot(results_df)
                + theme_bw()
                + theme(axis_text_x = element_text(size=font),
                        strip_text_x = element_text(angle = 90, vjust = 0),
                        strip_background_x = element_rect(fill = 'white', color = 'white')
                       )
                + geom_bar(aes('gene','log2change',fill='neglog10_pvals_adj'), stat='identity')
                + geom_hline(aes(yintercept = 0))
                + coord_flip()
                + facet_grid('.~varval')
                + ylim([-ylimval,ylimval])
                + scale_fill_cmap('cool')
                + labs(x='')
               )
    if len(label_df) > 0:
        plot = plot + geom_label(label_df,
                             aes('gene', 0), label='n.s.', size=font)
    print(plot)
    
    return results_df

def rect_converter(df, xval, yval, y_upper, y_lower, grouping):
    """
    Covert survival dataframe to ammenable to step plot
    
    Input: survival dataframe
    Output: updated dataframe
    """
    master = pd.DataFrame()
    for label in set(df[grouping]):
        reassign = pd.DataFrame()
        df_slice = df[df[grouping] == label]
        reassign['xmin'] = df_slice[xval].shift(-1).tolist()
        reassign['xmax'] = df_slice[xval].tolist()
        reassign['ymin'] = df_slice[y_lower].tolist()
        reassign['ymax'] = df_slice[y_upper].tolist()
        reassign['label'] = label
        master = master.append(reassign[:-1])
    return master.dropna()


# Global variables

color_code_dict = {'dendritic':'#b99abf',
                   'cyc_dendritic':'#9abfb9',
                   'eccrine':'#CBB7E3',
                   'cyc_eccrine':'#b7e3cb',
                   'krt':'#E9E1F2',
                   'cyc_krt':'#e1f2e9',
                   'mast':'#AE90C2',
                   'T_cell':'#5F3D68',
                   'mel':'#000000',
                   'cyc_mel':'#999999',
                   'cutaneous_mel':'#FF0000',
                   'cutaneous':'#FF0000',
                   'leg':'#FF0000',
                   'arm':'#FF0000',
                   'acral_mel':'#0000FF',
                   'acral':'#0000FF',
                   'palm':'#0000FF',
                   'sole':'#0000FF',
                   'foreskin_mel':'#FFA500',
                   'foreskin':'#FFA500',
                   'dark_foll_mel':'#003300',
                   'light_foll_mel':'#99cc99',
                   'follicular':'#008000',
                   'hair_follicle':'#008000',
                   'fet_cutaneous_mel':'#ff4c4c',
                   'adt_cutaneous_mel':'#b20000',
                   'shallow_regime':'#b20000',
                   'steep_regime':'#00b2b2',
                   'fet_acral_mel':'#4c4cff',
                   'adt_acral_mel':'#0000b2',
                   'neo_foreskin_mel':'#FFA500',
                   'fet_dark_foll_mel':'#003300',
                   'fet_light_foll_mel':'#99cc99',
                   'fet':'#dbc2a9',
                   'neo':'#c09569',
                   'adt':'#a5682a',
                   'NTRK2+/HPGD+':'#474747',
                   'NTRK2-/HPGD-':'#DDDDDD',
                   'NTRK2+/HPGD-':'#0000FF',
                   'NTRK2-/HPGD+':'#FF0000',
                   'black':'#000000',
                   'grey':'#D3D3D3',
                   'melanoma':'#935aff',
                   'mel':'#935aff',
                   'follicular_like':'#6514ff',
                   'adult_interfollicular':'#ff1439',
                   'follicular_low':'#ff1439',
                   'interfoll_mel':'#ff1439',
                   'neonatal_interfollicular':'#ffda14',
                   'fetal_interfollicular':'#1439ff',
                   'fetal_follicular':'#39ff14',
                   'follicular_high':'#39ff14',
                   'light_foll_mel':'#39ff14',
                   'dark_foll_mel':'#93ba8b',
                   'norm':'#000000',
                   'cluster_1':'#ff1439',
                   'cluster_0':'#ffda14',
                   'cluster_2':'#39ff14',
                  }

heatmap_cmap = 'jet'

type_order_1 = ['mel',
                'cyc_mel',
                'krt',
                'cyc_krt',
                'eccrine',
                'cyc_eccrine',
                'dendritic',
                'cyc_dendritic',
                'mast',
                'T-cell',]

type_order_2 = ['light_foll_mel',
                'dark_foll_mel',
                'foreskin_mel',
                'acral_mel',
                'cutaneous_mel',
                'cyc_foll_mel',
                'cyc_mel',
                'krt',
                'cyc_krt',
                'eccrine',
                'cyc_eccrine',
                'dendritic',
                'cyc_dendritic',
                'mast',
                'T-cell',]

type_order_3 = ['adt_cutaneous_mel',
                'adt_acral_mel',
                'neo_foreskin_mel',
                'fet_cutaneous_mel',
                'fet_acral_mel',
                'fet_dark_foll_mel',
                'fet_light_foll_mel',][::-1] + \
                ['cyc_mel',
                'cyc_foll_mel',
                'krt',
                'cyc_krt',
                'eccrine',
                'cyc_eccrine',
                'dendritic',
                'cyc_dendritic',
                'mast',
                'T-cell',]
