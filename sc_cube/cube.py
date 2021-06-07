import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import numpy as np
from numba import jit
import os
from scipy.sparse import issparse
import networkx as nx
import scanpy as sc
import time
import anndata
from scipy.sparse import dok_matrix




##################
MIN_CN_CELLS = 30
CN_MAX_LOW_SCORE = 0.2
CN_MIN_HIGH_SCORE = 1
##################



PROJECT_NAME = 'tcell_differentiation'
########################################
if PROJECT_NAME == 'lloyd_bko':
    OBS_KEYS = ['sampleID', 'genotype']
    OPTIMIZATION_OBS_KEY = 'sampleID'
    BAD_GENES_SET = {'Junb', 'Jund'}
elif PROJECT_NAME == 'bcell_timecourse':
    OBS_KEYS = ['tissue', 'date', 'leiden']
    OPTIMIZATION_OBS_KEY = 'tissue'
elif PROJECT_NAME == 'homo':
    OBS_KEYS = ['donor_organism.sex', 'louvain', 'celltype']
    OPTIMIZATION_OBS_KEY = 'celltype'
elif PROJECT_NAME == 'thelper':
    OBS_KEYS = ['celltype']
    OPTIMIZATION_OBS_KEY = 'celltype'
    lower_gene_names = True
else:
    OBS_KEYS = ['timepoint']
    OPTIMIZATION_OBS_KEY = 'timepoint'
    lower_gene_names = True



@jit(nopython=True, parallel=False)
def cancel_distance_function(a, b):

    a = (a - np.mean(a)) / np.std(a)
    b = (b - np.mean(b)) / np.std(b)
    diff = np.abs(a - b)

    return np.median(diff)


@jit(nopython=True, parallel=False)
def run_summation_fast(dense_expr, input_g1_idx, input_g2_idx, skip_gene_indices, num_search_children, gene2, go_adata_dense, go_adata_size):
    num_genes = dense_expr.shape[1]
    gene_nonzero = dense_expr > 0
    best_result_list_high = np.full(num_search_children, np.inf)
    best_pathways_high = np.full(num_search_children, 0)
    best_result_index_1_high = np.full(num_search_children, 0)
    best_result_index_2_high = np.full(num_search_children, 0)
    addition_thresh_high = np.inf
    ###
    best_result_list_low = np.full(num_search_children, -np.inf)
    best_pathways_low = np.full(num_search_children, 0)
    best_result_index_1_low = np.full(num_search_children, 0)
    best_result_index_2_low = np.full(num_search_children, 0)
    addition_thresh_low = -np.inf

    go_adata_names = np.arange(go_adata_dense.shape[0])
    #
    num_input_genes = 2 if input_g2_idx is not None else 1
    for g1_idx in range(1, num_genes):
        # if g1_idx == input_g1_idx or g1_idx == input_g2_idx:
        #     continue
        if g1_idx in skip_gene_indices:
            continue
        g1_nonzero = gene_nonzero[:,g1_idx]
        for g2_idx in range(-1, g1_idx - 1):
            if g1_idx == g2_idx:
                continue
            if g2_idx in skip_gene_indices:
                continue
            if g1_idx == input_g1_idx or g2_idx == input_g2_idx:
                continue
            if g1_idx == input_g2_idx or g2_idx == input_g1_idx:
                continue
            num_output_genes = 1
            g2_nonzero = None
            if g2_idx != -1:
                g2_nonzero = gene_nonzero[:,g2_idx]
                num_output_genes = 2
            #
            ### GO filter ##
            num_matches_required = num_input_genes + num_output_genes
            search_genes = np.array([g1_idx, input_g1_idx])
            if gene2 is not None:
                search_genes = np.append(search_genes, input_g2_idx)
            if g2_idx != -1:
                search_genes = np.append(search_genes, g2_idx)


            go_adata_dense_ss = go_adata_dense[:,search_genes]
            valid_pathways = np.sum(go_adata_dense_ss, axis=1) >= num_matches_required

            go_adata_dense_ss = go_adata_dense_ss[valid_pathways,:]
            if go_adata_dense_ss.shape[0] < 3:
                continue
            go_adata_size_ss = go_adata_size[valid_pathways]
            go_adata_names_ss = go_adata_names[valid_pathways]
            

            go_sum = np.ravel(np.sum(go_adata_dense_ss, axis=1))
            #print('go_sum', go_sum)
            best_pathway_idx = np.argmax(go_sum / go_adata_size_ss)
            pathway_name = go_adata_names_ss[best_pathway_idx]

            intersecting_cells = None
            if g2_nonzero is not None:
                intersecting_cells = g1_nonzero & g2_nonzero
            else:
                intersecting_cells = g1_nonzero

            if np.sum(intersecting_cells) < MIN_CN_CELLS:
                continue
            g1_expression_vec = dense_expr[:,g1_idx][intersecting_cells]
            
            g2_expression_vec = None
            if g2_idx != -1:
                g2_expression_vec = dense_expr[:,g2_idx][intersecting_cells]

            ##################
            
            desired_g1_vec = dense_expr[:,input_g1_idx][intersecting_cells]
            
            desired_g2_full_vec = None
            if input_g2_idx is None:
                desired_g2_full_vec = np.exp(desired_g1_vec)
            else:
                desired_g2_vec = dense_expr[:,input_g2_idx][intersecting_cells]
                desired_g2_full_vec = np.exp(desired_g1_vec + desired_g2_vec)

            score_input_vec = None
            if g2_idx != -1:
                score_input_vec = np.exp(g1_expression_vec + g2_expression_vec)
            else:
                score_input_vec = np.exp(g1_expression_vec)

            score = cancel_distance_function(score_input_vec, desired_g2_full_vec)
            if score < addition_thresh_high and score < CN_MAX_LOW_SCORE:
                addition_index = np.argmax(best_result_list_high)
                best_result_list_high[addition_index] = score
                best_pathways_high[addition_index] = pathway_name
                best_result_index_1_high[addition_index] = g1_idx
                best_result_index_2_high[addition_index] = g2_idx
                addition_thresh_high = np.max(best_result_list_high)
            elif score > addition_thresh_low and score > CN_MIN_HIGH_SCORE:
                addition_index = np.argmin(best_result_list_low)
                best_pathways_low[addition_index] = pathway_name
                best_result_list_low[addition_index] = score
                best_result_index_1_low[addition_index] = g1_idx
                best_result_index_2_low[addition_index] = g2_idx
                addition_thresh_low = np.max(best_result_list_low)


    return best_result_list_high, best_result_index_1_high, best_result_index_2_high, best_pathways_high, \
            best_result_list_low, best_result_index_1_low, best_result_index_2_low, best_pathways_low
     


@jit
def find_cancellations(adata, gene1, gene2, maxdepth, lower_gene_names, go_files, 
                    skip_gene_indices={-100}, depth=0, G=nx.Graph(), go_adata_dense=None, go_adata_size=None, 
                    go_adata_names=None, num_search_children=4):

    if go_adata_dense is None:
        go_adata = build_data_structures(go_files, lower_gene_names)
        go_adata_names = list(go_adata.obs.index)
        go_adata_dense = go_adata.X
        go_adata_size = np.array(go_adata.obs['size'], dtype=np.int)
        assert gene1 in adata.var.index
        assert gene1 in go_adata.var.index, 'Seed Gene 1 Not Found in Any Pathways. Consider adding more pathways or changing input seed genes.'
        if gene2 is not None:
            assert gene2 in go_adata.var.index, 'Seed Gene 2 Not Found in Any Pathways. Consider adding more pathways or changing input seed genes.'
        if gene2 is not None:
            assert gene2 in adata.var.index
        adata = adata[:,list(go_adata.var.index)]
        print('Found', len(set(go_adata.var.index) & set(adata.var.index)), 'genes in GO lists & single-cell data to search in.')
        print('🧊 Cubé is Running! 🧊')

    source_node_name = gene1
    if gene2 is not None:
        source_node_name += ' * ' + gene2
        if gene2 + ' * ' + gene1 in G.nodes:
            source_node_name = gene2 + ' * ' + gene1
    adata_ss = adata[np.array(adata[:,gene1].X.toarray()).flatten() > 0]
    if gene2 is not None:
        adata_ss = adata_ss[np.array(adata_ss[:,gene2].X.toarray()).flatten() > 0]
    
    
    input_g1_idx = list(adata_ss.var.index).index(gene1)
    input_g2_idx = list(adata_ss.var.index).index(gene2) if gene2 is not None else None
    var_names = list(adata_ss.var.index)
    
    num_cells = adata_ss.shape[0]
    if not num_cells > MIN_CN_CELLS:
        if depth == 0:
            p_string = 'Only found ' + str(num_cells) + ' cells expressing ' + gene1
            if gene2 is not None:
                p_string += ' and ' + gene2
            print(p_string, 'Need at least ' + str(MIN_CN_CELLS) + '. Terminating.')
        return G

    dense_expr = adata_ss.X
    if issparse(dense_expr):
        dense_expr = adata_ss.X.toarray() ##assumption!!!

    if issparse(go_adata_dense):
        go_adata_dense = np.array(go_adata_dense.toarray(), dtype=np.int)

    start = time.time()

    res = run_summation_fast(dense_expr, input_g1_idx, input_g2_idx, skip_gene_indices, num_search_children, gene2, go_adata_dense, go_adata_size)

    print('Iteration took', round(time.time() - start, 2), 'seconds!!')

    skip_gene_indices.add(input_g1_idx)
    if gene2 is not None:
        skip_gene_indices.add(input_g2_idx)

    for res_idx in range(num_search_children):  
        for correlation in ['positive', 'negative']:
            if correlation == 'positive':
                score = res[0][res_idx]
                index_1 = res[1][res_idx]
                index_2 = res[2][res_idx]
                pathway = go_adata_names[res[3][res_idx]]
            else:
                score = res[4][res_idx]
                index_1 = res[5][res_idx]
                index_2 = res[6][res_idx]
                pathway = go_adata_names[res[7][res_idx]]

            dest_g2 = None
            dest_g1 = var_names[index_1]
            if index_2 != -1:
                dest_g2 = var_names[index_2]
            dest_node_name = None
            if dest_g2 is not None and dest_g2 + ' * ' + dest_g1 in G.nodes:
                dest_node_name = dest_g2 + ' * ' + dest_g1
            elif dest_g2 is not None:
                dest_node_name = dest_g1 + ' * ' + dest_g2
            else:
                dest_node_name = dest_g1

            distance = score
            if np.isnan(distance) or np.isinf(distance):
                continue
            
            print('Found connection:\t', source_node_name, '~', dest_node_name)
            G.add_edge(source_node_name, dest_node_name, weight=distance, correlation=correlation, pathway=pathway)

            if depth < maxdepth - 1:
                G = find_cancellations(adata, dest_g1, dest_g2, maxdepth, lower_gene_names, go_files, 
                        skip_gene_indices=skip_gene_indices, depth=depth + 1, G=G, go_adata_dense=go_adata_dense, 
                            go_adata_size=go_adata_size, go_adata_names=go_adata_names, num_search_children=num_search_children)

    return G


if PROJECT_NAME == 'bcell_timecourse':
    
    INPUT_ADATA = '/home/ec2-user/data/Time_Course_Cellbender_corrected_batch_correct_harmony_new_annotForCellphoneDB_scirpy_hvgs_only_bcells_only.h5ad'
    gois = set([g.lower().capitalize() for g in {'tigit', 'havcr1','havcr2','icosl','pdcd1','nt5e','entpd1','mki67','il10','ifng','tnf'}])
    EXTRA_COLS = ['tissue', 'date'] + list(gois)

elif PROJECT_NAME == 'tcell_differentiation':
    # INPUT_ADATA = '/home/ec2-user/data/tcell_differentiation_with_louvain_umap.h5ad'
    INPUT_ADATA = '/home/ec2-user/data/tcell_differentiation_with_louvain_umap_log_semidefinite_hvgs.h5ad'
    
    gois = {'twist1', 'tbx21', 'stat4', 'runx3', 'ifng', 'il12rb2', 'p2rx7', 'dusp5', 'cd4', 'gata3', 'il4',
                'stat1', 'stat4', 'stat6', 'il12', 'il18', 'nfat', 'ap1', 'nfkb1', 'il12a', 'il12rb2'}
    EXTRA_COLS = ['n_genes', 'timepoint'] + list(gois)
elif PROJECT_NAME == 'thelper':
    INPUT_ADATA = '/home/ec2-user/data/thelper_hvgs.h5ad'
    gois = {'twist1', 'tbx21', 'stat4', 'runx3', 'ifng', 'il12rb2', 'p2rx7', 'dusp5', 'cd4', 'gata3', 'il4',
                'stat1', 'stat4', 'stat6', 'il12', 'il18', 'nfat', 'ap1', 'nfkb1', 'ccr5', 'il12a', 'il12rb2'}
    EXTRA_COLS = ['celltype'] + list(gois)

elif PROJECT_NAME == 'tcell_tfs':
    EXTRA_COLS = ['n_genes', 'timepoint']
    INPUT_ADATA = '/home/ec2-user/data/tcell_differentiation_with_louvain_umap_tfs_only.h5ad'
    gois = {'twist1', 'tbx21', 'stat4', 'runx3', 'ifng', 'il12rb2', 'p2rx7', 'dusp5', 'ms4a4b', 'pla2g12a', 'crabp2', 'cd4', 'gata3', 'il4',
                'stat1', 'stat4', 'stat6', 'il12', 'il18', 'nfat', 'ap1', 'nfkb'}

elif PROJECT_NAME == 'ilc2':
    EXTRA_COLS = ['n_counts', 'tissue', 'treatment']
    INPUT_ADATA = '/home/ec2-user/lamp/ilc2_with_louvain_umap.h5ad'
    gois = set([i.lower() for i in ['Thy1', 'Tbx21', 'Gata3', 'Rorc', 'Il1rl1', 
                        'Il17rb', 'Klrg1', 'Cd3', 'Cd4', 'Cd8', 'Il7r', 'Foxp3', 'Il10', 'Il4', 'Il5', 'Il13', 
                        'Ifng', 'Il17a', 'Il22', 'Nmur1', 'Ramp1', 'Ramp2', 'Ramp3', 'Calcrl', 'Vipr1', 'Vipr2', 
                        'Calca', 'Calcb', 'Nmu', 'Vip', 'crem', 'ramp3', 'pdcd1', 'fam46a', 'Il5', 'Il13', 'Il1rl1']])
elif PROJECT_NAME == 'lloyd_bko':
    
    INPUT_ADATA = '/home/ec2-user/data/WT_BKO_5hashed_B_cells_harmony_v2_HVGs.h5ad'
    #INPUT_ADATA = '/home/ec2-user/data/WT_BKO_5hashed_B_cells_harmony_v2.h5ad'
    
    gois = set([g.lower().capitalize() for g in {'tigit', 'havcr1','havcr2','icosl','pdcd1','nt5e','entpd1','mki67','il10','ifng','tnf', 'cd74','ciita', 'H2-Ab1', 'H2-Aa', 'H2-Eb1', 'H2-DMb2', 'H2-Oa', 'H2-Ob', 'H2-Dma', 'H2-DMb1',
            'H2-Ke6', 'H2-Ke2', 'H2-Eb2'}])

    EXTRA_COLS = ['n_counts', 'leiden', 'tissue', 'genotype'] + list(gois)
    
elif PROJECT_NAME == 'lloyd_bko_Tum':
    EXTRA_COLS = ['n_counts', 'sampleID', 'leiden', 'tissue', 'genotype']
    INPUT_ADATA = '/home/ec2-user/data/Explore_WT_BKO_5prime_Hashed_bcells_only_logged_top_3k_HVGs_nonsparse_Tum_only.h5ad'
    gois = [i.lower() for i in {'tigit','ctla4','areg','mki67','il10','ifng','tnf', 'ly6g', 'ly6c1', 'itgax', 'enpp1', 'mki67',
            'ly6a', 'alox5ap', 'cd68', 'flt3', 'H2-Ab1', 'H2-Aa', 'H2-Eb1', 'H2-DMb2', 'H2-Oa', 'H2-Ob', 'H2-Dma', 'H2-DMb1', 
            'H2-Ke6', 'H2-Ke2', 'H2-Eb2'}]

elif PROJECT_NAME == 'lloyd_bko_dLN':
    EXTRA_COLS = ['n_counts', 'sampleID', 'leiden', 'tissue', 'genotype']
    INPUT_ADATA = '/home/ec2-user/data/Explore_WT_BKO_5prime_Hashed_bcells_only_logged_top_3k_HVGs_nonsparse_dLN_only.h5ad'
    gois = [i.lower() for i in {'tigit','ctla4','areg','mki67','il10','ifng','tnf', 'ly6g', 'ly6c1', 'itgax', 'enpp1', 'mki67',
            'ly6a', 'alox5ap', 'cd68', 'flt3', 'H2-Ab1', 'H2-Aa', 'H2-Eb1', 'H2-DMb2', 'H2-Oa', 'H2-Ob', 'H2-Dma', 'H2-DMb1', 
            'H2-Ke6', 'H2-Ke2', 'H2-Eb2'}]

elif PROJECT_NAME == 'lloyd_bko_nLN':
    EXTRA_COLS = ['n_counts', 'sampleID', 'leiden', 'tissue', 'genotype']
    INPUT_ADATA = '/home/ec2-user/data/Explore_WT_BKO_5prime_Hashed_bcells_only_logged_top_3k_HVGs_nonsparse_nLN_only.h5ad'
    gois = [i.lower() for i in {'tigit','ctla4','areg','mki67','il10','ifng','tnf', 'ly6g', 'ly6c1', 'itgax', 'enpp1', 'mki67',
            'ly6a', 'alox5ap', 'cd68', 'flt3', 'H2-Ab1', 'H2-Aa', 'H2-Eb1', 'H2-DMb2', 'H2-Oa', 'H2-Ob', 'H2-Dma', 'H2-DMb1', 
            'H2-Ke6', 'H2-Ke2', 'H2-Eb2'}]

elif PROJECT_NAME == 'homo':
    # https://data.humancellatlas.org/explore/projects/cc95ff89-2e68-4a08-a234-480eca21ce79
    gois = {'TBX21', 'STAT6', 'STAT4', 'IL4', 'RUNX3', 'TWIST1', 'GATA3', 'STAT1', 'NFKB', 'IL18', 'AP1', 
            'CD4', 'CRABP2', 'IL12', 'DUSP5', 'NFAT', 'PLA2G12A', 'IL12RB2', 'IFNG', 'P2RX7', 'MS4A4B', 'IL3'}
    EXTRA_COLS = OBS_KEYS
    INPUT_ADATA = '/home/ec2-user/data/homo_sapiens_louvain_immune_hvgs.h5ad'



def build_data_structures(go_files, lower_gene_names):
    go_dict = dict()
    genes_set = set()
    valid_gene_symbols = set(adata.var.index)
    go_adata_dict = dict()
    pathway_list = []
    for go_file in go_files:
        go_data = open(go_file, 'r').readlines()
        counter = 0
        for l in go_data:
            t_split = l.split('\t')
            pathway_name = None
            if 'GeneSigDB' in go_file:
                pathway_name = 'Pubmed: ' + t_split[0]
            else:
                pathway_name = t_split[0]
            g_list = None
            if lower_gene_names:
                g_list_total = [g.lower() for g in t_split[1:] if len(g) > 0 and g != '\n']
                g_list = [g for g in g_list_total if g in valid_gene_symbols]
                go_adata_dict[pathway_name] = len(g_list_total)
            else:
                g_list_total = [g.lower().capitalize() for g in t_split[1:] if len(g) > 0 and g != '\n']
                g_list = [g for g in g_list_total if g in valid_gene_symbols]
                go_adata_dict[pathway_name] = len(g_list_total)
            if len(g_list) < 4:
                counter += 1
                continue
            pathway_list.append(pathway_name)
            go_dict[pathway_name] = g_list
            genes_set.update(g_list)
            counter += 1

    gene_list = list(genes_set)

    mat = dok_matrix((len(pathway_list), len(gene_list)), dtype=np.int8)

    pathway_counter = 0
    for pathway_name in pathway_list:
        for g in go_dict[pathway_name]:
            g_index = gene_list.index(g)
            mat[pathway_counter, g_index] = 1
        pathway_counter += 1

    mat = mat.tocsr()
    go_adata = anndata.AnnData(X=mat, obs=pathway_list, var=gene_list)
    go_adata.obs.index = pathway_list
    go_adata.var.index = gene_list
    go_adata.obs['size'] = np.sum(go_adata.X, axis=1)
    return go_adata



def run_cube(adata=None, seed_gene_1=None, seed_gene_2=None, go_files=None, out_directory=None, num_search_children=4, search_depth=1):
    assert len(out_directory) > 0, 'Bad output directory.'
    assert adata is not None, 'Must pass in valid AnnData object for adata. Example: adata=adata'
    assert seed_gene_1 is not None, 'Must enter at least one search gene! Examples: seed_gene_1=\'Ifng\''
    assert go_files is not None, 'Must pass in TSV of gene signatures. Example: go_files=\'mygenelist.tsv\' or go_files=[\'gene_list1.tsv\', \'gene_list2.tsv\']. You can find some here: https://maayanlab.cloud/Enrichr/#stats'
    if out_directory[-1] != '/':
        out_directory += '/'
    if not os.path.isdir(out_directory):
        os.mkdir(out_directory)

    if type(go_files) == list:
        for gf in go_files:
            assert os.path.isfile(gf), 'Cannot find gene signature file: ' + str(gf)
    else:
        assert os.path.isfile(go_files), 'Cannot find gene signature file: ' + str(go_files)
        go_files = [go_files]

    assert os.path.isdir(out_directory), 'Bad output directory!!'
    
    assert adata.shape[0] >= 200, 'Need at least 200 cells to run!'
    first_cell = adata.X[0].copy()
    if issparse(first_cell):
        first_cell = first_cell.toarray()
    assert np.all(first_cell >= 0), 'Cubé expects all positive logged counts'

    if issparse(adata.X) or hasattr(adata.X, 'toarray'):
        adata.X = adata.X.toarray()
    assert not np.any(np.isnan(adata.X)), 'Found nans in expression matrix!!'
    assert seed_gene_1 in adata.var.index, 'Seed Gene 1 not found in gene names!'
    if seed_gene_2 is not None:
        assert seed_gene_2 in adata.var.index, 'Seed Gene 2 not found in gene names!'

    adata.var_names_make_unique()
    
    if seed_gene_1 == seed_gene_1.lower():
        lower_gene_names = True
    G = find_cancellations(adata, seed_gene_1, seed_gene_2, search_depth, lower_gene_names, go_files, num_search_children=num_search_children)
    out_name = 'Cubé_' + CN_GENE_1
    if seed_gene_2 is not None:
        out_name += '_' + str(CN_GENE_2)

    

    nx.write_graphml(G, out_directory + out_name + '.graphml')
    
    print('DONE!!')
    print('🧊🧊')
    ##############


if __name__ == '__main__':
    adata = sc.read_h5ad(INPUT_ADATA)
    go_files = ['/home/ec2-user/data/BioPlanet_2019.tsv', '/home/ec2-user/data/GeneSigDB.tsv', '/home/ec2-user/data/KEGG_2019_Mouse.tsv']
    run_cube(adata, CN_GENE_1, CN_GENE_2, go_files, '/home/ec2-user/lamp', search_depth=CN_MAX_DEPTH)
