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
import matplotlib.pyplot as plt
import pandas as pd



##################
MIN_CN_CELLS = 30
CN_MAX_LOW_SCORE = 0.2
CN_MIN_HIGH_SCORE = 1
##################




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
def find_cancellations(adata, gene1, gene2, maxdepth, go_files, 
                    skip_gene_indices={-100}, depth=0, G=nx.Graph(), go_adata_dense=None, go_adata_size=None, 
                    go_adata_names=None, num_search_children=4):

    if go_adata_dense is None:
        go_adata = build_data_structures(adata, go_files)
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
            
            print('Found connection:\t', source_node_name, '~=', dest_node_name)
            G.add_edge(source_node_name, dest_node_name, weight=distance, correlation=correlation, pathway=pathway)

            if depth < maxdepth - 1:
                G = find_cancellations(adata, dest_g1, dest_g2, maxdepth, go_files, 
                        skip_gene_indices=skip_gene_indices, depth=depth + 1, G=G, go_adata_dense=go_adata_dense, 
                            go_adata_size=go_adata_size, go_adata_names=go_adata_names, num_search_children=num_search_children)

    return G


def build_data_structures(adata, go_files):
    go_dict = dict()
    genes_set = set()
    valid_gene_symbols = set(adata.var.index)

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

            ###

            ### Will adapt gene names to either abc, Abc, or ABC depending on what matches!
            g_list = []
            for potential_gene in t_split[1:]:
                if len(potential_gene) == 0 or potential_gene == '\n':
                    continue
                if potential_gene.lower() in valid_gene_symbols:
                    g_list.append(potential_gene.lower())
                elif potential_gene.lower().capitalize() in valid_gene_symbols:
                    g_list.append(potential_gene.lower().capitalize())
                elif potential_gene.upper() in valid_gene_symbols:
                    g_list.append(potential_gene.upper())


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
    assert go_files is not None, 'Must pass in TSV of gene signatures. Example: go_files=\'mygenelist.tsv\' or go_files=[\'gene_list1.tsv\', \'gene_list2.tsv\']. You can find some here: https://github.com/connerlambden/Cube/tree/main/pathways'
    if out_directory[-1] != '/':
        out_directory += '/'
    if not os.path.isdir(out_directory):
        os.mkdir(out_directory)

    if adata.shape[1] > 3000:
        print('WARNING: Found', adata.shape[1], 'genes. So many genes could potentially cause memory errors. We recommend subsetting to Highly Variable Genes first (eg "adata = adata[:,adata.var[\'highly_variable\']]").')

    if search_depth > 2:
        print('WARNING search_depth is greater than 2. This may take a while!!')

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
    
    # print('type(adata.X)', type(adata.X))
    try:
        assert not np.any(np.isnan(adata.X)), 'Found nans in expression matrix!!'
    except TypeError:
        # Ugh data types issues..
        pass
    assert np.all(adata.X < 50), 'Cubé expects log(counts) data. You can use scanpy\'s log1p function'
    assert seed_gene_1 in adata.var.index, 'Seed Gene 1 not found in gene names!'
    if seed_gene_2 is not None:
        assert seed_gene_2 in adata.var.index, 'Seed Gene 2 not found in gene names!'

    adata.var_names_make_unique()
    

    G = find_cancellations(adata, seed_gene_1, seed_gene_2, search_depth, go_files, num_search_children=num_search_children)
    out_name = 'Cubé_' + seed_gene_1
    if seed_gene_2 is not None:
        out_name += '_' + str(seed_gene_2)

    

    nx.write_graphml(G, out_directory + out_name + '.graphml')


    edges = G.edges()
    edge_colors = ['green' if G[u][v]['correlation'] == 'positive' else 'red' for u, v in edges]
    edge_widths = [5 for u, v in edges]
    pos = nx.spring_layout(G, weight='weight', k=10)

    for show_pathways in [True, False]:
        plt.clf()
        plt.cla()
        fig, ax = plt.subplots(figsize=(20, 20))
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', ax=ax, node_size=2000)
        nx.draw_networkx_labels(G, pos, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, ax=ax)
        edge_labels = nx.get_edge_attributes(G, 'pathway')
        if show_pathways:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=6, ax=ax, alpha=0.6)
        plt.title('Cubé Gene Disccovery Network')
        save_n = 'Cubé_network'
        if not show_pathways:
            save_n += '_without_pathway_names'
        plt.tight_layout()
        plt.savefig(out_directory + save_n + '.png', dpi=600)

    edges_list = list(edges)
    edges_list_names = [e[0] + ' ~= ' + e[1] for e in edges_list]

    results_df = pd.DataFrame(index=edges_list_names, columns=['Correlation', 'Edge Weight', 'Pathway'])
    results_df.index.name = 'Gene Interactions'
    c = 0
    for e in edges_list:
        results_df['Correlation'].loc[edges_list_names[c]] = G.edges[e]['correlation']
        results_df['Pathway'].loc[edges_list_names[c]] = G.edges[e]['pathway']
        results_df['Edge Weight'].loc[edges_list_names[c]] = G.edges[e]['weight']
        c += 1
    results_df.to_csv(out_directory + 'Cubé_data_table.csv')

    
    print('DONE!!')
    print('🧊🧊')
    ##############


if __name__ == '__main__':
    adata = sc.read_h5ad('/Users/cl144/Downloads/tcell_differentiation_with_louvain_umap_log_semidefinite_hvgs.h5ad')
    go_files = ['/Users/cl144/Documents/Work/Cubé pypi/Signatures/BioPlanet_2019.tsv', '/Users/cl144/Documents/Work/Cubé pypi/Signatures/GeneSigDB.tsv', '/Users/cl144/Documents/Work/Cubé pypi/Signatures/KEGG_2019_Mouse.tsv']
    # Visualizing Product of 2 Genes using Scanpy (assuming adata.X is logged)
    gene_1 = 'ifng'
    gene_2 = 'tbx21'
    adata_expressing_both = adata[(adata[:,gene_1].X.toarray().flatten() > 0) & (adata[:,gene_2].X.toarray().flatten() > 0),:]
    adata_expressing_both.obs[gene_1 + ' * ' + gene_2] = np.exp(adata_expressing_both[:,gene_1].X.toarray() + adata_expressing_both[:,gene_2].X.toarray())
    sc.pl.umap(adata_expressing_both, color=[gene_1 + ' * ' + gene_2])
    #run_cube(adata=adata, seed_gene_1='ifng', seed_gene_2='tbx21', go_files=go_files, 
    #        out_directory='/Users/cl144/Downloads/cube_test', num_search_children=4, search_depth=2)