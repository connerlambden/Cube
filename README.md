# Cubé
![Cubé](https://github.com/connerlambden/Cube/raw/main/images/cube%CC%81_header.jpg)
## Intuitive Nonparametric Gene Network Search Algorithm

### How It Works
Given a single-cell dataset and an input gene(s), Cubé looks for simple & nonlinear gene-gene relationships to construct a regulation network informed by prior gene signatures. For example, Cubé might give you the result that GeneA * GeneB ~= GeneC, potentially meaning that genes A & B coregulate to produce C, or there is some other nonlinear relationship. Cubé then recursively feeds outputs back into itself to great a gene network.

![Cubé](https://github.com/connerlambden/Cube/raw/main/images/cube_network_genes_discovery.png)

### Install


`$ python3 -m pip install git+https://github.com/connerlambden/Cube.git`

### Running Cubé

```
from sc_cube import cube
import scanpy as sc
adata = sc.read_h5ad('my_expression_data.h5ad') # Load AnnData Object
go_files = ['BioPlanet_2019.tsv', 'GeneSigDB.tsv'] # Load Gene Signatures to Search In

cube.run_cube(adata=adata, seed_gene_1='ifng', seed_gene_2='tbx21', go_files=go_files, 
            out_directory='Cubé_Results', num_search_children=4, search_depth=2)
```

__[Example Outputs](https://github.com/connerlambden/Cube/blob/main/cube_example_results.zip?raw=true)__

### Parameters

__adata__: [AnnData Object](https://anndata.readthedocs.io/en/latest/) with logged expression matrix

__seed_gene_1__: Starting search gene of interest

__seed_gene_2__: Optional: Additional seed gene of interest to search for seed_gene_1 * seed_gene_2

__go_files__: List of Pathway files to search in. Each edge in Cubé requires all connected genes to be present in at least 2 pathways. [Examples To Download](https://github.com/connerlambden/Cube/tree/main/pathways) or [Download More From Enrichr](https://maayanlab.cloud/Enrichr/#libraries). Cubé will automatically convert pathway gene names to match the capitalization/case of genes in your data.

__out_directory__: Folder to put results

__num_search_children__: How many search children to add to the network on each iteration. For example, a value of 2 will add two children to each node.

__search_depth__: Recursive search depth. Values above 2 may take a long time to run

Memory: Depending on the size of your data, Cubé should have at least 32GiB of RAM or else you might get Segmentation Faults

### Outputs

__Cubé_data_table.csv__: Table showing the genes, pathways, and weight for each edge in the network. Positive correlations will have small edge weights and negative correlations will have large edge weights.

__*.graphml file__. Network file that can be visualized in programs like [Cytoscape](https://cytoscape.org/)

__Cubé_network.png__: Network visualization where green edges are positive correlation & red edges are negative correlation. For better visualizations, we recommend loading the .graphml file into [Cytoscape](https://cytoscape.org/)

### Visualizing The Product of 2 Genes Using Scanpy

```
import numpy as np
# Visualizing Product of 2 Genes using Scanpy (assuming adata.X is logged and sparse)
gene_1 = 'ifng'
gene_2 = 'tbx21'
adata_expressing_both = adata[(adata[:,gene_1].X.toarray().flatten() > 0) & (adata[:,gene_2].X.toarray().flatten() > 0),:]
adata_expressing_both.obs[gene_1 + ' * ' + gene_2] = np.exp(adata_expressing_both[:,gene_1].X.toarray() + adata_expressing_both[:,gene_2].X.toarray())
sc.pl.umap(adata_expressing_both, color=[gene_1 + ' * ' + gene_2])
```


### Why Cubé?

![Cubé](https://github.com/connerlambden/Cube/raw/main/images/gata3_cube_gene_network.png)


Single-cell RNA sequencing has allowed for unprecedented resolution into the transcriptome of single cells, however the sheer complexity of the data and high rates of dropout have posed interpretive and computational challenges to create biological meanings and gene relationships. Many methods have been proposed for inferring gene regulatory networks, leading to sometimes dramatic differences depending upon the initial assumptions made 😬. Even in the case of unsupervised learning ([UMAP](https://umap-learn.readthedocs.io/en/latest/)) or clustering ([Leiden](https://github.com/vtraag/leidenalg)), it’s not clear how to balance local/global structure or what data features are most important. Additionally, these “black-box” machine learning methods are closed to scrutiny of their inner workings and cannot explicate logical, understandable steps and tend to be fragile to model parameters. Cubé addresses the dropout issue by only comparing sets of genes together in cells that have nonzero expression in all cells. This removes the need for biased imputation methods and focuses each relationship to relevant cells. Cubé addresses the interpretability problem by presenting solutions in the form of expression(gene1) ~= expression(gene2) * expression(gene3) which succinctly express nonlinear relationships between specific genes in an understandable way without any pesky parameters. Since Cubé samples from the space of all possible nonlinear gene-gene pairs, results have high representational capacity and low ambiguity. Cubé is a descriptive search algorithm that optimizes for biologically & statistically informed gene patterns.

### How It Works

![Cubé](https://github.com/connerlambden/Cube/raw/main/images/cube_gene_regulatory_network.png)

Special Thanks to [Vijay Kuchroo](https://kuchroolab.bwh.harvard.edu/) & [Ana Anderson](https://anacandersonlab.com/)