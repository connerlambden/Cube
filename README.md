# Cubé
![Cubé](https://github.com/connerlambden/Cube/raw/main/cube%CC%81_header.jpg)
## Simple Nonparametric Gene Network Search Algorithm

## How It Works
Given a single-cell dataset and an input gene(s), Cubé looks for simple & nonlinear gene-gene relationships to construct a regulation network informed by prior gene signatures. For example, Cubé might give you the result that GeneA * GeneB ~= GeneC

## Install

How to install Cubé:

`$ pip3 install git+https://github.com/connerlambden/Cube.git`

## Running Cubé

```
from sc_cube import cube
import scanpy as sc
adata = sc.read_h5ad('my_expression_data.h5ad')
cube.run_cube(adata=adata)

```

## Introduction

Single-cell RNA sequencing has allowed for unprecedented resolution into the transcriptome of single cells, however the sheer complexity of the data and high rates of dropout have posed interpretive and computational challenges to create biological meanings and gene relationships. Many methods have been proposed for inferring gene regulatory networks, leading to sometimes dramatic differences depending upon the initial assumptions made 😬. Even in the case of unsupervised learning [UMAP](https://umap-learn.readthedocs.io/en/latest/) or clustering [Leiden](https://github.com/vtraag/leidenalg), it’s not clear how to balance local/global structure or what data features are most important. Additionally, these “black-box” machine learning methods are closed to scrutiny of their inner workings and cannot explicate logical, understandable steps and tend to be fragile to model parameters. Cubé addresses the dropout issue by only comparing sets of genes together in cells that have nonzero expression in all cells. This removes the need for biased imputation methods and focuses each relationship to relevant cells. Cubé addresses the interpretability problem by presenting solutions in the form of expression(gene1) ~= expression(gene2) * expression(gene3) which succinctly express nonlinear relationships between specific genes in an understandable way without any pesky parameters. Since Cubé samples from the space of all possible nonlinear gene-gene pairs, results have high representational capacity and low ambiguity. Cubé is a descriptive search algorithm that optimizes for biologically & statistically informed gene patterns.