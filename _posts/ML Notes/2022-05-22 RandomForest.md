---
title: "Additional topics for Random Forest"
category: Machine Learning Notes
tags:
  - [Machine Learning, Notes]
permalink: /mlnote/rf/
last_modified_at: Now

layout: single_v2
katex: true
---


## Number of Estimators
Random forest uses bagging and random subspace method (picking a sample of features rather than all of them, in other words - attribute bagging) to grow a tree. 
If the number of observations is large, but the number of trees is too small, then some observations will be predicted only once or even not at all. If the number of predictors is large but the number of trees is too small, then some features can (theoretically) be missed in all subspaces used. 
Both cases results in the decrease of random forest predictive power. But the last is a rather extreme case, since the selection of subspace is performed at each node.

During classification the subspace dimensionality is $\sqrt{p}$ (rather small, 𝑝 is the total number of predictors) by default, but a tree contains many nodes. During regression the subspace dimensionality is 𝑝/3 (large enough) by default, though a tree contains fewer nodes. So the optimal number of trees in a random forest depends on the number of predictors only in extreme cases.

The official page of the algorithm states that random forest does not overfit, and you can use as much trees as you want. But Mark R. Segal (April 14 2004. "Machine Learning Benchmarks and Random Forest Regression." Center for Bioinformatics & Molecular Biostatistics) has found that it overfits for some noisy datasets. So to obtain optimal number you can try training random forest at a grid of ntree parameter (simple, but more CPU-consuming) or build one random forest with many trees with keep.inbag, calculate out-of-bag (OOB) error rates for first 𝑛 trees (where 𝑛 changes from 1 to ntree) and plot OOB error rate vs. number of trees (more complex, but less CPU-consuming).