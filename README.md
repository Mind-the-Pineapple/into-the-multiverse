# Into the Multiverse

Code for the paper "Neuroimaging: Into the multiverse".
![Image](figures/overview_analysis.png)

## Abstract
For most neuroimaging questions, there exist a huge range of analysis decisions
that need to be made: any single analysis approach considered in isolation may
lead to misleadingly specific interpretations or may be sub-optimal for the
data. One response is to perform a multiverse analysis, evaluating all possible
analysis decisions; however, this will be computationally challenging and the
repeated sequential analyses on the same data will compromise inferential and
predictive power. Here, we demonstrate how using active learning on a
low-dimensional space that captures the inter-relationships between analysis
approaches, can be used to approximate the whole multiverse of analyses, while
only sparsely sampling the space. This allows for the benefits of the multiverse
analysis while limiting statistical and computational challenges. We illustrate
the approach with a functional MRI dataset of functional connectivity across
adolescence; we show how a multiverse of graph theoretic and simple
pre-processing steps can be efficiently navigated using active learning,
simultaneously finding the subset of analysis approaches which can be used to
strongly predict participants’ ages as well as approximating how all of the
analysis approaches perform.

## Run the code
All the code can be run using Colab:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Mind-the-Pineapple/into-the-multiverse/blob/master/notebooks/multiverse_analysis.ipynb)

## Citation
If you find this code useful, please cite:

