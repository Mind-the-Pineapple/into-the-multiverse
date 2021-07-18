# Into the Multiverse

Code for the paper "Neuroimaging: Into the multiverse".
![Image](figures/overview_analysis.png)

## Abstract
<p align="justify">
For most neuroimaging questions the range of possible analytic choices makes it
unclear how to evaluate conclusions from any single analytic method may be
misleading. To address this issue one possible approach is a multiverse
analysis evaluating all possible analyses, however, this can be computationally
challenging and sequential analyses on the same data can compromise inferential
and predictive power. Here, we establish how active learning on a
low-dimensional space capturing the inter-relationships between pipelines can
efficiently approximate the whole multiverse of analyses. This approach balances
the benefits of a multiverse analysis without incurring the cost on
computational and statistical power. We illustrate this approach with two
functional MRI datasets (predicting brain age and autism diagnosis)
demonstrating how a multiverse of analyses can be efficiently navigated and
mapped out using active learning. Furthermore, our presented approach not only
identifies the subset of analysis techniques that are best able to predict age
or classify individuals with autism spectrum disorder and healthy controls, but
it also allows the relationships between analyses to be quantified.
</p>

## Run the code
The code can be run using Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mind-the-Pineapple/into-the-multiverse/blob/master/notebooks/multiverse_analysis_regression.ipynb)

## Citation
If you find this code useful, please cite:

**Neuroimaging: Into the Multiverse,**
*Jessica Dafflon, Pedro F. Da Costa, František Váša, Ricardo Pio Monti, Danilo Bzdok, Peter J. Hellyer, Federico Turkheimer, Jonathan Smallwood, Emily Jones, Robert Leech,* bioRxiv 2020.10.29.359778; doi: [https://doi.org/10.1101/2020.10.29.359778](https://www.biorxiv.org/content/10.1101/2020.10.29.359778v1)

