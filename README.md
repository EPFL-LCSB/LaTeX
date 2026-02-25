Latent Space Exploration of Generative Neural Networks
=====


Code and user instructions for usage in Python 3.6
Paper: Subham Choudhury, Ilias Toumpe, Oussama Gabouj, Jakob Sebastian Behler, Vassily Hatzimanikatis, and Ljubisa Miskovic. "Generative Approaches to Kinetic Parameter Inference in Metabolic Networks via Latent Space Exploration"
DOI: xxx

Requirements
------------
All the code implemented in these studies utilizes the ReKinDLE environment (https://github.com/EPFL-LCSB/rekindle, https://gitlab.com/EPFL-LCSB/rekindle ) and RENAISSANCE toolbox (https://github.com/EPFL-LCSB/RENAISSANCE). 

Hardware requirements
---------------------
- Latent space exploration and training generators with RENAISSANCE: CPU is sufficient.
- Training generators with ReKinDLE: a CUDA-capable GPU is recommended.

Installation
-----------------------
Please follow the installation instructions from the repositories listed above to install ReKinDLE and RENAISSANCE. 

Data
=====
All the data for this paper can be downloaded from: https://doi.org/10.5281/zenodo.15131706.

Quick start
===========
To implement latent space manipulation, the user only needs the functionality of custom input and transformations of the latent noise input in REKINDLE and RENAISSANCE. This has already been implemented in the RENAISSANCE code as detailed in
https://github.com/EPFL-LCSB/RENAISSANCE/blob/master/docker/work/1-renaissance.py and https://github.com/EPFL-LCSB/RENAISSANCE/blob/master/docker/work/2-load_and_generate.py.  A REKINDLE implementation can be found as REKINDLE_latent.py in this repository. Download all the data in the same directory from the link above.

Notebooks
=============
Once the data has been downloaded, Jupyter Notebooks have been prepared for conducting all analyses and generating all figures for the manuscript and supplementary material.   
   
License
========
The software in this repository is put under an APACHE-2.0 licensing scheme - please see the `LICENSE <https://github.com/EPFL-LCSB/pytfa/blob/master/LICENSE.txt>`_ file for more details.



