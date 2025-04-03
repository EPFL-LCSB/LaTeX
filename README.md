Latent Space Manipulation of Generative Neural Networks
=====


Code and user instructions for usage in Python 3.6
Paper : Subham Choudhury, Ilias Toumpe, Oussama Gabouj, Vassily hatzimanikatis and Ljubisa Miskovic. "Generative Approaches to Kinetic Parameter Inference in Metabolic Networks via Latent Space Exploration"
DOI: xxx

Requirements
------------
All the code implemented in these studies utilizes the ReKinDLE environment (https://github.com/EPFL-LCSB/rekindle, https://gitlab.com/EPFL-LCSB/rekindle ) and RENAISSANCE toolbox (https://github.com/EPFL-LCSB/RENAISSANCE, https://gitlab.com/EPFL-LCSB/renaissance). 

Installation
-----------------------
Please follow the installation instructions from the repositories listed above to install ReKinDLE and RENAISSANCE. 


Data
=====
All the data for this paper can be downloaded from: xxx

Quick start
===========

To implement latent space manipulation, the user only needs the functionality of custom input and transformations of the latent noise input in REKINDLE and RENAISSANCE. This has already been implemented in the RENAISSANCE code as detailed in
https://github.com/EPFL-LCSB/RENAISSANCE/blob/master/docker/work/1-renaissance.py and https://github.com/EPFL-LCSB/RENAISSANCE/blob/master/docker/work/2-load_and_generate.py.  A REKINDLE implementation can be found as REKINDLE_latent.py in this repository. Download all the data in the same directory from the link above.

Notebooks
=============

Once the data has been downloaded, marked JuPyter Notebooks have been made available for conducting all the analyses and the for generating all the figures in the manuscript and supplementary material.   
   
License
========

The software in this repository is put under an APACHE-2.0 licensing scheme - please see the `LICENSE <https://github.com/EPFL-LCSB/pytfa/blob/master/LICENSE.txt>`_ file for more details.



