# SSPPI
A novel module applied to the prediction of protein-protein interaction sites using se-quence and spatial geometric features
# Requirements
* [Python](https://www.python.org/) (3.8.13).
* [reduce](http://kinemage.biochem.duke.edu/software/reduce.php) (3.23).
* [MSMS](http://mgltools.scripps.edu/packages/MSMS/) (2.6.1).
* [BioPython](https://github.com/biopython/biopython) (1.66). 
* [PyMesh](https://github.com/PyMesh/PyMesh) (0.1.14).
* PDB2PQR (2.1.1) 
* [APBS](http://www.poissonboltzmann.org/) (1.5).
* [open3D](https://github.com/IntelVCL/Open3D) (0.5.0.0).
* [Paddlepaddle](https://www.paddlepaddle.org.cn/).
# Usage
1.use pdb_preatreat.py to get the spatial geometric features, make sure that the corresponding files in the directory data_preparation/raw_pdbs/ and abfa/, the result will be produce at the oridata/.

2.take the corresponding PSSM into the pssm/

3.'python net.py' to start train
