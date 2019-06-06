# ProteinPatchAnalysis
Codes to take a PDB and parcel it into a set of patches based on a voxel representation

Organization:
ProteinPatchAnalysis contains python scripts to take PDBs from the folder ProteinPatchAnalysis/pdb and convert them to a pickle file
of a numpy array containing voxelized representations of "protein surface patches" for the proteins contianed in the PDBs. This array has
dimensions n x c x l x l x l, where n is the number of samples, c is the number of channels (which refer here to occupancy of different
atom types), and l refers to the size length in voxels.

ProteinPatchAnalysis/AutoencoderNoteboks contains a series of jupyter noteboks containing code written using the Keras module to create
sample deep, convolutional, variational autoencoder neural networks for the MNIST dataset (2D bw images). In the future, we aim to use
this framework to develop a similar autoencoder for the samples obtained using the codes in ProteinPatchAnalysis. This will require
making our model 3D and introducing channels (in addition to tuning hyperparameters).
