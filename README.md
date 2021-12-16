# Weighted_normalization

This is the official repository of the paper. X Pan, E Kartal, LGonzalo Giraldo, O Schwartz, “Brain-Inspired Weighted Normalization for CNN Image Classification” ICLR Workshop: How Can Findings About The Brain Improve AI Systems, 2021.


"normalizations.py" contains TF2.0 implementation of normalization layers introduced in the paper, includes Weighted Normalization - center and surround (WN_s), Weighted Nomalization - center only (WN_c), and their fixed-weight version WN_s_fix and WN_c_fix.


"Alexnet_models.py", "Cifar_models.py", "TexturedMNIST_models.py" contains TF2.0 implementation of CNN models used in the experiments. Trained model weights are avaible upon request.


In the paper, we introducted a new dataset, namely textured MNIST. The folder "naturalistic" contains textures that used to generate the dataset. "Textured_MNIST.py" is the code for generating the dataset.
