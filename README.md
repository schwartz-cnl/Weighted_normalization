# Weighted_normalization


Python code for end to end training of normalization models.


"normalizations.py" contains TF2.0 implementation of normalization layers, including Weighted Normalization - center and surround (WN_s), Weighted Normalization - center only (WN_c), their fixed-weight version WN_s_fix and WN_c_fix, and a heuristic of Flexible Normalization (FNM).


"Alexnet_models.py", "Cifar_models.py", "TexturedMNIST_models.py" contain TF2.0 implementation of CNN models.


We also introduce a new dataset, namely textured MNIST. The folder "naturalistic" contains textures that were used to generate the dataset. "Textured_MNIST.py" is the code for generating the dataset.


Reference:


 X Pan, E Kartal, L Gonzalo Giraldo, O Schwartz, “Brain-Inspired Weighted Normalization for CNN Image Classification” ICLR Workshop: How Can Findings About The Brain Improve AI Systems, 2021.
 
 
 R C Cagli, A Kohn*, O Schwartz*, Flexible Gating of Contextual Modulation During Natural Vision. Nature Neuroscience 8(11):1648-55, 2015.
