# MPRAGER - MP2RAGE to pseudo-MPRAGE with minimal dependencies

This is a quick python script implementing a similar approach to https://github.com/srikash/presurfer 's "MPRAGEise" functionality. It is intended to mimic MPRAGE contrast using only the second inversion pulse image and the UNI image as output by on current Siemens scanners by default at time of writing. If you have the ability to use the original complex images, I would recommend trying the robust MP2RAGE reconstruction here: https://github.com/baxpr/mp2rage . That repo also contains a nice overview of the relevant citations.
