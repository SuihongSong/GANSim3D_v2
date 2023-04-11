## GANSim3D_v2
### An updated version of our original GANSim3D codes, with easier set up of training data dimensions, conditioning choices, etc.

Original GANSim3D repositary (https://github.com/SuihongSong/GeoModeling_GANSim-3D_For_large_arbitrary_reservoirs) is a TensorFlow implementation of the following paper:

> **GANSim-3D for conditional geomodelling: theory and field application**<br>
> Suihong Song (PengCheng Lab, Stanford, and CUPB), Tapan Mukerji (Stanford), Jiagen Hou (CUPB), Dongxiao Zhang (PengCheng Lab), Xinrui Lyu (Sinopec) <br>
> CUPB: China University of Petroleum - Beijing

> Available at https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021WR031865 and my researchgate profile https://www.researchgate.net/profile/Suihong_Song


## System requirements

* Both Linux and Windows are supported, but Linux is suggested.
* 64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.14.3 or newer.
* TensorFlow 1.10.0 or newer with GPU support. 
* (NOTE: the codes can be run with TensorFlow2.x environment after adjusting several lines of codes, see 'Codes adjustments for TensorFlow 2'. The uploaded codes here have been revised to run within TensorFlow2.x environment - tf2.6.2 + cudnn 8.1.1.33 + cuda 11.2.0).
* One or more high-end NVIDIA GPUs.
* Codes are not compatible with A100 GPUs currently. 
* NVIDIA driver 391.35 or newer, CUDA toolkit 9.0 or newer, cuDNN 7.4.1 or newer.


## License
Most code files of this study are derived from the original Progressive GANs work (https://github.com/tkarras/progressive_growing_of_gans), but we have largely amended the original codes, especially networks.py, loss.py, dataset.py, and train.py. The original Progressive GANs codes are under license of Attribution-NonCommercial 4.0 International (https://creativecommons.org/licenses/by-nc/4.0/). Other materials produced by us are under MIT license.


Please give appropriate credit to our work, if it is valuable for you to some extent.
