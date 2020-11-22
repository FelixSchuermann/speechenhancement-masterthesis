# Masters Thesis on Deep Learning Methods for Speech Enhancement

The thesis is available in german language.

Speech enhancement is achieved with feed forward neural networks in Tensorflow 1.14 with python.

Herefore single STFT gain vectors are predicted from context windows:

 <img src="images\contextwindow.jpg" alt="contextwindow"  />

A novel architecture containing Subband 1D-Convolutions followed by LSTMs is presented:

 <img src="images\1dconvlstm.jpg" alt="1dconvlstm"  />

Text me if you are interested in trained weight files.

#### Abstract

Speech enhancement remains an active research topic as its demand increased for
modern communication systems as well as automatic speech recognition (ASR) in
ever more versatile acoustic environments. Deep learning based artificial neural networks offer solutions to the drawbacks of classic statistical based algorithms. This
thesis herefore illustrates the historic development of speech enhancement starting
with spectral subtraction and the algorithms of Eprahim and Malah and going further to state-of- the-art DL systems like DeepXi and SEGAN. Furthermore the
fundamentals of modern speech enhancement, such as Ideal Ratio/Binary Masks
(IRM/IBM) are derived aswell as deep learning techniques which are commonly
used in speech enhancement naming ResNets, LSTMs and dilated CNNs among
others. Hereafter novel architectures based on these are proposed and evaluated
with objective metrics such as PESQ, STOI and SDR. Here it is shown that a training strategy based on sub-datasets with learning rate decay can improve PESQ
metrics up to 0.25 in contrast to standard training strategies. Furthermore adjusted
loss functions such as PMSQE are evaluated in high spectral variance vehicular environments. Accordingly an adaptation of the loss function for use in masking-based
systems is proposed for the mentioned environment in which MSE loss functions
may fail. Furthermore a formulation of the a-priori SNR as training target and post
processing methods such as the global variance equalization are shown and evaluated. Finally the findings are summarized and compared to state-of-the-art systems.
Moreover future prospects for speech enhancement are given.  
