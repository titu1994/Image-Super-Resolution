# Image Super Resolution using in Keras

Implementation of Image Super Resolution CNN in Keras from the paper 
<i><a href="https://arxiv.org/pdf/1501.00092v3.pdf">Image Super-Resolution Using Deep Convolutional Networks</a></i>.

Also contains models that outperforms the above mentioned model, termed Expanded Super Resolution, Denoiseing Auto Encoder SRCNN which outperforms both of the above models and Deep Denoise SR, which with certain limitations, outperforms all of the above.

## Setup
Since it uses deconvolutions, which are available only in the master branch of Keras and Theano at the moment, the requirements are as follows:

```
pip install git+https://github.com/fchollet/keras.git --upgrade

pip install git+https://github.com/Theano/Theano.git --upgrade --no-deps
```

## Usage
The model weights are already provided in the weights folder, therefore simply running :<br>
`python main.py "imgpath"`, where imgpath is a full path to the image.

The default model is DSRCNN (dsr), which outperforms the other three models. To switch models,<br>
`python main.py "imgpath" --model="type"`, where type = `sr`, `esr`, `dsr`, `ddsr`

If the scaling factor needs to be altered then :<br>
`python main.py "imgpath" --scale=s`, where s can be any number. Default `s = 2`

If the intermediate step (bilinear scaled image) is needed, then:<br>
`python main.py "imgpath" --scale=s --save_intermediate="True"`

## Window Helper
The windows_helper script contains a C# program for Windows to easily use the Super Resolution script using any of the available models.

## Parameters
```
--model : Can be one of "sr" (Image Super Resolution), "esr" (Expanded SR), "dsr" (Denoiseing Auto Encoder SR), "ddsr" (Deep Denoise SR)
--scale : Scaling factor can be any integer number. Default is 2x scaling.
--save_intermediate= : Save the intermediate results before applying the Super Resolution algorithm.
```

## Model Architecture
### Super Resolution CNN (SRCNN)
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/architectures/SRCNN.png" height=100% width=25%>

The model above is the simplest model of the ones described in the paper above, consisting of the 9-1-5 model.
Larger architectures can be easily made, but come at the cost of execution time, especially on CPU.

However there are some differences from the original paper:
<br><b>[1]</b> Used the AdaDelta optimizer instead of RMSProp.
<br><b>[2]</b> This model contains some 21,000 parameters, more than the 8,400 of the original paper.

It is to be noted that the original models underperform compared to the results posted in the paper. This may be due to the only 91 images being the training set compared to the entire ILSVR 2013 image set. It still performs well, however images are slightly noisy.

### Expanded Super Resolution CNN (ESRCNN)
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/architectures/ESRCNN.png" height=100% width=75%>

The above is called "Expanded SRCNN", which performs slightly better than the default SRCNN model on Set5 (PSNR 33.37 dB vs 32.4 dB).

The "Expansion" occurs in the intermediate hidden layer, in which instead of just 1x1 kernels, we also use 3x3 and 5x5 kernels in order to maximize information learned from the layer. The outputs of this layer are then averaged, in order to construct more robust upscaled images.

### Denoiseing (Auto Encoder) Super Resolution CNN (DSRCNN)
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/architectures/Denoise.png" height=100% width=40%>

The above is the "Denoiseing Auto Encoder SRCNN", which performs even better than Expanded SRCNN on Set5 (PSNR 36.28 dB vs 33.37 dB).

This model uses bridge connections between the convolutional layers of the same level in order to speed up convergence and improve output results. The bridge connections are averaged to be more robust. 

Since the training images are passed through a gausian filter (sigma = 0.5), then downscaled to 1/3rd the size, then upscaled to the original 33x33 size images, the images can be considered "noisy". Thus, this auto encoder quickly improves on the earlier results, and reduces the noisy output image problem faced by the simpler SRCNN model.

### Deep Denoiseing Super Resolution (DDSRCNN)
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/architectures/Deep Denoise.png" height=100% width=40%>

The above is the "Deep Denoiseing SRCNN", which is a modified form of the architecture described in the paper <a href="http://arxiv.org/abs/1606.08921">"Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections"</a> applied to image super-resolution. It can perform far better than even the Denoiseing SRCNN, but is currently not working properly.

Following the same principle as DSRCNN, it uses skip connections at same levels, but increases the depth of the network by cascading 2 CNNs at each level. This drastically increases the speed of learning, acheiving near 36.1~ dB validation PSNR value in the first 5 epochs on the 91 image dataset of the original SRCNN.

Similar to the paper <a href="http://arxiv.org/abs/1606.08921">Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections</a>, this can be considered a highly simplified and shallow model compared to the 30 layer architecture used in the above paper. However even with just 8 layers, DDSRCNN acheives very high PSNR score, close to the scores in the paper (37.56 dB for 30-layer architecture vs 37.46 dB for 8 layer DDSRCNN architecture) on Set5 validation images.

<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/architectures/DDSRCNN%20validation%20plot.png" width=100% height=100%>

## Training
If you wish to train the network on your own data set, follow these steps (Performance may vary) :
<br><b>[1]</b> Save all of your input images of any size in the <b>"input_images"</b> folder
<br><b>[2]</b> Run img_utils.py function, `transform_images(input_path)`. By default, input_path is "input_images" path.
<br><b>[3]</b> Open <b>tests.py</b> and un-comment the lines at model.fit(...), where model can be sr, esr or dsr. 
<br><b>Note: It may be useful to save the original weights in some other location.</b>
<br><b>[4]</b> Execute tests.py to begin training. GPU is recommended, although if small number of images are provided then GPU may not be required.

## Caveats
Very large images may not work with the GPU. Therefore, 
<br>[1] If using Theano, set device="cpu" and cnmem=0.0 in theanorc.txt
<br>[2] If using Tensorflow, set it to cpu mode

On the CPU, extremely high resolution images of the size upto 6000 x 6000 pixels can be handled if 16 GB RAM is provided. 

## Examples
There are 14 extra images provided in results, 2 of which (Monarch Butterfly and Zebra) have been scaled using both bilinear, SRCNN, ESRCNN and DSRCNN.

### Monarch Butterfly
Bilinear
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/monarch_intermediate.jpg" width=25% height=25%> SRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/monarch_sr(2x).jpg" width=25% height=25%> <br>ESRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/monarch_esr(2x).jpg" width=25% height=25%> 
DDSRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/monarch_ddsr(2x).png" height=25% width=25%>

### Zebra
Bilinear
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/zebra_intermediate.jpg" width=25% height=25%> SRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/zebra_sr(2x).jpg" width=25% height=25%>
<br>ESRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/zebra_esr(2x).jpg" width=25% height=25%>
DDSRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/zebra_ddsr(2x).png" width=25% height=25%>
