# Image Super Resolution using Convolutional Neural Networks

Implementation of Image Super Resolution CNN in Keras from the paper 
<i><a href="https://arxiv.org/pdf/1501.00092v3.pdf">Image Super-Resolution Using Deep Convolutional Networks</a></i>. <br> 
Also contains a model that outperforms the above mentioned model, termed Expanded Super Resolution, and another model termed Denoiseing Auto Encoder SRCNN which outperforms both of the above models.

## Model Architecture
### Super Resolution CNN (SRCNN)
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/SRCNN.png" height=100% width=25%>

The model above is the simplest model of the ones described in the paper above, consisting of the 9-1-5 model.
Larger architectures can be easily made, but come at the cost of execution time, especially on CPU.

However there are some differences from the original paper:
<br><b>[1]</b> Used the AdaDelta optimizer instead of RMSProp.
<br><b>[2]</b> This model contains some 21,000 parameters, more than the 8,400 of the original paper.

It is to be noted that the original models underperform compared to the results posted in the paper. This may be due to the only 91 images being the training set compared to the entire ILSVR 2013 image set. It still performs well, however images are slightly noisy.

### Expanded Super Resolution CNN (ESRCNN)
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/ESRCNN.png" height=100% width=75%>

The above is called "Expanded SRCNN", which performs slightly better than the default SRCNN model on Set5 (PSNR 33.37 dB vs 32.4 dB).

The "Expansion" occurs in the intermediate hidden layer, in which instead of just 1x1 kernels, we also use 3x3 and 5x5 kernels in order to maximize information learned from the layer. The outputs of this layer are then averaged, in order to construct more robust upscaled images.

### Denoiseing (Auto Encoder) Super Resolution CNN (DSRCNN)
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/Denoise.png" height=100% width=40%>

The above is the "Denoiseing Auto Encoder SRCNN", which performs even better than Expanded SRCNN on Set5 (PSNR 34.88 dB vs 33.37 dB).

This model uses bridge connections between the convolutional layers of the same level in order to speed up convergence and improve output results. The bridge connections are averaged to be more robust. 

Since the training images are passed through a gausian filter (sigma = 0.5), then downscaled to 1/3rd the size, then upscaled to the original 33x33 size images, the images can be considered "noisy". Thus, this auto encoder quickly improves on the earlier results, and reduces the noisy output image problem faced by the simpler SRCNN model.

## Usage
The model weights are already provided, therefore simply running :<br>
`python main.py "imgpath"`, where imgpath is a full path to the image.

The default model is DSRCNN, which outperforms the other two models. To switch models,<br>
`python main.py "imgpath" --model="type"`, where type = `sr`, `esr` or `dsr`

If the scaling factor needs to be altered then :<br>
`python main.py "imgpath" --scale=s`, where s can be any number. Default `s = 2`

If the intermediate step (bilinear scaled image) is needed, then:<br>
`python main.py "imgpath" --scale=s --save_intermediate="True"`

## Training
If you wish to train the network on your own data set, follow these steps (Performance may vary) :
<br><b>[1]</b> Save all of your input images of any size in the <b>"input_images"</b> folder
<br><b>[2]</b> Open img_utils.py and manually compute the <b>nb_images</b>, located at line 10. 
<br>(<b>nb_images = 400 * number of images in the "input_images" folder</b>). This is needed to efficiently create the sub-images.
<br><b>[3]</b> Run img_utils.py function, `transform_images(input_path)`. By default, input_path is "input_images" path.
<br><b>[4]</b> Open <b>tests.py</b> and un-comment the lines at model.fit(...), where model can be sr, esr or dsr. 
<br><b>Note: It may be useful to save the original weights in some other location.</b>
<br><b>[4]</b> Execute tests.py to begin training. GPU is recommended, although if small number of images are provided then GPU may not be required.

## Caveats
Very large images may not work with the GPU. Therefore, 
<br>[1] If using Theano, set device="cpu" and cnmem=0.0 in theanorc.txt
<br>[2] If using Tensorflow, set it to cpu mode

Denoiseing Auto Encoder requires MaxPooling and subsequent UpSampling of the input. Since there are 3 MaxPooling and 3 UpSampling layers, therefore the image size must be multiples of 8. 

In case the image size is not a multiple of 8, the image will be auto scaled to the nearest approximation of required size and then Denoiseing Auto Encoder upsampling will be performed.

## Examples
There are 14 extra images provided in results, 2 of which (Monarch Butterfly and Zebra) have been scaled using both bilinear, SRCNN, ESRCNN and DSRCNN.

### Monarch Butterfly
Bilinear
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/monarch_intermediate.jpg" width=25% height=25%> SRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/monarch_sr(2x).jpg" width=25% height=25%> <br>ESRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/monarch_esr(2x).jpg" width=25% height=25%> 
DSRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/monarch_denoise(2x).jpg" height=25% width=25%>

### Zebra
Bilinear
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/zebra_intermediate.jpg" width=25% height=25%> SRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/zebra_sr(2x).jpg" width=25% height=25%>
<br>ESRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/zebra_esr(2x).jpg" width=25% height=25%>
DSRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/zebra_denoise(2x).jpg" width=25% height=25%>
