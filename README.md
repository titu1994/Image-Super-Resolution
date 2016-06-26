# Image Super Resolution using Convolutional Neural Networks

Implementation of Image Super Resolution CNN in Keras from the paper 
<i><a href="https://arxiv.org/pdf/1501.00092.pdf">Image Super-Resolution Using Deep Convolutional Networks</a></i>. Also contains a model that outperforms the above mentioned model, termed Expanded Super Resolution.

## Model Architecture
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/SRCNN.png" height=100% width=25%>
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/ESRCNN.png" height=100% width=50%>
<br>
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/Denoise.png" height=100% width=25%>

The model on the left is the simplest model of the ones described in the paper above, consisting of the 9-1-5 model.
Larger architecures can be easily made, but come at the cost of execution time, especially on CPU.

However there are some differences from the original paper:
<br><b>[1]</b> Used the AdaDelta optimizer instead of RMSProp.
<br><b>[2]</b> This model contains some 21,000 parameters, more than the 8,400 of the original paper.

On the right is the "Expanded SRCNN", which performs slightly better on Set5 (PSNR 33.37 dB vs 32.4 dB) than the default SRCNN model.

On the bottom is the "Denoising Auto Encoder SR", which performs even better than Expanded SRCNN on Set5 (PSNR 34.88 dB vs 33.37 dB).

It is to be noted that the original models underperform compared to the results posted in the paper. This maybe due to the only 91 images being the training set compared to the entire ILSVR 2013 image set. It still performs well, however images are slightly noisy.

## Usage
The model weights are already provided, therefore simply running :<br>
`python main.py "imgpath"`, where imgpath is a full path to the image.

The default model is SRCNN, which underperforms compared to the Expanded SRCNN or Denoising SR. To switch models,<br>
`python main.py "imgpath" --model="type"`, where type = sr, esr or dsr

If the scaling factor needs to be altered then :<br>
`python main.py "imgpath" --scale=s`, where s can be any number. Default s = 2

If the intermediate step (bilinear scaled image) is needed, then:<br>
`python main.py "imgpath" --scale=s --save_intermediate="True"`

If you wish to train the network on your own data set, follow these steps (Performance may vary) :
<br><b>[1]</b> Save all of your input images of any size in the <b>"input_images"</b> folder
<br><b>[2]</b> Open img_utils.py and manually compute the <b>nb_images</b>, located at line 10. 
<br>(nb_images = 400 * number of images in the "input_images" folder)
<br><b>[3]</b> Run img_utils.py function, `transform_images(input_path)`. By default, input_path is input_images path.
<br><b>[4]</b> Open <b>ImageSRModel.py</b> and un-comment the 3 lines at model.fit(...). 
<br><b>Note: It may be usefull to save the original weights in some other location</b>
<br><b>[4]</b> Execute ImageSRModel.py to begin training. GPU is recomended, althoug if small number of images are there then not required.

## Caveats
Very large images may not work with the GPU. Therefore, 
<br>[1] If using Theano, set device="cpu" and cnmem=0.0 in theanorc.txt
<br>[2] If using Tensorflor, set it to cpu mode

Denoising Auto Encoder requires MaxPooling and subsequent UpSampling of the input. Since there are 3 MaxPooling and 3 UpSampling layers, therefore the image size must be multiples of 8. 

In case the image size is not a multiple of 8, the image will be auto scaled to the nearest approximation of required size and then Denoising Auto Encoder upsampling will be performed.

## Examples
There are 14 extra images provided in results, 2 of which (Monarch Butterfly and Zebra) have been scaled using both bilinear, SRCNN, ESRCNN and DSRCNN.

### Monarch Butterfly
Bilinear
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/monarch_intermediate.jpg" width=25% height=25%> SRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/monarch_sr(2x).jpg" width=25% height=25%> ESRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/monarch_esr(2x).jpg" width=25% height=25%> 
DSRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/monarch_denoise(2x).jpg" height=25% width=25%>

### Zebra
Bilinear
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/zebra_intermediate.jpg" width=25% height=25%> SRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/zebra_sr(2x).jpg" width=25% height=25%>
ESRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/zebra_esr(2x).jpg" width=25% height=25%>
DSRCNN
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/results/zebra_denoise(2x).jpg" width=25% height=25%>
