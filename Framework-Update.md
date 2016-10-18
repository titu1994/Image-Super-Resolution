# Super Resolution Framework

This project contains 3 main modules which have been almost completely reworked:
- img_utils : Contains training data set generation, validation data set generation and general image manipulation.
- advanced : Contains custom Keras regularizers and callbacks that are used 
- models: Contains the model creation and handling code. The main code base.

Two minor modules are :
- test : Contains code which can be uncommented to plot, train, evaluate and test the models
- main : Contains the image upscaling features (basically creates model, loads weights, and calls upscale method)

# Major Changes

## Img_Utils.py
`img_utils.py` used to contain various method to create image data sets to train. However, it's glaring drawback was the number of images that it could manipulate. 

Previously:
- It would create 96k images of size 33x33 which were highly redundant
- It loaded all 96k images into a single numpy array, which increased startup time
- It failed if you did not have over 8GB RAM to load these images
- It used stride=1 patching mechanism to get sub-images, which was highly inefficient.

Now:
- It creates rougly 24k images
- It uses generators to load batch_size number of images, determined by the batch_index_generator. 
- It can handle any number of training images.
- It wont run out of memory due to loading only a subset of images
- It uses stride=16 and generates images of size 32x32
- It can create 256 sub images out of each image provided, therefore larger image datasets like ImageNet and MS COCO can be used to create image training sets.
- It saves them in the Users home path, with a "Image Super Resolution Dataset" sub-directory

## Advanced.py
`advanced.py` now contains TVRegularization code, which will be used for ResNetSR models.

## Models.py
`models.py` has been drastically refactored to allow for new Image Super Resolution architectures. It now provides a more general framework to add Super Resolution models.

Below, 'True' upscaling models refer to those models who take a (width X height) image and internally upscale this image such that output is of size (width * scale X height * scale). Examples : Efficient Sub-Pixel CNN (ESPCNN) / SRGAN

'General' upscaling models refer to those models which take a (width X height) image, perform pooling and subsequent upsampling to provide an output that is of size (width X height), which is sharper and clearer than blurred input.

Previously:
- LearningModel was the base class for all other SR models.
  - `create_model` method did not support 'true' upscaling models.
  - `fit` method drawbacks :<br>
    - Required entire training dataset as input (could cause memory errors)
    - No support for 'true' upscaling models
  - `evaluate` method performed evaluation of model on sub images of Set5. This gave inacurate results.
- Auto Encoder style SR models had to have special parameters to load image of size which could be divisible by 4.
- All models were trained by "Adadelta" optimizer with default parameters.
- `upscale` method needed special parametrs to handle Auto Encoder style SR models.

Now:
- BaseSuperResolutionModel (renamed) is the base class of all other SR models:
  - `create_model` method supports 'true' upscaling models via `small_train_images` parameter
  - `fit` method improvements: <br>
    - No longer requires entire set. Uses internal iterator for loading training images.
    - Supports 'true' upscaling models with `small_train_images` parameter
  - `evaluate` method now uses the full size image of the Set5 for validation without patching. More accurate test results.
- Auto Encoder style SR models can simply add their name to the list of `auto_encoder_models` to assert that image size is correct
- All models will be trained by "Adam" optimizer with lower learning rate.
- `upscale` method now handles Auto Encoder style SR models properly
  
## Test.py
Now updated to use generators for training data instead of loading all the training data at once.

# Repercussions of Above Changes
Due to such large changes to the image handling, training dataset changes and how the models are created and evaluated, all model weights are now under performing unless they are re-trained.

Due to the above changes, it is recommended to :
- Delete the entire training data numpy arrays 
- Delete the older 96k training images
- Run the img_utils.py script again to generate the new dataset and the correct file structure

## Models which have been updated
- Image Super Resolution Model (SR model)
- Expantion Super Resolution Model (ESR model)

## Models which will be updated soon
- Denoising Auto Encoder SR model (DSR model)
- Deep Denoise SR model (DDSR model)

## New models which can now be trained 
- Residual Network SR (ResNetSR model) (RNSR)
- Efficient Sub-Pixel Convolutional Neural Network model (ESPCNN) [Once Sub-Pixel Convolution is added to Keras]


