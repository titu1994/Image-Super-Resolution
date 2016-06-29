from keras.models import Model
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D
import keras.callbacks as callbacks
import numpy as np
import keras.backend as K
import img_utils

path_X = r"output_images_X\\"
path_Y = r"output_images_Y\\"

import os

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * np.log10(K.mean(K.square(y_pred - y_true)))

class LearningModel(object):

    def __init__(self, model_name):
        """
        Base model to provide a standard interface of adding Super Resolution models
        """
        self.model = None
        self.model_name = model_name

        self.n1 = 64
        self.n2 = 32

    def create_model(self, height=33, width=33, channels=3, load_weights=False) -> Model:
        """
        Subclass dependent implementation.
        """
        pass

    def fit(self, trainX, trainY, weight_fn, batch_size=128, nb_epochs=100) -> Model:
        """
        Standard method to train any of the models.
        The last 2000 images belong to the Set5 Validation images.
        """
        if self.model == None: self.create_model()

        self.model.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epochs,
                callbacks=[callbacks.ModelCheckpoint(weight_fn, monitor='val_PSNRLoss', save_best_only=True,  mode='max')],
                validation_split=2000. / 38400)

        return self.model

    def evaluate(self, is_denoise=False):
        """
        Evaluates the model on the Set5 Validation images
        """
        if self.model == None: self.create_model(load_weights=True)

        if not is_denoise:
            trainX, trainY = img_utils.loadImages()
        else:
            trainX, trainY = img_utils.loadDenoisingImages()

        error = self.model.evaluate(trainX[-2000:], trainY[-2000:])
        print("Mean Squared Error (Compared to bilinear upscaled version) : ", error[0])
        print("Peak Signal to Noise Ratio (Compared to bilinear upscaled version) : ", error[1])

    def upscale(self, img_path, scale_factor=2, save_intermediate=False, return_image=False, suffix="scaled", verbose=True,
            evaluate=True):
        """
        Standard method to upscale an image.

        :param img_path:  path to the image
        :param scale_factor: scale factor can be any value, usually is an integer
        :param save_intermediate: saves the intermediate upscaled image (bilinear upscale)
        :param return_image: returns a image of shape (height, width, channels).
        :param suffix: suffix of upscaled image
        :param verbose: whether to print messages
        :param evaluate: evaluate the upscaled image on the original image.
        """
        import os
        import theano
        from scipy.misc import imread, imresize, imsave

        # Flag that may cause crash if algo_fwd = 'time_once'
        theano.config.dnn.conv.algo_fwd = 'small'

        # Destination path
        path = os.path.splitext(img_path)
        filename = path[0] + "_" + suffix + "(%dx)" % (scale_factor) + path[1]

        # Read image
        scale_factor = int(scale_factor)
        true_img = imread(img_path, mode='RGB')
        init_height, init_width = true_img.shape[0], true_img.shape[1]
        if verbose: print("Old Size : ", true_img.shape)
        if verbose: print("New Size : (%d, %d, 3)" % (init_height * scale_factor, init_width * scale_factor))

        # Denoiseing SR needs image size to be divisible by 4 (2 MaxPooling ops, 2 UpSampling ops).
        # Also since we will be slicing the image, we need to compensate for that.
        if self.model_name == "Denoise AutoEncoder SR":
            if (init_height // scale_factor % 4 != 0) or (init_width // scale_factor % 4 != 0):
                print("Image shard size needs to be divisible by 4 to use denoise auto encoder.")
                print("Resizing")
                true_height = (init_height // scale_factor // 4) * 4 * scale_factor
                true_width = (init_width // scale_factor // 4) * 4 * scale_factor
                true_img = imresize(true_img, (true_height, true_width))

                print("Image has been modified to size (%d, %d)" % (true_height, true_width))

        # Slicing old image into scale_factor number of shards.
        true_shards = img_utils.split_image(true_img, scale_factor)

        nb_shards = true_shards.shape[0]
        true_height, true_width = true_shards.shape[1], true_shards.shape[2]
        print("Number of shards = %d, Shard Shape = (%d, %d)" % (nb_shards, true_height, true_width))

        model = None
        holder = None

        for i_shard in range(nb_shards):
            # Pre Upscale
            img = imresize(true_shards[i_shard], (true_height * scale_factor, true_width * scale_factor))

            # Save intermediate bilinear scaled image is needed for comparison.
            if save_intermediate:
                if verbose: print("Saving intermediate shard.")
                fn = path[0] + "_intermediate_%d_" % (i_shard+1) + path[1]
                imsave(fn, img)

            # Split the shard into multiple sub-shards
            shards = img_utils.split_image(img, scale_factor)

            # Compute new height
            height, width = shards.shape[1], shards.shape[2]

            # Create a holder for the sub shards
            if holder is None:
                holder = np.empty((scale_factor * scale_factor, height * scale_factor, width * scale_factor, 3))

            # Transpose and Process images
            img_conv = shards.transpose((0, 3, 1, 2)).astype('float64') / 255

            if model == None:
                model = self.create_model(height, width, load_weights=True)
                if verbose: print("Model loaded.")

            # Create prediction for image
            result = model.predict(img_conv, verbose=verbose)
            if verbose: print("Model finished upscaling shard %d." % (i_shard + 1))

            # Deprocess
            result = result.transpose((0, 2, 3, 1)).astype('float64') * 255
            result = img_utils.merge_images(result, scale_factor)
            result = np.clip(result, 0, 255).astype('uint8')

            # Store intermediate shard into holder
            holder[i_shard, :, :, :] = result

            if verbose: print("Completed shard %d" % (i_shard + 1))

        if verbose: print("Merging shards.")
        final_result = img_utils.merge_images(holder, scale_factor)

        if return_image:
            # Return the image without saving. Usefull for testing images.
            return final_result

        if verbose: print("Saving image.")
        imsave(filename, final_result)

        if evaluate:
            # Evaluate agains bilinear upscaled image. Just to measure if a good result was obtained.
            if verbose: print("Evaluating results.")

            # Shard true_img into peices
            shards = img_utils.split_image(true_img, scale_factor)
            true_height, true_width = shards.shape[1], shards.shape[2]

            shards = shards.transpose((0, 3, 1, 2)).astype('float64') / 255

            # Create a model to evaluate the shards and true image
            eval_model = self.create_model(true_height, true_width, load_weights=True)

            # Evaluating the input image, which gets transformed to the output image, to the input image
            error = eval_model.evaluate(shards, shards)
            print("Mean Squared Error of %s (Compared to bilinear upscaling) : " % (self.model_name), error[0])
            print("Peak Signal to Noise Ratio of %s (Compared to bilinear upscaling) : " % (self.model_name), error[1])

class ImageSuperResolutionModel(LearningModel):

    def __init__(self):
        super(ImageSuperResolutionModel, self).__init__("Image SR")

        self.f1 = 9
        self.f2 = 1
        self.f3 = 5

    def create_model(self, height=33, width=33, channels=3, load_weights=False):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        init = Input(shape=(channels, height, width))

        x = Convolution2D(self.n1, self.f1, self.f1, activation='relu', border_mode='same', name='level1')(init)
        x = Convolution2D(self.n2, self.f2, self.f2, activation='relu', border_mode='same', name='level2')(x)

        out = Convolution2D(channels, self.f3, self.f3, border_mode='same', name='output')(x)

        model = Model(init, out)
        model.compile(optimizer='adadelta', loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights("SR Weights.h5")

        self.model = model
        return model

    def fit(self, trainX, trainY, weight_fn="SR Weights.h5", batch_size=128, nb_epochs=100) -> Model:
        return super(ImageSuperResolutionModel, self).fit(trainX, trainY, weight_fn, batch_size, nb_epochs)

class ExpantionSuperResolution(LearningModel):

    def __init__(self):
        super(ExpantionSuperResolution, self).__init__("Expanded Image SR")

        self.f1 = 9
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

    def create_model(self, height=33, width=33, channels=3, load_weights=False):
        """
            Creates a model to be used to scale images of specific height and width.
        """

        init = Input(shape=(channels, height, width))

        x = Convolution2D(self.n1, self.f1, self.f1, activation='relu', border_mode='same', name='level1')(init)

        x1 = Convolution2D(self.n2, self.f2_1, self.f2_1, activation='relu', border_mode='same', name='lavel1_1')(x)
        x2 = Convolution2D(self.n2, self.f2_2, self.f2_2, activation='relu', border_mode='same', name='lavel1_2')(x)
        x3 = Convolution2D(self.n2, self.f2_3, self.f2_3, activation='relu', border_mode='same', name='lavel1_3')(x)

        x = merge([x1, x2, x3], mode='ave')

        out = Convolution2D(channels, self.f3, self.f3, activation='relu', border_mode='same', name='output')(x)

        model = Model(init, out)
        model.compile(optimizer='adadelta', loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights("Expantion SR Weights.h5")

        self.model = model
        return model

    def fit(self, trainX, trainY, weight_fn="Expantion SR Weights.h5", batch_size=128, nb_epochs=100) -> Model:
        return super(ExpantionSuperResolution, self).fit(trainX, trainY, weight_fn, batch_size, nb_epochs)

class DenoisingAutoEncoderSR(LearningModel):

    def __init__(self):
        super(DenoisingAutoEncoderSR, self).__init__("Denoise AutoEncoder SR")

    def create_model(self, height=32, width=32, channels=3, load_weights=False):
        """
            Creates a model to remove / reduce noise from upscaled images.
        """
        from keras.layers.convolutional import MaxPooling2D, UpSampling2D
        from keras.layers import merge

        init = Input(shape=(channels, height, width))

        level1_1 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(init)
        x = MaxPooling2D((2,2))(level1_1)
        level2_1 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(x)
        x = MaxPooling2D((2,2))(level2_1)

        level3 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(level3)

        level2_2 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(x)
        level2 = merge([level2_1, level2_2], mode='ave')

        x = UpSampling2D((2, 2))(level2)
        level1_2 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(x)
        level1 = merge([level1_1, level1_2], mode='ave')

        decoded = Convolution2D(channels, 5, 5, activation='linear', border_mode='same')(level1)

        model = Model(init, decoded)
        model.compile(optimizer='adadelta', loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights("Denoising AutoEncoder.h5")

        self.model = model
        return model

    def fit(self, trainX, trainY, weight_fn="Denoising AutoEncoder.h5", batch_size=128, nb_epochs=100):
        return super(DenoisingAutoEncoderSR, self).fit(trainX, trainY, weight_fn, batch_size, nb_epochs)

    def evaluate(self, is_denoise=False):
        """
        Evaluates the model on the Set5 Validation images
        """
        super(DenoisingAutoEncoderSR, self).evaluate(is_denoise=True)
