from keras.models import Model
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D
import keras.callbacks as callbacks
import numpy as np
import keras.backend as K
import img_utils

path_X = r"output_images_X\\"
path_Y = r"output_images_Y\\"

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

    def upscale(self, img_path, scale_factor=2, save_intermediate=False, return_image=False, suffix="scaled", verbose=True, evaluate=True):
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

        # Read image
        true_img = imread(img_path, mode='RGB')
        true_height, true_width = true_img.shape[0], true_img.shape[1]
        if verbose: print("Old Size : ", true_img.shape)

        if self.model_name == "Denoise AutoEncoder SR":
            if (true_height % 8 != 0) or (true_width % 8 != 0):
                print("Image size needs to be divisible by 8 to use denoise auto encoder.")
                print("Resizing")
                true_height = (true_height // 8) * 8
                true_width = (true_width // 8) * 8
                true_img = imresize(true_img, (true_height, true_width))

                print("Image has been modified to size (%d, %d)" % (true_height, true_width))

        # Pre Upscale
        img = imresize(true_img, (true_height * scale_factor, true_width * scale_factor))
        height, width = img.shape[0], img.shape[1]
        if verbose: print("New Size : ", img.shape)

        # Transpose and Process image
        img_conv = img.transpose((2, 0, 1)).astype('float64') / 255
        img_conv = np.expand_dims(img_conv, axis=0)

        model = self.create_model(height, width, load_weights=True)
        if verbose: print("Model loaded.")

        # Create prediction for image
        result = model.predict(img_conv)
        if verbose: print("Model finished upscaling.")

        if evaluate:
            # Evaluate agains bilinear upscaled image. Just to measure if a good result was obtained.
            if verbose: print("Evaluating results.")
            # Pre Downscale if evaluate = True
            true_img = imresize(true_img, (true_height // scale_factor, true_width // scale_factor))
            true_img = imresize(true_img, (true_height, true_width))
            true_img = true_img.transpose((2, 0, 1)).astype('float64') / 255
            true_img = np.expand_dims(true_img, axis=0)

            eval_model = self.create_model(true_height, true_width, load_weights=True)
            eval_result = eval_model.predict(true_img)
            error = eval_model.evaluate(eval_result, true_img)
            print("Mean Squared Error of %s : " % (self.model_name), error[0])
            print("Peak Signal to Noise Ratio of %s : " % (self.model_name), error[1])

        # Deprocess
        result = result.reshape((3, height, width))
        result = result.transpose((1, 2, 0)).astype('float64') * 255
        result = np.clip(result, 0, 255).astype('uint8')

        if return_image:
            # Return the image without saving. Usefull for testing images.
            return result

        path = os.path.splitext(img_path)
        filename = path[0] + "_" + suffix + "(%dx)" % (scale_factor) + path[1]

        if verbose: print("Saving image.")
        # Save intermediate bilinear scaled image is needed for comparison.
        if save_intermediate:
            fn = path[0] + "_intermediate" + path[1]
            imsave(fn, img)

        imsave(filename, result)

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
