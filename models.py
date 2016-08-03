from keras.models import Model
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D
from advanced import HistoryCheckpoint
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

    def create_model(self, height=33, width=33, channels=3, load_weights=False, batch_size=128) -> Model:
        """
        Subclass dependent implementation.
        """
        pass

    def fit(self, trainX, trainY, weight_fn, batch_size=128, nb_epochs=100,
                                save_history=True, history_fn="Model History.txt") -> Model:
        """
        Standard method to train any of the models.
        The last 2000 images belong to the Set5 Validation images.
        """
        nb_train = trainX.shape[0] - 2000
        trainX, testX = trainX[:nb_train, :, :, :], trainX[nb_train:, :, :, :]
        trainY, testY = trainY[:nb_train, :, :, :], trainY[nb_train:, :, :, :]

        if self.model == None: self.create_model(batch_size=batch_size)

        callback_list = [callbacks.ModelCheckpoint(weight_fn, monitor='val_PSNRLoss', save_best_only=True,  mode='max', save_weights_only=True),]
        if save_history: callback_list.append(HistoryCheckpoint(history_fn))

        self.model.fit(trainX, trainY, batch_size=batch_size,  nb_epoch=nb_epochs, callbacks=callback_list,
                    validation_data=(testX, testY))

        return self.model

    def evaluate(self, is_denoise=False):
        """
        Evaluates the model on the Set5 Validation images
        """
        if self.model == None: self.create_model(load_weights=True)

        if not is_denoise:
            trainX, trainY = img_utils.loadImages()
        else:
            trainX, trainY = img_utils.loadDenoiseImages()

        error = self.model.evaluate(trainX[-2000:], trainY[-2000:])
        print("Mean Squared Error : ", error[0])
        print("Peak Signal to Noise Ratio : ", error[1])

    def upscale(self, img_path, scale_factor=2, save_intermediate=False, return_image=False, suffix="scaled",
                patch_size=3, patch_stride=1, verbose=True, evaluate=True):
        """
        Standard method to upscale an image.

        :param img_path:  path to the image
        :param scale_factor: scale factor can be any value, usually is an integer
        :param save_intermediate: saves the intermediate upscaled image (bilinear upscale)
        :param return_image: returns a image of shape (height, width, channels).
        :param suffix: suffix of upscaled image
        :param patch_size: size of each patch grid
        :param patch_stride: patch stride is generally 1.
        :param verbose: whether to print messages
        :param evaluate: evaluate the upscaled image on the original image.
        """
        import os
        from scipy.misc import imread, imresize, imsave

        # Destination path
        path = os.path.splitext(img_path)
        filename = path[0] + "_" + suffix + "(%dx)" % (scale_factor) + path[1]

        # Read image
        scale_factor = int(scale_factor)
        true_img = imread(img_path, mode='RGB')
        init_height, init_width = true_img.shape[0], true_img.shape[1]
        if verbose: print("Old Size : ", true_img.shape)
        if verbose: print("New Size : (%d, %d, 3)" % (init_height * scale_factor, init_width * scale_factor))

        denoise_models = ['Deep Denoise SR']

        # Create patches
        if self.model_name in denoise_models:
            print("Deep Denoise requires patch size which is multiple of 4.\n Setting patch_size = 4.")
            patch_size = 4

        patches = img_utils.make_patches(true_img, scale_factor, patch_size, patch_stride, verbose)

        nb_patches = patches.shape[0]
        patch_height, patch_width = patches.shape[1], patches.shape[2]
        print("Number of patches = %d, Patch Shape = (%d, %d)" % (nb_patches, patch_height, patch_width))

        model = None

        # Save intermediate bilinear scaled image is needed for comparison.
        if save_intermediate:
            if verbose: print("Saving intermediate image.")
            fn = path[0] + "_intermediate_" + path[1]
            intermediate_img = imresize(true_img, (init_height * scale_factor, init_width * scale_factor))
            imsave(fn, intermediate_img)

        # Transpose and Process images
        img_conv = patches.transpose((0, 3, 1, 2)).astype('float64') / 255

        if model == None:
            model = self.create_model(patch_height, patch_width, load_weights=True)
            if verbose: print("Model loaded.")

        # Create prediction for image patches
        result = model.predict(img_conv, batch_size=128, verbose=verbose)

         # Deprocess patches
        result = result.transpose((0, 2, 3, 1)).astype('float64') * 255

        # Output shape is (original_height * scale, original_width * scale, nb_channels)
        out_shape = (init_height * scale_factor, init_width * scale_factor, 3)
        result = img_utils.combine_patches(result, out_shape, scale_factor)
        result = np.clip(result, 0, 255).astype('uint8')

        if verbose: print("\nCompleted merging shards")

        if return_image:
            # Return the image without saving. Useful for testing images.
            return result

        if verbose: print("Saving image.")
        imsave(filename, result)

        if evaluate:
            if verbose: print("Evaluating results.")
            # Convert initial image into patches
            eval_img = img_utils.make_patches(true_img, scale_factor, patch_size, patch_stride, False, verbose)
            eval_model = self.create_model(patch_height, patch_width, load_weights=True)

            # Evaluating the initial image patches, which gets transformed to the output image, to the input image
            error = eval_model.evaluate(eval_img, eval_img, batch_size=128)
            print("\nMean Squared Error of %s  : " % (self.model_name), error[0])
            print("Peak Signal to Noise Ratio of %s : " % (self.model_name), error[1])

class ImageSuperResolutionModel(LearningModel):

    def __init__(self):
        super(ImageSuperResolutionModel, self).__init__("Image SR")

        self.f1 = 9
        self.f2 = 1
        self.f3 = 5

    def create_model(self, height=33, width=33, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        init = Input(shape=(channels, height, width))

        x = Convolution2D(self.n1, self.f1, self.f1, activation='relu', border_mode='same', name='level1')(init)
        x = Convolution2D(self.n2, self.f2, self.f2, activation='relu', border_mode='same', name='level2')(x)

        out = Convolution2D(channels, self.f3, self.f3, border_mode='same', name='output')(x)

        model = Model(init, out)
        model.compile(optimizer='adadelta', loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights("weights/SR Weights.h5")

        self.model = model
        return model

    def fit(self, trainX, trainY, weight_fn="weights/SR Weights.h5", batch_size=128, nb_epochs=100,
                                save_history=True, history_fn="SRCNN History.txt") -> Model:
        return super(ImageSuperResolutionModel, self).fit(trainX, trainY, weight_fn, batch_size, nb_epochs, save_history, history_fn)

class ExpantionSuperResolution(LearningModel):

    def __init__(self):
        super(ExpantionSuperResolution, self).__init__("Expanded Image SR")

        self.f1 = 9
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

    def create_model(self, height=33, width=33, channels=3, load_weights=False, batch_size=128):
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
        if load_weights: model.load_weights("weights/Expantion SR Weights.h5")

        self.model = model
        return model

    def fit(self, trainX, trainY, weight_fn="weights/Expantion SR Weights.h5", batch_size=128, nb_epochs=100,
                                        save_history=True, history_fn="ESRCNN History.txt") -> Model:
        return super(ExpantionSuperResolution, self).fit(trainX, trainY, weight_fn, batch_size, nb_epochs, save_history, history_fn)

class DenoisingAutoEncoderSR(LearningModel):

    def __init__(self):
        super(DenoisingAutoEncoderSR, self).__init__("Denoise AutoEncoder SR")

    def create_model(self, height=33, width=33, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to remove / reduce noise from upscaled images.
        """
        from keras.layers.convolutional import Deconvolution2D
        from keras.layers import merge

        init = Input(shape=(channels, height, width))

        level1_1 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(init)
        level2_1 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(level1_1)

        level2_2 = Deconvolution2D(self.n1, 3, 3, activation='relu', output_shape=(None, channels, height, width), border_mode='same')(level2_1)
        level2 = merge([level2_1, level2_2], mode='sum')

        level1_2 = Deconvolution2D(self.n1, 3, 3, activation='relu', output_shape=(None, channels, height, width), border_mode='same')(level2)
        level1 = merge([level1_1, level1_2], mode='sum')

        decoded = Convolution2D(channels, 5, 5, activation='linear', border_mode='same')(level1)

        model = Model(init, decoded)
        model.compile(optimizer='adam', loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights("weights/Denoising AutoEncoder.h5")

        self.model = model
        return model

    def fit(self, trainX, trainY, weight_fn="weights/Denoising AutoEncoder.h5", batch_size=128, nb_epochs=100,
                            save_history=True, history_fn="DSRCNN History.txt"):
        return super(DenoisingAutoEncoderSR, self).fit(trainX, trainY, weight_fn, batch_size, nb_epochs, save_history, history_fn)

    def evaluate(self, is_denoise=True):
        """
        Evaluates the model on the Set5 Validation images
        """
        super(DenoisingAutoEncoderSR, self).evaluate(is_denoise)


class DeepDenoiseSR(LearningModel):

    def __init__(self):
        super(DeepDenoiseSR, self).__init__("Deep Denoise SR")

        self.n1 = 64
        self.n2 = 128
        self.n3 = 256

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        from keras.layers.convolutional import MaxPooling2D, UpSampling2D
        from keras.layers import merge

        init = Input(shape=(channels, height, width))

        c1 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(init)
        c1 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(c1)

        x = MaxPooling2D((2, 2))(c1)

        c2 = Convolution2D(self.n2, 3, 3, activation='relu', border_mode='same')(x)
        c2 = Convolution2D(self.n2, 3, 3, activation='relu', border_mode='same')(c2)

        x = MaxPooling2D((2, 2))(c2)

        c3 = Convolution2D(self.n3, 3, 3, activation='relu', border_mode='same')(x)

        x = UpSampling2D()(c3)

        c2_2 = Convolution2D(self.n2, 3, 3, activation='relu', border_mode='same')(x)
        c2_2 = Convolution2D(self.n2, 3, 3, activation='relu', border_mode='same')(c2_2)

        m1 = merge([c2, c2_2], mode='sum')
        m1 = UpSampling2D()(m1)

        c1_2 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(m1)
        c1_2 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(c1_2)

        m2 = merge([c1, c1_2], mode='sum')

        decoded = Convolution2D(channels, 5, 5, activation='linear', border_mode='same')(m2)

        model = Model(init, decoded)
        model.compile(optimizer='adam', loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights("weights/Deep Denoise Weights.h5")

        self.model = model
        return model

    def fit(self, trainX, trainY, weight_fn="weights/Deep Denoise Weights.h5", batch_size=128, nb_epochs=100,
                         save_history=True, history_fn="Deep DSRCNN History.txt"):
        super(DeepDenoiseSR, self).fit(trainX, trainY, weight_fn, batch_size, nb_epochs, save_history, history_fn)

    def evaluate(self, is_denoise=True):
        super(DeepDenoiseSR, self).evaluate(is_denoise)
