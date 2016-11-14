from __future__ import print_function, division

from keras.models import Model
from keras.layers import Input, merge, BatchNormalization, Activation, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras import backend as K
import keras.callbacks as callbacks
import keras.optimizers as optimizers

from advanced import HistoryCheckpoint, SubPixelUpscaling
import img_utils

import numpy as np
import os
import time

train_path = img_utils.output_path
validation_path = img_utils.validation_output_path
path_X = img_utils.output_path + "X/"
path_Y = img_utils.output_path + "y/"

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

def psnr(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                         " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                   str(y_pred.shape))

    return -10. * np.log10(np.mean(np.square(y_pred - y_true)))


class BaseSuperResolutionModel(object):

    def __init__(self, model_name, scale_factor):
        """
        Base model to provide a standard interface of adding Super Resolution models
        """
        self.model = None # type: Model
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.weight_path = None

        self.type_requires_divisible_shape = False
        self.type_true_upscaling = False

        self.evaluation_func = None
        self.uses_learning_phase = False

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128) -> Model:
        """
        Subclass dependent implementation.
        """
        if self.type_requires_divisible_shape:
            assert height * img_utils._image_scale_multiplier % 4 == 0, "Height of the image must be divisible by 4"
            assert width * img_utils._image_scale_multiplier % 4 == 0, "Width of the image must be divisible by 4"

        if K.image_dim_ordering() == "th":
            shape = (channels, width * img_utils._image_scale_multiplier, height * img_utils._image_scale_multiplier)
        else:
            shape = (width * img_utils._image_scale_multiplier, height * img_utils._image_scale_multiplier, channels)

        init = Input(shape=shape)

        return init

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="Model History.txt") -> Model:
        """
        Standard method to train any of the models.
        """

        samples_per_epoch = img_utils.image_count()
        val_count = img_utils.val_image_count()
        if self.model == None: self.create_model(batch_size=batch_size)

        callback_list = [callbacks.ModelCheckpoint(self.weight_path, monitor='val_PSNRLoss', save_best_only=True,
                                                   mode='max', save_weights_only=True)]
        if save_history: callback_list.append(HistoryCheckpoint(history_fn))

        print("Training model : %s" % (self.__class__.__name__))
        self.model.fit_generator(img_utils.image_generator(train_path, scale_factor=self.scale_factor,
                                                           small_train_images=self.type_true_upscaling,
                                                           batch_size=batch_size),
                                 samples_per_epoch=samples_per_epoch,
                                 nb_epoch=nb_epochs, callbacks=callback_list,
                                 validation_data=img_utils.image_generator(validation_path,
                                                                           scale_factor=self.scale_factor,
                                                                           small_train_images=self.type_true_upscaling,
                                                                           batch_size=batch_size),
                                 nb_val_samples=val_count)

        return self.model

    def evaluate(self, validation_dir):
        if self.type_requires_divisible_shape:
            _evaluate_denoise(self, validation_dir)
        else:
            _evaluate(self, validation_dir)

    
    def upscale(self, img_path, save_intermediate=False, return_image=False, suffix="scaled",
                patch_size=8, mode="patch", verbose=True):
        """
        Standard method to upscale an image.

        :param img_path:  path to the image
        :param save_intermediate: saves the intermediate upscaled image (bilinear upscale)
        :param return_image: returns a image of shape (height, width, channels).
        :param suffix: suffix of upscaled image
        :param patch_size: size of each patch grid
        :param verbose: whether to print messages
        :param mode: mode of upscaling. Can be "patch" or "fast"
        """
        import os
        from scipy.misc import imread, imresize, imsave

        # Destination path
        path = os.path.splitext(img_path)
        filename = path[0] + "_" + suffix + "(%dx)" % (self.scale_factor) + path[1]

        # Read image
        scale_factor = int(self.scale_factor)
        true_img = imread(img_path, mode='RGB')
        init_width, init_height = true_img.shape[0], true_img.shape[1]
        if verbose: print("Old Size : ", true_img.shape)
        if verbose: print("New Size : (%d, %d, 3)" % (init_height * scale_factor, init_width * scale_factor))

        img_height, img_width = 0, 0

        if mode == "patch" and self.type_true_upscaling:
            # Overriding mode for True Upscaling models
            mode = 'fast'
            print("Patch mode does not work with True Upscaling models yet. Defaulting to mode='fast'")

        if mode == 'patch':
            # Create patches
            if self.type_requires_divisible_shape:
                if patch_size % 4 != 0:
                    print("Deep Denoise requires patch size which is multiple of 4.\nSetting patch_size = 8.")
                    patch_size = 8

            images = img_utils.make_patches(true_img, scale_factor, patch_size, verbose)

            nb_images = images.shape[0]
            img_width, img_height = images.shape[1], images.shape[2]
            print("Number of patches = %d, Patch Shape = (%d, %d)" % (nb_images, img_height, img_width))
        else:
            # Use full image for super resolution
            img_width, img_height = self.__match_autoencoder_size(img_height, img_width, init_height,
                                                                  init_width, scale_factor)

            images = imresize(true_img, (img_width, img_height))
            images = np.expand_dims(images, axis=0)
            print("Image is reshaped to : (%d, %d, %d)" % (images.shape[1], images.shape[2], images.shape[3]))

        # Save intermediate bilinear scaled image is needed for comparison.
        intermediate_img = None
        if save_intermediate:
            if verbose: print("Saving intermediate image.")
            fn = path[0] + "_intermediate_" + path[1]
            intermediate_img = imresize(true_img, (init_width * scale_factor, init_height * scale_factor))
            imsave(fn, intermediate_img)

        # Transpose and Process images
        if K.image_dim_ordering() == "th":
            img_conv = images.transpose((0, 3, 1, 2)).astype(np.float32) / 255.
        else:
            img_conv = images.astype(np.float32) / 255.

        model = self.create_model(img_height, img_width, load_weights=True)
        if verbose: print("Model loaded.")

        # Create prediction for image patches
        result = model.predict(img_conv, batch_size=128, verbose=verbose)

        if verbose: print("De-processing images.")

         # Deprocess patches
        if K.image_dim_ordering() == "th":
            result = result.transpose((0, 2, 3, 1)).astype(np.float32) * 255.
        else:
            result = result.astype(np.float32) * 255.

        # Output shape is (original_width * scale, original_height * scale, nb_channels)
        if mode == 'patch':
            out_shape = (init_width * scale_factor, init_height * scale_factor, 3)
            result = img_utils.combine_patches(result, out_shape, scale_factor)
        else:
            result = result[0, :, :, :] # Access the 3 Dimensional image vector

        result = np.clip(result, 0, 255).astype('uint8')

        if verbose: print("\nCompleted De-processing image.")

        if return_image:
            # Return the image without saving. Useful for testing images.
            return result

        if verbose: print("Saving image.")
        imsave(filename, result)

    def __match_autoencoder_size(self, img_height, img_width, init_height, init_width, scale_factor):
        if self.type_requires_divisible_shape:
            if not self.type_true_upscaling:
                # AE model but not true upsampling
                if ((init_height * scale_factor) % 4 != 0) or ((init_width * scale_factor) % 4 != 0) or \
                        (init_height % 2 != 0) or (init_width % 2 != 0):

                    print("AE models requires image size which is multiple of 4.")
                    img_height = ((init_height * scale_factor) // 4) * 4
                    img_width = ((init_width * scale_factor) // 4) * 4

                else:
                    # No change required
                    img_height, img_width = init_height * scale_factor, init_width * scale_factor
            else:
                # AE model and true upsampling
                if ((init_height) % 4 != 0) or ((init_width) % 4 != 0) or \
                        (init_height % 2 != 0) or (init_width % 2 != 0):

                    print("AE models requires image size which is multiple of 4.")
                    img_height = ((init_height) // 4) * 4
                    img_width = ((init_width) // 4) * 4

                else:
                    # No change required
                    img_height, img_width = init_height, init_width
        else:
            # Not AE but true upsampling
            if self.type_true_upscaling:
                img_height, img_width = init_height, init_width
            else:
                # Not AE and not true upsampling
                img_height, img_width = init_height * scale_factor, init_width * scale_factor

        return img_height, img_width


def _evaluate(sr_model : BaseSuperResolutionModel, validation_dir):
    """
        Evaluates the model on the Validation images
        """
    print("Validating %s model" % sr_model.model_name)
    if sr_model.model == None: sr_model.create_model(load_weights=True)
    if sr_model.evaluation_func is None:
        if sr_model.uses_learning_phase:
            sr_model.evaluation_func = K.function([sr_model.model.layers[0].input, K.learning_phase()],
                                                  [sr_model.model.layers[-1].output])
        else:
            sr_model.evaluation_func = K.function([sr_model.model.layers[0].input],
                                              [sr_model.model.layers[-1].output])
    predict_path = "val_predict/"
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
    validation_path_set5 = validation_dir + "set5/"
    validation_path_set14 = validation_dir + "set14/"
    validation_dirs = [validation_path_set5, validation_path_set14]
    for val_dir in validation_dirs:
        image_fns = [name for name in os.listdir(val_dir)]
        nb_images = len(image_fns)
        print("Validating %d images from path %s" % (nb_images, val_dir))

        total_psnr = 0.0

        for impath in os.listdir(val_dir):
            t1 = time.time()

            # Input image
            y = img_utils.imread(val_dir + impath, mode='RGB')
            width, height, _ = y.shape

            if sr_model.type_requires_divisible_shape:
                # Denoise models require precise width and height, divisible by 4

                if ((width // sr_model.scale_factor) % 4 != 0) or ((height // sr_model.scale_factor) % 4 != 0) \
                        or (width % 2 != 0) or (height % 2 != 0):
                    width = ((width // sr_model.scale_factor) // 4) * 4 * sr_model.scale_factor
                    height = ((height // sr_model.scale_factor) // 4) * 4 * sr_model.scale_factor

                    print("Model %s require the image size to be divisible by 4. New image size = (%d, %d)" % \
                          (sr_model.model_name, width, height))

                    y = img_utils.imresize(y, (width, height), interp='bicubic')

            y = y.astype('float32') / 255.
            y = np.expand_dims(y, axis=0)

            x_width = width if not sr_model.type_true_upscaling else width // sr_model.scale_factor
            x_height = height if not sr_model.type_true_upscaling else height // sr_model.scale_factor

            x_temp = y.copy()
            img = img_utils.imresize(x_temp[0], (x_width // sr_model.scale_factor, x_height // sr_model.scale_factor),
                                     interp='bicubic')

            if not sr_model.type_true_upscaling:
                img = img_utils.imresize(img, (x_width, x_height), interp='bicubic')

            x = np.expand_dims(img, axis=0)

            if K.image_dim_ordering() == "th":
                x = x.transpose((0, 3, 1, 2))
                y = y.transpose((0, 3, 1, 2))

            if sr_model.uses_learning_phase:
                y_pred = sr_model.evaluation_func([x, 0])[0][0]
            else:
                y_pred = sr_model.evaluation_func([x])[0][0]

            psnr_val = psnr(y[0], np.clip(y_pred, 0, 255) / 255)
            total_psnr += psnr_val

            t2 = time.time()
            print("Validated image : %s, Time required : %0.2f, PSNR value : %0.4f" % (impath, t2 - t1, psnr_val))

            generated_path = predict_path + "%s_%s_generated.png" % (sr_model.model_name, os.path.splitext(impath)[0])

            if K.image_dim_ordering() == "th":
                y_pred = y_pred.transpose((1, 2, 0))

            y_pred = np.clip(y_pred, 0, 255).astype('uint8')
            img_utils.imsave(generated_path, y_pred)

        print("Average PRNS value of validation images = %00.4f \n" % (total_psnr / nb_images))


def _evaluate_denoise(sr_model : BaseSuperResolutionModel, validation_dir):
    print("Validating %s model" % sr_model.model_name)
    predict_path = "val_predict/"
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)

    validation_path_set5 = validation_dir + "set5/"
    validation_path_set14 = validation_dir + "set14/"

    validation_dirs = [validation_path_set5, validation_path_set14]
    for val_dir in validation_dirs:
        image_fns = [name for name in os.listdir(val_dir)]
        nb_images = len(image_fns)
        print("Validating %d images from path %s" % (nb_images, val_dir))

        total_psnr = 0.0

        for impath in os.listdir(val_dir):
            t1 = time.time()

            # Input image
            y = img_utils.imread(val_dir + impath, mode='RGB')
            width, height, _ = y.shape

            if ((width // sr_model.scale_factor) % 4 != 0) or ((height // sr_model.scale_factor) % 4 != 0) \
                    or (width % 2 != 0) or (height % 2 != 0):
                width = ((width // sr_model.scale_factor) // 4) * 4 * sr_model.scale_factor
                height = ((height // sr_model.scale_factor) // 4) * 4 * sr_model.scale_factor

                print("Model %s require the image size to be divisible by 4. New image size = (%d, %d)" % \
                      (sr_model.model_name, width, height))

                y = img_utils.imresize(y, (width, height), interp='bicubic')

            y = y.astype('float32') / 255.
            y = np.expand_dims(y, axis=0)

            x_temp = y.copy()
            img = img_utils.imresize(x_temp[0], (width // sr_model.scale_factor, height // sr_model.scale_factor),
                                     interp='bicubic')

            if not sr_model.type_true_upscaling:
                img = img_utils.imresize(img, (width, height), interp='bicubic')

            x = np.expand_dims(img, axis=0)

            if K.image_dim_ordering() == "th":
                x = x.transpose((0, 3, 1, 2))
                y = y.transpose((0, 3, 1, 2))

            sr_model.model = sr_model.create_model(height, width, load_weights=True)

            if sr_model.evaluation_func is None:
                if sr_model.uses_learning_phase:
                    sr_model.evaluation_func = K.function([sr_model.model.layers[0].input, K.learning_phase()],
                                                          [sr_model.model.layers[-1].output])
                else:
                    sr_model.evaluation_func = K.function([sr_model.model.layers[0].input],
                                                      [sr_model.model.layers[-1].output])

            if sr_model.uses_learning_phase:
                y_pred = sr_model.evaluation_func([x, 0])[0][0]
            else:
                y_pred = sr_model.evaluation_func([x])[0][0]

            psnr_val = psnr(y[0], np.clip(y_pred, 0, 255) / 255)
            total_psnr += psnr_val

            t2 = time.time()
            print("Validated image : %s, Time required : %0.2f, PSNR value : %0.4f" % (impath, t2 - t1, psnr_val))

            generated_path = predict_path + "%s_%s_generated.png" % (sr_model.model_name, os.path.splitext(impath)[0])

            if K.image_dim_ordering() == "th":
                y_pred = y_pred.transpose((1, 2, 0))

            y_pred = np.clip(y_pred, 0, 255).astype('uint8')
            img_utils.imsave(generated_path, y_pred)

        print("Average PRNS value of validation images = %00.4f \n" % (total_psnr / nb_images))



class ImageSuperResolutionModel(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ImageSuperResolutionModel, self).__init__("Image SR", scale_factor)

        self.f1 = 9
        self.f2 = 1
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32

        self.weight_path = "weights/SR Weights %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        init = super(ImageSuperResolutionModel, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(self.n1, self.f1, self.f1, activation='relu', border_mode='same', name='level1')(init)
        x = Convolution2D(self.n2, self.f2, self.f2, activation='relu', border_mode='same', name='level2')(x)

        out = Convolution2D(channels, self.f3, self.f3, border_mode='same', name='output')(x)

        model = Model(init, out)

        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)

        self.model = model
        return model

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="SRCNN History.txt"):
        return super(ImageSuperResolutionModel, self).fit(batch_size, nb_epochs, save_history, history_fn)


class ExpantionSuperResolution(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ExpantionSuperResolution, self).__init__("Expanded Image SR", scale_factor)

        self.f1 = 9
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32

        self.weight_path = "weights/Expantion SR Weights %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        init = super(ExpantionSuperResolution, self).create_model(height, width, channels, load_weights, batch_size)

        x = Convolution2D(self.n1, self.f1, self.f1, activation='relu', border_mode='same', name='level1')(init)

        x1 = Convolution2D(self.n2, self.f2_1, self.f2_1, activation='relu', border_mode='same', name='lavel1_1')(x)
        x2 = Convolution2D(self.n2, self.f2_2, self.f2_2, activation='relu', border_mode='same', name='lavel1_2')(x)
        x3 = Convolution2D(self.n2, self.f2_3, self.f2_3, activation='relu', border_mode='same', name='lavel1_3')(x)

        x = merge([x1, x2, x3], mode='ave')

        out = Convolution2D(channels, self.f3, self.f3, activation='relu', border_mode='same', name='output')(x)

        model = Model(init, out)
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)

        self.model = model
        return model

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="ESRCNN History.txt"):
        return super(ExpantionSuperResolution, self).fit(batch_size, nb_epochs, save_history, history_fn)


class DenoisingAutoEncoderSR(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(DenoisingAutoEncoderSR, self).__init__("Denoise AutoEncoder SR", scale_factor)

        self.n1 = 64
        self.n2 = 32

        self.weight_path = "weights/Denoising AutoEncoder %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to remove / reduce noise from upscaled images.
        """
        from keras.layers.convolutional import Deconvolution2D

        # Perform check that model input shape is divisible by 4
        init = super(DenoisingAutoEncoderSR, self).create_model(height, width, channels, load_weights, batch_size)

        if K.image_dim_ordering() == "th":
            output_shape = (None, channels, width, height)
        else:
            output_shape = (None, width, height, channels)

        level1_1 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(init)
        level2_1 = Convolution2D(self.n1, 3, 3, activation='relu', border_mode='same')(level1_1)

        level2_2 = Deconvolution2D(self.n1, 3, 3, activation='relu', output_shape=output_shape, border_mode='same')(level2_1)
        level2 = merge([level2_1, level2_2], mode='sum')

        level1_2 = Deconvolution2D(self.n1, 3, 3, activation='relu', output_shape=output_shape, border_mode='same')(level2)
        level1 = merge([level1_1, level1_2], mode='sum')

        decoded = Convolution2D(channels, 5, 5, activation='linear', border_mode='same')(level1)

        model = Model(init, decoded)
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)

        self.model = model
        return model

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="DSRCNN History.txt"):
        return super(DenoisingAutoEncoderSR, self).fit(batch_size, nb_epochs, save_history, history_fn)

class DeepDenoiseSR(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(DeepDenoiseSR, self).__init__("Deep Denoise SR", scale_factor)

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        self.type_requires_divisible_shape = True

        self.n1 = 64
        self.n2 = 128
        self.n3 = 256

        self.weight_path = "weights/Deep Denoise Weights %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        # Perform check that model input shape is divisible by 4
        init = super(DeepDenoiseSR, self).create_model(height, width, channels, load_weights, batch_size)

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
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)

        self.model = model
        return model

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="Deep DSRCNN History.txt"):
        super(DeepDenoiseSR, self).fit(batch_size, nb_epochs, save_history, history_fn)

class ResNetSR(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ResNetSR, self).__init__("ResNetSR", scale_factor)

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        self.type_requires_divisible_shape = True
        self.uses_learning_phase = True

        self.n = 64
        self.mode = 0

        self.weight_path = "weights/ResNetSR %dX.h5" % (self.scale_factor)

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        init =  super(ResNetSR, self).create_model(height, width, channels, load_weights, batch_size)

        x0 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='sr_res_conv1')(init)

        x1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', subsample=(2, 2), name='sr_res_conv2')(x0)

        x2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', subsample=(2, 2), name='sr_res_conv3')(x1)

        x = self._residual_block(x2, 1)

        nb_residual = 14
        for i in range(nb_residual):
            x = self._residual_block(x, i + 2)

        x = self._upscale_block(x, 1)
        x = merge([x, x1], mode='sum')

        x = self._upscale_block(x, 2)
        x = merge([x, x0], mode='sum')

        x = Convolution2D(3, 3, 3, activation="linear", border_mode='same', name='sr_res_conv_final')(x)

        model = Model(init, x)

        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)

        self.model = model
        return model

    def _residual_block(self, ip, id):
        init = ip

        x = Convolution2D(64, 3, 3, activation='linear', border_mode='same',
                          name='sr_res_conv_' + str(id) + '_1')(ip)
        x = BatchNormalization(axis=1, mode=self.mode, name="sr_res_batchnorm_" + str(id) + "_1")(x)
        x = Activation('relu', name="sr_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(64, 3, 3, activation='linear', border_mode='same',
                          name='sr_res_conv_' + str(id) + '_2')(x)
        x = BatchNormalization(axis=1, mode=self.mode, name="sr_res_batchnorm_" + str(id) + "_2")(x)

        m = merge([x, init], mode='sum', name="sr_res_merge_" + str(id))

        return m

    def _upscale_block(self, ip, id):
        init = ip

        x = Convolution2D(256, 3, 3, activation="relu", border_mode='same', name='sr_res_upconv1_%d' % id)(init)
        x = SubPixelUpscaling(r=2, channels=self.n, name='sr_res_upscale1_%d' % id)(x)
        x = Convolution2D(self.n, 3, 3, activation="relu", border_mode='same', name='sr_res_filter1_%d' % id)(x)

        return x

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="ResNetSR History.txt"):
        super(ResNetSR, self).fit(batch_size, nb_epochs, save_history, history_fn)


class EfficientSubPixelConvolutionalSR(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(EfficientSubPixelConvolutionalSR, self).__init__("ESPCNN SR", scale_factor)

        self.n1 = 64
        self.n2 = 32

        self.f1 = 5
        self.f2 = 3
        self.f3 = 3

        self.weight_path = "weights/ESPCNN Weights %d.h5" % scale_factor

        # Flag to denote that this is a "true" upsampling model.
        # Image size will be multiplied by scale factor to get output image size
        self.true_upsampling = True

    def create_model(self, height=16, width=16, channels=3, load_weights=False, batch_size=128):
        # Note height, width = 16 instead of 32 like usual
        init = super(EfficientSubPixelConvolutionalSR, self).create_model(height, width, channels,
                                                                          load_weights, batch_size)

        x = Convolution2D(self.n1, self.f1, self.f1, activation='relu', border_mode='same', name='level1')(init)
        x = Convolution2D(self.n2, self.f2, self.f2, activation='relu', border_mode='same', name='level2')(x)

        x = self._upscale_block(x, 1)

        out = Convolution2D(3, 5, 5, activation='linear', border_mode='same')(x)

        model = Model(init, out)

        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)

        self.model = model
        return model

    def _upscale_block(self, ip, id):
        init = ip

        x = Convolution2D(256, 3, 3, activation="relu", border_mode='same', name='espcnn_upconv1_%d' % id)(init)
        x = SubPixelUpscaling(r=2, channels=self.n1, name='espcnn_upconv1__upscale1_%d' % id)(x)
        x = Convolution2D(256, 3, 3, activation="relu", border_mode='same', name='espcnn_upconv1_filter1_%d' % id)(x)

        return x

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="ESPCNN History.txt"):
        super(EfficientSubPixelConvolutionalSR, self).fit(batch_size, nb_epochs, save_history, history_fn)