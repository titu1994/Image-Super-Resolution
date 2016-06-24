from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D
import numpy as np
import keras.backend as K
import img_utils

path_X = r"output_images_X\\"
path_Y = r"output_images_Y\\"

epochs = 100
batchSize = 128

fsub = low_resolution_size = 33
high_resolution_size = 400

f1, f2, f3 = 9, 5, 5
n1, n2 = 64, 32

c = 3 # Number of channels in input image

def createModel(height, width) -> Model:
    """
    Creates a model to be used to scale images of specific height and width.
    """
    init = Input(shape=(c, height, width))

    x = Convolution2D(n1, f1, f1, activation='relu', border_mode='same', name='level1')(init)
    x = Convolution2D(n2, f2, f2, activation='relu', border_mode='same', name='level2')(x)

    out = Convolution2D(c, f3, f3, border_mode='same', name='output')(x)

    model = Model(init, out)
    model.compile(optimizer='adadelta', loss='mse', metrics=[PSNRLoss])
    model.load_weights("SR Weights.h5")

    return model

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(255) - 10 * log10(MSE)

    Here, 20 * log10(255)== 48.1308036087. Precomputed to improve performance.

    Note, PSNR values are very high, much higher than the ones in the paper. This is due to use of all 3 channels of
    RGB instead of only thr Y channel of YCbCr. It averages around 80~ after 100 epochs.
    """
    return 48.1308036087 - 10. * np.log10(K.mean(K.square(y_pred - y_true)))

if __name__ == "__main__":
    init = Input(shape=(c, low_resolution_size, low_resolution_size))

    x = Convolution2D(n1, f1, f1, activation='relu', border_mode='same', name='level1')(init)
    x = Convolution2D(n2, f2, f2, activation='relu', border_mode='same', name='level2')(x)

    out = Convolution2D(c, f3, f3, border_mode='same', name='output')(x)

    model = Model(init, out)
    model.summary()

    model.compile(optimizer='adadelta', loss='mse', metrics=[PSNRLoss])

    trainX, trainY = img_utils.loadImages()

    model.load_weights("SR Weights.h5")
    print('Model Loaded')

    # Training is not needed since weights are provided.
    # Validates on the last 5 x 400 images, which are the sub-images of the 5 validation images in the paper.

    #model.fit(trainX, trainY, batch_size=batchSize, nb_epoch=epochs,
    #      callbacks=[callbacks.ModelCheckpoint("SR Weights.h5", monitor='val_PSNRLoss', save_best_only=True, mode='max')],
    #      validation_split=2000./38400)

    error = model.evaluate(trainX[-2000:], trainY[-2000:])
    print("Mean Squared Error (Compared to bilinear upscaled version) : ", error[0])
    print("Peak Signal to Noise Ratio (Compared to bilinear upscaled version) : ", error[1])


