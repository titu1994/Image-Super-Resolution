import main
from keras.utils.visualize_util import plot
import models
import img_utils

if __name__ == "__main__":
    path = r"D:\Yue\Desktop\Neural Art\Output\River Large.png"

    """
    Plot the models
    """

    #model = models.ImageSuperResolutionModel().create_model()
    #plot(model, to_file="SRCNN.png", show_shapes=True, show_layer_names=True)
    #model = models.ExpantionSuperResolution().create_model()
    #plot(model, to_file="ESRCNN.png", show_layer_names=True, show_shapes=True)
    #model = models.DenoisingAutoEncoderSR().create_model()
    #plot(model, to_file="Denoise.png", show_layer_names=True, show_shapes=True)

    """
    Train Super Resolution
    """
    #trainX, trainY = img_utils.loadImages()

    #sr = models.ImageSuperResolutionModel()
    #sr.create_model()
    #sr.fit(trainX, trainY, nb_epochs=100)

    """
    Train ExpantionSuperResolution
    """
    #trainX, trainY = img_utils.loadImages()

    #esr = models.ExpantionSuperResolution()
    #esr.create_model()
    #esr.fit(trainX, trainY, nb_epochs=100)

    """
    Train DrnoisingAutoEncoderSR
    """
    # NOTE: Denoising auto encoder requires even integer height x width. Thus the training images are downscaled.
    #trainX, trainY = img_utils.loadDenoisingImages()

    #dsr = models.DenoisingAutoEncoderSR()
    #dsr.create_model(load_weights=False).summary()

    #dsr.fit(trainX, trainY)

    """
    Compare output images of sr, esr and denoise models
    """
    scale = 2

    sr = models.ImageSuperResolutionModel()
    sr.upscale(path, scale_factor=scale, save_intermediate=False, suffix="sr")

    esr = models.ExpantionSuperResolution()
    esr.upscale(path, scale_factor=scale, save_intermediate=False, suffix="esr")

    dsr = models.DenoisingAutoEncoderSR()
    dsr.upscale(path, scale_factor=scale, save_intermediate=False, suffix="dsr")



