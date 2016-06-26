import main
from keras.utils.visualize_util import plot
import models
import img_utils

if __name__ == "__main__":
    path = ""

    """
    Plot the models
    """

    #model = models.ImageSuperResolutionModel().create_model()
    #plot(model, to_file="SRCNN.png", show_shapes=True, show_layer_names=True)
    #model = models.ExpantionSuperResolution().create_model()
    #plot(model, to_file="ESRCNN.png", show_layer_names=True, show_shapes=True)
    model = models.DenoisingAutoEncoderSR().create_model()
    plot(model, to_file="Denoise.png", show_layer_names=True, show_shapes=True)

    """
    Train ExpantionSuperResolution
    """
    #trainX, trainY = img_utils.loadImages()

    #esr = models.ExpantionSuperResolution()
    #esr.create_model(load_weights=True)
    #esr.fit(trainX, trainY, nb_epochs=100)

    """
    Train DrnoisingAutoEncoderSR
    """
    # NOTE: Denoising auto encoder requires even integer height x width. Thus the training images are downscaled.
    #trainX, trainY = img_utils.loadDenoisingImages()
    #denoise = models.DenoisingAutoEncoderSR()
    #denoise.create_model(load_weights=False).summary()

    #denoise.fit(trainX, trainY)

    """
    Compare output images of sr, esr and denoise models
    """
    #sr = models.ImageSuperResolutionModel()
    #sr.upscale(path, scale_factor=2, save_intermediate=True, suffix="sr")

    #esr = models.ExpantionSuperResolution()
    #esr.upscale(path, scale_factor=2, save_intermediate=False, suffix="esr")

    #dsr = models.DenoisingAutoEncoderSR()
    #dsr.upscale(path, scale_factor=2, save_intermediate=False, suffix="denoise")



