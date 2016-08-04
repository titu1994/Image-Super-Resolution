from keras.utils.visualize_util import plot
import models
import img_utils

if __name__ == "__main__":
    path = r""

    """
    Plot the models
    """

    #model = models.ImageSuperResolutionModel().create_model()
    #plot(model, to_file="architectures/SRCNN.png", show_shapes=True, show_layer_names=True)
    #model = models.ExpantionSuperResolution().create_model()
    #plot(model, to_file="architectures/ESRCNN.png", show_layer_names=True, show_shapes=True)
    #model = models.DenoisingAutoEncoderSR().create_model()
    #plot(model, to_file="architectures/Denoise.png", show_layer_names=True, show_shapes=True)
    #model = models.DeepDenoiseSR().create_model()
    #plot(model, to_file="architectures/Deep Denoise.png", show_layer_names=True, show_shapes=True)


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
    Train DenoisingAutoEncoderSR
    """
    #trainX, trainY = img_utils.loadImages()

    #dsr = models.DenoisingAutoEncoderSR()
    #dsr.create_model(load_weights=True).summary()

    #dsr.fit(trainX, trainY)

    """
    Train Deep Denoise SR
    """
    #trainX, trainY = img_utils.loadDenoiseImages()

    #ddsr = models.DeepDenoiseSR()
    #ddsr.create_model(load_weights=False).summary()

    #ddsr.fit(trainX, trainY, nb_epochs=60, batch_size=128)

    """
    Compare output images of sr, esr, dsr and ddsr models
    """
    #scale = 2

    #sr = models.ImageSuperResolutionModel()
    #sr.upscale(path, scale_factor=scale, save_intermediate=False, suffix="sr")

    #esr = models.ExpantionSuperResolution()
    #esr.upscale(path, scale_factor=scale, save_intermediate=False, suffix="esr")

    #dsr = models.DenoisingAutoEncoderSR()
    #dsr.upscale(path, scale_factor=scale, save_intermediate=False, suffix="dsr")

    #ddsr = models.DeepDenoiseSR()
    #ddsr.upscale(path, scale_factor=scale, save_intermediate=False, suffix="ddsr")



