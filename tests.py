from keras.utils.visualize_util import plot
import models
import img_utils

if __name__ == "__main__":
    path = r""
    val_path = "val_images/"

    scale = 2

    """
    Plot the models
    """

    # model = models.ImageSuperResolutionModel(scale).create_model()
    # plot(model, to_file="architectures/SRCNN.png", show_shapes=True, show_layer_names=True)
    #
    # model = models.ExpantionSuperResolution(scale).create_model()
    # plot(model, to_file="architectures/ESRCNN.png", show_layer_names=True, show_shapes=True)
    #
    # model = models.DenoisingAutoEncoderSR(scale).create_model()
    # plot(model, to_file="architectures/Denoise.png", show_layer_names=True, show_shapes=True)
    #
    # model = models.DeepDenoiseSR(scale).create_model()
    # plot(model, to_file="architectures/Deep Denoise.png", show_layer_names=True, show_shapes=True)
    #
    # model = models.ResNetSR(scale).create_model()
    # plot(model, to_file="architectures/ResNet.png", show_layer_names=True, show_shapes=True)

    """
    Train Super Resolution
    """

    # sr = models.ImageSuperResolutionModel(scale)
    # sr.create_model()
    # sr.fit(nb_epochs=300)

    """
    Train ExpantionSuperResolution
    """

    # esr = models.ExpantionSuperResolution(scale)
    # esr.create_model(load_weights=False)
    # esr.fit(nb_epochs=250)

    """
    Train DenoisingAutoEncoderSR
    """

    dsr = models.DenoisingAutoEncoderSR(scale)
    dsr.create_model()
    dsr.fit(nb_epochs=250)

    """
    Train Deep Denoise SR
    """

    #ddsr = models.DeepDenoiseSR()
    #ddsr.create_model()
    #ddsr.fit(scale_factor=scale, nb_epochs=60)

    """
    Train Res Net SR
    """

    # rnsr = models.ResNetSR()
    # rnsr.create_model()
    # rnsr.fit(scale_factor=scale, nb_epochs=250)

    """
    Evaluate Super Resolution on Set5
    """

    # sr = models.ImageSuperResolutionModel()
    # sr.evaluate(val_path, scale_factor=scale)

    """
    Evaluate ESRCNN on Set5
    """

    #esr = models.ExpantionSuperResolution(scale)
    #esr.evaluate(val_path)

    """
    Evaluate DSRCNN on Set5 cannot be performed at the moment.
    This is because this model uses Deconvolution networks, whose output shape must be pre determined.
    This causes the model to fail to predict different images of different image sizes.
    """

    dsr = models.DenoisingAutoEncoderSR(scale)
    dsr.evaluate(val_path)

    """
    Evaluate DDSRCNN on Set5
    """

    #ddsr = models.DeepDenoiseSR()
    #ddsr.evaluate(val_path, scale_factor=scale)

    """
    Compare output images of sr, esr, dsr and ddsr models
    """

    #sr = models.ImageSuperResolutionModel()
    #sr.upscale(path, scale_factor=scale, save_intermediate=False, suffix="sr")

    #esr = models.ExpantionSuperResolution()
    #esr.upscale(path, scale_factor=scale, save_intermediate=False, suffix="esr")

    #dsr = models.DenoisingAutoEncoderSR()
    #dsr.upscale(path, scale_factor=scale, save_intermediate=False, suffix="dsr")

    #ddsr = models.DeepDenoiseSR()
    #ddsr.upscale(path, scale_factor=scale, save_intermediate=False, suffix="ddsr")


