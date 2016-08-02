import models

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Up-Scales an image using Image Super Resolution Model")
    parser.add_argument("imgpath", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="dsr", help="Use either image super resolution (sr), "
                        "expanded super resolution (esr), denoising auto encoder sr (dsr) or "
                        "deep denoising sr (ddsr)")
    parser.add_argument("--scale", default=2, help='Scaling factor. Default = 2x')
    parser.add_argument("--save_intermediate", dest='save', default='False', type='str',
                        help="Whether to save bilinear upscaled image")
    def strToBool(v):
        return v.lower() in ("true", "yes", "t", "1")

    args = parser.parse_args()

    path = args.imgpath

    model_type = str(args.model).lower()
    assert model_type in ["sr", "esr", "dsr", "ddsr"], 'Model type must be either "sr", "esr" or "dsr"'

    scale_factor = args.scale
    save = strToBool(args.save)

    if model_type == "sr":
        model = models.ImageSuperResolutionModel()
    elif model_type == "esr":
        model = models.ExpantionSuperResolution()
    elif model_type == "dsr":
        model = models.DenoisingAutoEncoderSR()
    elif model_type == "ddsr":
        model = models.DeepDenoiseSR()

    model.upscale(path, scale_factor=scale_factor, save_intermediate=save)