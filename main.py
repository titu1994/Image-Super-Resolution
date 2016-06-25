import models

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Up-Scales an image using Image Super Resolution Model")
    parser.add_argument("imgpath", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="sr", help="Use either image super resolution (sr) or expanded super resolution (esr)")
    parser.add_argument("--scale", default=2, help='Scaling factor. Default = 2x')
    parser.add_argument("--save_intermediate", dest='save', default='False', type='str',
                        help="Whether to save bilinear upscaled image")
    def strToBool(v):
        return v.lower() in ("true", "yes", "t", "1")

    args = parser.parse_args()

    path = args.imgpath
    model_type = str(args.model).lower()
    assert model_type in ["sr", "esr"], 'Model type must be either "sr" or "esr"'

    scale_factor = args.scale
    save = strToBool(args.save)

    model = models.ImageSuperResolutionModel() if model_type == "sr" else models.ExpantionSuperResolution()
    model.upscale(path, scale_factor=scale_factor, save_intermediate=save)