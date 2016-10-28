import models
import argparse

parser = argparse.ArgumentParser(description="Up-Scales an image using Image Super Resolution Model")
parser.add_argument("imgpath", type=str, help="Path to input image")
parser.add_argument("--model", type=str, default="dsr", help="Use either image super resolution (sr), "
                        "expanded super resolution (esr), denoising auto encoder sr (dsr), "
                        "deep denoising sr (ddsr) or res net sr (rnsr)")
parser.add_argument("--scale", default=2, help='Scaling factor. Default = 2x')
parser.add_argument("--mode", default="patch", type=str, help='Mode of operation. Choices are "fast" or "patch"')
parser.add_argument("--save_intermediate", dest='save', default='False', type=str,
                        help="Whether to save bilinear upscaled image")
parser.add_argument("--suffix", default="scaled", type=str, help='Suffix of saved image')
parser.add_argument("--patch_size", type=int, default=8, help='Patch Size')

def strToBool(v):
    return v.lower() in ("true", "yes", "t", "1")

args = parser.parse_args()

path = args.imgpath
suffix = args.suffix

model_type = str(args.model).lower()
assert model_type in ["sr", "esr", "dsr", "ddsr", "rnsr"], 'Model type must be either "sr", "esr", "dsr", ' \
                                                           '"ddsr" or "rnsr"'

mode = str(args.mode).lower()
assert mode in ['fast', 'patch'], 'Mode of operation must be either "fast" or "patch"'

scale_factor = int(args.scale)
save = strToBool(args.save)

patch_size = int(args.patch_size)
assert patch_size > 0, "Patch size must be a positive integer"

if model_type == "sr":
    model = models.ImageSuperResolutionModel(scale_factor)
elif model_type == "esr":
    model = models.ExpantionSuperResolution(scale_factor)
elif model_type == "dsr":
    model = models.DenoisingAutoEncoderSR(scale_factor)
elif model_type == "ddsr":
    model = models.DeepDenoiseSR(scale_factor)
elif model_type == "rnsr":
    model = models.ResNetSR(scale_factor)
else:
    model = models.DeepDenoiseSR(scale_factor)

model.upscale(path, save_intermediate=save, mode=mode, patch_size=patch_size, suffix=suffix)


'''


import theano.tensor as T

    scale = int(scale)

    if dim_ordering == "th":
        x = x.transpose((0, 2, 3, 1)) # batch, width, height, channels

    b, r, c, k = x.shape
    out_b, out_k, out_r, out_c = b, k // (scale * scale), r * scale, c * scale

    out = T.zeros((out_b, out_r, out_c, out_k), dtype='float32')

    for i in range(out_r):
        for j in range(out_c):
            for channel in range(channels):
                channel += 1
                a = T.floor(i / scale).astype('int32')
                b = T.floor(j / scale).astype('int32')
                d = channel * scale * (j % scale) + channel * (i % scale)

                out = T.set_subtensor(out[:, i, j, channel - 1], x[:, a, b, d])

    if dim_ordering == 'th':
        out = out.transpose((0, 3, 1, 2))

    return out
'''