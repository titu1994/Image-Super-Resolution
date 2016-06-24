import numpy as np
from scipy.misc import imsave, imread, imresize

def upscaleImage(imgPath, scalingFactor=2, save_intermediate=False):
    import os
    import theano
    # Flag that may cause crash if algo_fwd = 'time_once'
    theano.config.dnn.conv.algo_fwd = 'small'

    from ImageSRModel import createModel

    # Read image
    img = imread(imgPath, mode='RGB')
    height, width = img.shape[0], img.shape[1]
    print("Old Size : ", img.shape)

    # Pre Upscale
    img = imresize(img, (height * scalingFactor, width * scalingFactor))
    height, width = img.shape[0], img.shape[1]
    print("New Size : ", img.shape)

    # Transpose and Process image
    img_conv = img.transpose((2, 0, 1)).astype('float64') / 255
    img_conv = np.expand_dims(img_conv, axis=0)

    model = createModel(height, width)
    print("Model loaded.")

    # Create prediction for image
    result = model.predict(img_conv, batch_size=1)
    print("Model finished upscaling.")

    # Evaluate agains bilinear upscaled image. Just to measure if a good result was obtained.
    error = model.evaluate(result, img_conv)
    print("Mean Squared Error (Compared to bilinear upscaled version) : ", error[0])
    print("Peak Signal to Noise Ratio (Compared to bilinear upscaled version) : ", error[1])

    # Deprocess
    result = result.reshape((3, height, width))
    result = result.transpose((1, 2, 0)).astype('float64') * 255
    result = np.clip(result, 0, 255).astype('uint8')

    path = os.path.splitext(imgPath)
    filename = path[0] + "_scaled(%dx)" % (scalingFactor) + path[1]

    print("Saving image.")
    # Save intermediate bilinear scaled image is needed for comparison.
    if save_intermediate:
        fn = path[0] + "_intermediate" + path[1]
        imsave(fn, img)

    imsave(filename, result)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Up-Scales an image using Image Super Resolution Model")
    parser.add_argument("imgpath", type=str, help="Path to input image")
    parser.add_argument("--scale", default=2, help='Scaling factor. Default = 2x')
    parser.add_argument("--save_intermediate", dest='save', default='False', type='str',
                        help="Whether to save bilinear upscaled image")
    def strToBool(v):
        return v.lower() in ("true", "yes", "t", "1")

    args = parser.parse_args()

    path = args.imgpath
    scale_factor = args.scale
    save = strToBool(args.save)

    upscaleImage(path, scalingFactor=scale_factor, save_intermediate=save)
