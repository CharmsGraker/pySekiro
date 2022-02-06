#!/usr/bin/env python

import torch

import getopt
import numpy
import PIL.Image
import sys

##########################################################
from .net.hed_net import Network

assert (int(str('').join(torch.__version__.split('.')[0:2])) >= 13)  # requires at least pytorch version 1.3.0

torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'bsds500'  # only 'bsds500' for now
arguments_strIn = './images/sample.png'
arguments_strOut = './out.png'

for strOption, strArgument in \
        getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--model' and strArgument != '': arguments_strModel = strArgument  # which model to use
    if strOption == '--in' and strArgument != '': arguments_strIn = strArgument  # path to the input image
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument  # path to where the output should be stored
# end

##########################################################


netNetwork = None


##########################################################

def estimate(tenInput):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cpu().eval()
    # end

    intWidth = tenInput.shape[2]
    intHeight = tenInput.shape[1]

    # assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    return netNetwork(tenInput.cpu().view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()


# end

##########################################################
def predict(raw_img):
    # preprocess the img
    tenInput = torch.FloatTensor(numpy.ascontiguousarray(
        numpy.array(raw_img)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
                1.0 / 255.0)))

    tenOutput = estimate(tenInput)

    out_sketch = (tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)

    return out_sketch


if __name__ == '__main__':
    save = False
    img = PIL.Image.open(arguments_strIn)
    out_sketch = predict(img)
    if save:
        PIL.Image.fromarray(out_sketch).save(arguments_strOut)
# end
