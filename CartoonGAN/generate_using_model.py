"""
This file contains a script to load a pickled GAN model and use it to animefy an image.
"""
import sys
sys.path.append("pytorch-CartoonGAN")

import os
import argparse
import torch
from torchvision.transforms import functional
import networks
from PIL import Image

# argument parsed from the command line
# lifted from train.py
parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='CartoonGan_Converter',  help='')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--generator_features', type=int, default=64)
parser.add_argument('--in_g_channel', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_g_channel', type=int, default=3, help='output channel for generator')
args = parser.parse_args()

def main():
    """
    Main function.

    This function takes in nothing.
    This function loads the GAN model from a pre-established directory
    and feeds it an image from another pre-established directory.
    The result output image is stored in a third directory.
    Currently the directories are placeholders and
    will be replaced in another commit.
    """
    # load the model
    generator = networks.generator(args.in_g_channel, args.out_g_channel, args.generator_features, args.batch_size)
    generator.load_state_dict(torch.load(os.path.join(args.name + '_results',  'generator_param.pkl')))
    generator.eval()
    # pass an image to the model
    # reference: https://discuss.pytorch.org/t/how-to-read-just-one-pic/17434
    # replace placeholders with dynamic paths later
    input = Image.open("PLACEHOLDER.png")
    tensor = functional.to_tensor(input)
    tensor.unsqueeze(0)
    output_tensor = generator(tensor)
    # save the output image
    output_tensor.squeeze(0)
    output = functional.to_pil_image(output_tensor)
    output = output.save("PLACEHOLDER_OUTPUT.png")
    return 0

if __name__ == "__main__":
    main()
