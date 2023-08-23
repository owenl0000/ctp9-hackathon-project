"""
This file contains a script to load a pickled GAN model and use it to animefy an image.
"""
import sys
sys.path.append("pytorch-CartoonGAN")

import os
import time
import pickle
import argparse
import torch
from torch import nn
import networks

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='CartoonGan_Converter',  help='')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--generator_features', type=int, default=64)
parser.add_argument('--in_g_channel', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_g_channel', type=int, default=3, help='output channel for generator')
args = parser.parse_args()

def main():
    """
    Main function where the unpickling and animefying ocurs.

    This function takes in nothing.
    This function loads the GAN model from a pre-established directory
    and feeds it an image from another pre-established directory.
    The result output image is stored in a third directory.
    """
    # load the model
    generator = networks.generator(args.in_g_channel, args.out_g_channel, args.generator_features, args.batch_size)
    generator.load_state_dict(torch.load(os.path.join(args.name + '_results',  'generator_param.pkl')))
    generator.eval()
    # pass an image to the model
    # save the output image
    return 0

if __name__ == "__main__":
    main()
