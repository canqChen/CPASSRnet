import argparse
import os
import tensorflow as tf
from CPASSRnet import CPASSRnet
from utils import *
import numpy as np


"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Implementation of CPASSRnet')

    parser.add_argument('--seed', type=int, default=1, help='random seed')

    parser.add_argument('--phase', dest='phase', type=str, default='train', help='train, test')
    parser.add_argument('--dataset_name', type=str, default='middlebury', help='middlebury, flickr_1024')

    parser.add_argument('--epoch', dest='epoch', type=int, default=20, help='# of epoch')  # each epoch contains 6k+ steps, total 100k+ steps

    parser.add_argument('--decay_epoch', dest='decay_epoch', type=int, default=10, help='# of epoch to decay lr')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=6, help='# images in batch')

    parser.add_argument('--sr_lr', type=float, default=0.0001, help='initial learning rate for sr net')

    parser.add_argument('--beta1', type=float, default=0.9, help='momentum of optimizer ')


    # network properties
    parser.add_argument('--lambda_per', type=float, default=0.1, help='weight for l1 perceptual loss')

    parser.add_argument('--lambda_recon', type=float, default=10.0, help='weight for L1 loss term')
    
    parser.add_argument('--use_per', type=str2bool, default=True, help='l1 perceptual loss use or not')

    parser.add_argument('--nf', type=int, default=64, help='number of filters in network conv layer')

    parser.add_argument('--n_down', type=int, default=4, help='number of downsampling in unet')

    # image properties
    parser.add_argument('--img_ch', dest='img_ch', type=int, default=3, help='# of input image channels')
    parser.add_argument('--img_h', type=int, default=96, help='image height')
    parser.add_argument('--img_w', type=int, default=288, help='image width')

    parser.add_argument('--epoch_to_restore', type=int, default=-1, help='epoch to restore')

    parser.add_argument('--save_freq', dest='save_freq', type=int, default=1, help='save a model every save_freq epoch')
    parser.add_argument('--sample_freq', dest='sample_freq', type=int, default=1000, help='sample generated images every sample_freq iterations')
    # directories
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
    parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
    parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
    parser.add_argument('--logs_dir', dest='logs_dir', default='./logs', help='logs are saved here') 

    parser.add_argument('--visible_devices', dest='visible_devices', type=str, default=None, help='set cuda_visible_devices')

    return parser.parse_args()


def main(_):
    args = parse_args()
    if args.visible_devices != None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
        
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = CSSR(sess, args)
        if args.phase == 'train':
            model.train()
            print(' [*] Training finished!')
        elif args.phase=='test':
            # model.test()
            model.fraction_test()
            print(' [*] Test finished!')
        else:
            print(' [!] Phase error!')

if __name__ == '__main__':
    tf.app.run()
