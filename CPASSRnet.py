from __future__ import division
import os
import time
from glob import glob
import pandas as pd
import tensorflow as tf
import numpy as np
from vgg19 import VGG19
from tensorflow.contrib.data import map_and_batch, shuffle_and_repeat
from utils import *
from ops import *
from skimage import measure


class CPASSRnet(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.model_name = "CPASSRnet"

        self.epoch_to_restore = args.epoch_to_restore
        self.is_training = (args.phase == 'train')

        self.gpu_num = len(args.visible_devices.split(','))

        self.dataset_name = args.dataset_name
        self.vgg = VGG19()
        self.content_layers = ['relu1_1', 'pool1',
            'conv2_2']  # cal content loss

        self.epoch = args.epoch

        self.decay_epoch = args.decay_epoch
        self.batch_size = args.batch_size

        self.sr_init_lr = args.sr_lr

        self.beta1 = args.beta1

        self.img_h = args.img_h
        self.img_w = args.img_w

        self.img_ch = args.img_ch

        self.lambda_per = args.lambda_per

        self.lambda_recon = args.lambda_recon

        self.save_freq = args.save_freq
        self.sample_freq = args.sample_freq

        self.nc = args.nf
        self.num_downs = args.n_down

        self.sample_dir = check_folder(
            os.path.join(args.sample_dir, self.model_dir))
        self.checkpoint_dir = check_folder(
            os.path.join(args.checkpoint_dir, self.model_dir))
        self.test_dir = check_folder(
            os.path.join(args.test_dir, self.model_dir))
        self.logs_dir = check_folder(
            os.path.join(args.logs_dir, self.model_dir))

        self.train_left_paths = glob(os.path.join('./datasets', self.dataset_name, 'train_patches_%d_%d' % (self.img_h, self.img_w), '*L*.*'))
        self.train_right_paths = [
            x.replace('L', 'R') for x in self.train_left_paths]
        self.train_pair_paths = list(
            zip(self.train_left_paths, self.train_right_paths))
        self.dataset_num = len(self.train_pair_paths)

        self._build_model()
        show_all_variables()

    def network(self, stereo_imgs, scope="CPASSRnet"):
        b, ori_h, ori_w, c = stereo_imgs.get_shape().as_list()
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            channels = self.nc
            max_channel = 1024
            norm = None
            down_out = []

            out = conv_na(stereo_imgs, channels, ks=7, s=1,
                          norm=norm, act=None, scope='feat_conv')

            down_out.append(out)

            for i in range(1, self.num_downs+1):
                out = conv_na(out, min(max_channel, channels*(2**i)), ks=3, s=1,
                              use_bias=True, norm=norm, act='relu', scope='down_conv%d' % i)
                out = max_pool(out, scope='maxpool%d' % i)
                out = resblock_V1(out, min(max_channel, channels*(2**i)), ks=3, s=1,
                                  use_bias=True, norm=norm, act='relu', scope='down_res%d' % i)

                if i != self.num_downs:
                    down_out.append(out)

            for i in np.arange(self.num_downs-1, -1, -1):
                _, out_h, out_w, _ = down_out[i].get_shape().as_list()

                out = upsample_conv_na(out, min(max_channel, channels*(2**(i))), ks=3, s=1, out_h=out_h,
                                       out_w=out_w, use_bias=True, norm=norm, act='relu', scope='up_conv%d' % (self.num_downs-i))

                out = out + conv_na(down_out[i], min(max_channel, channels*(2**(i))), ks=1, s=1,
                                    use_bias=True, norm=norm, act='relu', scope='up_sample%d_fuse' % (self.num_downs-i))

                left_tmp = CPAM(out[0:b//2, :, :, :], out[b//2::, :, :, :],
                                scope='attention_block_left_%d' % (self.num_downs-i))
                right_tmp = CPAM(out[b//2::, :, :, :], out[0:b//2, :, :, :],
                                 scope='attention_block_right_%d' % (self.num_downs-i))

                out = tf.concat(values=[left_tmp, right_tmp], axis=0)

                out = resblock_V1(out, min(max_channel, channels*(2**(i))), ks=3, s=1, use_bias=True,
                                  norm=norm, act='relu', scope='up_sample%d_resblock' % (self.num_downs-i))

            out = conv_na(out, c, ks=7, s=1, use_bias=False,
                          act=None, scope='pred')

            return out+stereo_imgs

    def _build_model(self):

        self.left_hr_phd = tf.placeholder(tf.float32, shape=[
                                              self.batch_size, self.img_h, self.img_w, self.img_ch], name='left_hr')
        self.right_hr_phd = tf.placeholder(tf.float32, shape=[
                                               self.batch_size, self.img_h, self.img_w, self.img_ch], name='right_hr')
        self.left_lr_phd = tf.placeholder(tf.float32, shape=[
                                              self.batch_size, self.img_h, self.img_w, self.img_ch], name='left_lr')
        self.right_lr_phd = tf.placeholder(tf.float32, shape=[
                                               self.batch_size, self.img_h, self.img_w, self.img_ch], name='right_lr')

        self.left_hr, self.right_hr, self.left_lr, self.right_lr = self.left_hr_phd, self.right_hr_phd, self.left_lr_phd, self.right_lr_phd
        cat = tf.concat(
                values=[self.left_hr, self.right_hr, self.left_lr, self.right_lr], axis=-1)
        cat = tf.random_shuffle(cat)
            self.left_hr, self.right_hr, self.left_lr, self.right_lr = cat[:, :,
                :, 0:3], cat[:, :, :, 3:6], cat[:, :, :, 6:9], cat[:, :, :, 9::]

        left_lr_sp = tf.split(value=self.left_lr, num_or_size_splits=self.gpu_num, axis=0)
        left_hr_sp = tf.split(value=self.left_hr, num_or_size_splits=self.gpu_num, axis=0)
        right_lr_sp = tf.split(value=self.right_lr, num_or_size_splits=self.gpu_num, axis=0)
        right_hr_sp = tf.split(value=self.right_hr, num_or_size_splits=self.gpu_num, axis=0)

        sr_loss_list = []

        pred_list = []
        for gpu_id in range(self.gpu_num):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                real_stereo = tf.concat(values=[left_hr_sp[gpu_id], right_hr_sp[gpu_id]],axis=0)
                pred_stereo = self.network(tf.concat(values=[left_lr_sp[gpu_id], right_lr_sp[gpu_id]], axis=0))

                pred_list.append(pred_stereo)

                if self.use_per:
                    recon_feats = self.vgg.feed_forward(pred_stereo, scope='vgg19')
                    real_feats = self.vgg.feed_forward(real_stereo, scope='vgg19')

                # define losses
                # content loss
                if self.use_per:
                    for layer in self.content_layers:
                        tf.add_to_collection('content_loss', mae_loss(real_feats[layer], recon_feats[layer])) if self.use_mae else \
                            tf.add_to_collection('content_loss', mse_loss(real_feats[layer], recon_feats[layer])) 
                    loss_set = tf.get_collection('content_loss')
                    content_loss = self.lambda_per * tf.add_n(loss_set)
                else:
                    content_loss = 0.0

                # reconstruction loss
                recon_loss = self.lambda_recon * mae_loss(pred_stereo, real_stereo)  

                # total loss
                sr_loss = content_loss  + recon_loss

                sr_loss_list.append(sr_loss)

        self.pred_left = tf.concat(values=[x[0:self.batch_size,:,:,:] for x in pred_list], axis=0)
        self.pred_right = tf.concat(values=[x[self.batch_size::,:,:,:] for x in pred_list], axis=0)

        self.sr_loss = tf.reduce_mean(sr_loss_list)

        # sr summary
        self.sr_loss_sum = tf.summary.scalar("sr_loss", self.sr_loss)

        self.sr_sum = tf.summary.merge([self.sr_loss_sum])

        # training params
        t_vars = tf.trainable_variables()

        self.sr_vars = [var for var in t_vars if ('CPASSRnet' in var.name)]
        self.sr_lr = tf.placeholder(tf.float32, None, name='sr_learning_rate')
        self.sr_optim = tf.train.AdamOptimizer(self.sr_lr, beta1=self.beta1).minimize(self.sr_loss, var_list=self.sr_vars)

    def train(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=100)
        # summary writer
        self.writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)
        # steps in each epoch
        steps = (self.dataset_num // self.batch_size) *3

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter + 1
            start_epoch = checkpoint_counter//steps + 1
            start_step = checkpoint_counter % steps + 1
            print(" [*] Successfully Loaded !")
        else:
            counter = 1
            start_epoch = 1
            start_step = 1
            print(" [!] No Checkpoint To Load...")

        start_time = time.time()
        for epoch in range(start_epoch, self.epoch + 1):
            # linearly decay learning rate 
            sr_lr = self.sr_init_lr if epoch<=self.decay_epoch else np.maximum(self.sr_init_lr * (1-(epoch-self.decay_epoch)/(self.epoch-self.decay_epoch)), 1e-7)
            np.random.shuffle(self.train_pair_paths)
            for idx in range(start_step, steps+1):
                # optimization
                # load data
                left, right, left_lr, right_lr = load_data_preup(self.train_pair_paths[((idx-1)*self.batch_size*self.gpu_num)//3:(idx*self.batch_size*self.gpu_num)//3])

                _, sr_loss, real_left, real_right, pred_left, pred_right, sr_summary_str = self.sess.run([self.sr_optim, self.sr_loss, self.left_hr, self.right_hr, self.pred_left, self.pred_right, self.sr_loss_sum],
                                            feed_dict={self.sr_lr: sr_lr, self.left_hr_phd:left, self.right_hr_phd:right, self.left_lr_phd:left_lr, self.right_lr_phd:right_lr})
                    
                print(("Epoch: [%3d/%3d] [%4d/%4d] time: %4.4f, sr loss: %.4f"\
                                % (epoch, self.epoch, idx, steps, time.time() - start_time, sr_loss)))

                # sample results during training
                if counter % self.sample_freq == 0:
                    integ_img = (np.clip(np.concatenate((pred_left, real_left, pred_right, real_right), axis=0),0.,1.) * 255.).round().astype(np.uint8)
                    imsave(integ_img, size=[4, self.batch_size], path='./{}/sample_{:03d}_{:04d}.png'.format(self.sample_dir, epoch, idx))

                # record summary data
                if counter % 10 == 0:
                    self.writer.add_summary(sr_summary_str, counter)
                    if self.use_d: self.writer.add_summary(d_summary_str, counter)
                counter += 1
            if epoch % self.save_freq == 0:
                self.save(self.checkpoint_dir, counter-1)

    def test(self):
        import re
        noise = False
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.saver = tf.train.Saver()
        # load all the images in the test set

        test_left_paths = glob(os.path.join('./datasets', self.dataset_name, 'test', '*L*.*'))
        test_left_paths = sorted(test_left_paths)
        test_right_paths = [x.replace('L','R') for x in test_left_paths]
        test_pair_paths = list(zip(test_left_paths, test_right_paths))

        # load model
        could_load, _ = self.load(self.checkpoint_dir, self.epoch_to_restore):
        if could_load:
            print(" [*] Successfully Loaded !")
        else:
            print(" [!] Load Failed...\n [!] Exit...")
            return None

        scales = [2,3,4]

        nan_excel = pd.DataFrame()
        logs_filename = os.path.join(check_folder('./test_logs' ), 'logs_{}.xlsx'.format(self.dataset_name)) 
        nan_excel.to_excel(logs_filename)

        writer = pd.ExcelWriter(logs_filename)
        
        for scale in scales:
            left_imgs = []
            right_imgs = []
            left_psnr = []
            right_psnr = []
            left_ssim = []
            right_ssim = []

            test_folder = check_folder(os.path.join(self.test_dir, 'x_%d'%scale))
            # test each image in the test set
            for pair_path in test_pair_paths:
                left_name = os.path.basename(pair_path[0]).split('.')[0]
                right_name = os.path.basename(pair_path[1]).split('.')[0]
                left_imgs.append(left_name)
                right_imgs.append(right_name)
                print('Processing image: {} and {} for scale {}'.format(left_name, right_name, scale))
                # test
                
                left_hr, left_lr, right_hr, right_lr = load_stereo_images_preup(pair_path, scale)

                h,w,c = left_hr.shape
                
                left_lr = np.array([left_lr]).astype(np.float32)
                right_lr = np.array([right_lr]).astype(np.float32)
                

                test_lr = tf.placeholder(tf.float32, [2, h, w, c], name='test_lr')

                test_pred = self.network(test_lr)

                pred = self.sess.run(test_pred, 
                            feed_dict={test_lr: np.concatenate((left_lr,right_lr), axis=0)})

                pred_left = pred[0:1,:,:,:]
                pred_right = pred[1::,:,:,:]

                
                # post processing
                pred_left = (np.clip(pred_left.squeeze(),0.,1.)* 255.)  # H x W x 3 [0,255]
                pred_right = (np.clip(pred_right.squeeze(),0.,1.)* 255.)  # H x W x 3  [0,255]

                # calculate psnr

                psnr_l = measure.compare_psnr(pred_left, left_hr, data_range=255)
                left_psnr.append(psnr_l)

                psnr_r = measure.compare_psnr(pred_right, right_hr, data_range=255)
                right_psnr.append(psnr_r)
                print("Psnr values of %s and %s for scale %d are: %f , %f"%(left_name, right_name, scale, psnr_l, psnr_r))

                # cal ssim

                ssim_l = measure.compare_ssim(pred_left, left_hr, data_range=255, multichannel=True, gaussian_weights=True)
                left_ssim.append(ssim_l)

                ssim_r = measure.compare_ssim(pred_right, right_hr, data_range=255, multichannel=True, gaussian_weights=True)
                right_ssim.append(ssim_r)
                print("ssim values of %s and %s for scale %d are: %f , %f"%(left_name, right_name, scale, ssim_l, ssim_r))

                # save test result
                # path to save test result
                
                left_save_path = os.path.join(test_folder, 'test_x{}_{}'.format(scale, os.path.basename(pair_path[0])))
                right_save_path = os.path.join(test_folder, 'test_x{}_{}'.format(scale, os.path.basename(pair_path[1])))
                left_hr_save_path = os.path.join(test_folder, 'hr_x{}_{}'.format(scale, os.path.basename(pair_path[0])))
                right_hr_save_path = os.path.join(test_folder, 'hr_x{}_{}'.format(scale, os.path.basename(pair_path[1])))

                imsave(np.array([pred_left.astype(np.uint8)]), size=[1,1], path=left_save_path)
                imsave(np.array([pred_right.astype(np.uint8)]), size=[1,1], path=right_save_path)
                imsave(np.array([left_hr.astype(np.uint8)]), size=[1,1], path=left_hr_save_path)
                imsave(np.array([right_hr.astype(np.uint8)]), size=[1,1], path=right_hr_save_path)

            left_imgs.append('mean')
            right_imgs.append('mean')
            left_psnr.append(np.mean(left_psnr))
            right_psnr.append(np.mean(right_psnr))
            left_ssim.append(np.mean(left_ssim))
            right_ssim.append(np.mean(right_ssim))
            df = pd.DataFrame({'left_img':left_imgs,'psnr_l':left_psnr,'ssim_l':left_ssim,
                                'right_img':right_imgs,'psnr_r':right_psnr,'ssim_r':right_ssim})
            df.to_excel(writer, sheet_name='x%d'%scale)
        writer.save()

    def save(self, checkpoint_dir, step):
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir, epoch_to_restore=-1):
        import re
        print(" [*] Loading Checkpoint...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path) if epoch_to_restore==-1 else os.path.basename(ckpt.all_model_checkpoint_paths[epoch_to_restore-1])
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            return True, counter
        else:
            return False, 0
            
    @property
    def model_dir(self):
        return "{}_{}_{}_x2-3-4".format(self.model_name, self.img_h, self.img_w)

