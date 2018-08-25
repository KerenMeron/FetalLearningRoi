'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-19 03:06:32
 * @modify date 2017-05-19 03:06:32
 * @desc [description]
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

SEED=0 # set set to allow reproducing runs
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)

import time
import os, shutil

from model import UNet
from net_utils import dice_coef, dice_coef_loss
from loader import dataLoader
from net_utils import VIS, mean_IU
# configure args
from opts import *
from opts import dataset_mean, dataset_std # set them in opts

CHECKPOINT = os.getcwd() + "\\checkpoints"

# save and compute metrics
vis = VIS(save_path=CHECKPOINT)

# configuration session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


''' Users define data loader (with train and test) '''
img_shape = [opt.imSize, opt.imSize, opt.num_channels]
train_generator, train_samples = dataLoader(os.getcwd()+'\\train\\', opt.batch_size,img_shape, mean=dataset_mean,
                                            std=dataset_std)
test_generator, test_samples = dataLoader(os.getcwd()+'\\test\\', 1,  img_shape, train_mode=False,mean=dataset_mean, std=dataset_std)
# test_generator, test_samples = dataLoader(opt.data_path+'/test2/', 1,  img_shape, train_mode=False,mean=dataset_mean, std=dataset_std)

if opt.iter_epoch == 0:
    opt.iter_epoch = int(train_samples)
# define input holders
label = tf.placeholder(tf.int32, shape=[None]+img_shape[:-1])
# define model
with tf.name_scope('unet'):
    model = UNet().create_model(img_shape=img_shape, num_class=opt.num_class)
    img = model.input
    pred = model.output


# define loss
# with tf.name_scope('cross_entropy'):
#     cross_entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=pred))

# compute dice score for simple evaluation during training
with tf.name_scope('dice_loss'):
    dice_loss = dice_coef_loss(label, pred)
# define optimizer
global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.name_scope('learning_rate'):
    learning_rate = tf.train.exponential_decay(opt.learning_rate, global_step,
                                           opt.iter_epoch, opt.lr_decay, staircase=True)
# train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss, global_step=global_step)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(dice_loss, global_step=global_step)


''' Tensorboard visualization '''
# cleanup pervious info
if opt.load_from_checkpoint == '':
    cf = os.listdir(CHECKPOINT)
    for item in cf:
        if 'event' in item:
            os.remove(os.path.join(CHECKPOINT, item))
# define summary for tensorboard
# tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
tf.summary.scalar('dice_loss', dice_loss)
tf.summary.scalar('learning_rate', learning_rate)
summary_merged = tf.summary.merge_all()
# define saver
train_writer = tf.summary.FileWriter(CHECKPOINT, sess.graph)
saver = tf.train.Saver() # must be added in the end

''' Main '''
tot_iter = opt.iter_epoch * opt.epoch
init_op = tf.global_variables_initializer()
sess.run(init_op)

with sess.as_default():
    # restore from a checkpoint if exists
    # the name_scope can not change
    last_checkpoint = tf.train.latest_checkpoint(CHECKPOINT)
    if last_checkpoint:
        try:
            print ('--> load from checkpoint '+CHECKPOINT)
            saver.restore(sess, last_checkpoint)
        except Exception as e:
            print ('unable to load checkpoint ...' + str(e))
    # debug
    start = global_step.eval()
    for it in range(start, tot_iter):
        if it % opt.iter_epoch == 0 or it == start:
            
            saver.save(sess, os.path.join(CHECKPOINT,'model'), global_step=global_step)
            print ('save a checkpoint at '+ CHECKPOINT+'/model-'+str(it))
            print ('start testing {} samples...'.format(test_samples))
            for ti in range(test_samples):
                x_batch, y_batch = next(test_generator)

                # tensorflow wants a different tensor ordermean_IU

                feed_dict = {
                                img: x_batch,
                                label: y_batch,
                            }
                # loss, pred_logits = sess.run([cross_entropy_loss, pred], feed_dict=feed_dict)
                loss, pred_logits = sess.run([dice_loss, pred], feed_dict=feed_dict)
                # pred_map_batch = np.argmax(pred_logits, axis=3)
                pred_map_batch = pred_logits > 0.5
                # import pdb; pdb.set_trace()
                for pred_map, y in zip(pred_map_batch, y_batch):
                    score = vis.add_sample(pred_map.squeeze(), y)
            vis.compute_scores(suffix=it)
        
        x_batch, y_batch = next(train_generator)

        feed_dict = {   img: x_batch,
                        label: y_batch
                    }

        # import numpngw
        # q = y_batch[0].squeeze().astype('uint8')
        # if (q > 0).any():
        #     numpngw.write_png('/tmp/blat.png', y_batch[0].squeeze().astype('uint8')*100)
        # q = y_batch[0].squeeze().astype('uint8')
        # if (q > 0).any():
        #     np.save('/tmp/blat.npy', q)

        _, loss, summary, lr, pred_logits = sess.run([train_step,
                                    # cross_entropy_loss,
                                    dice_loss,
                                    summary_merged,
                                    learning_rate,
                                    pred
                                    ], feed_dict=feed_dict)
        global_step.assign(it).eval()
        train_writer.add_summary(summary, it)
        
        pred_map = np.argmax(pred_logits[0], axis=2)
        score, _ = mean_IU(pred_map, y_batch[0])

       
        if it % 20 == 0 : 
            print('%s [iter %d, epoch %.3f]: lr=%f loss=%f, mean_IU=%f' % (time.ctime(), it, float(it)/opt.iter_epoch, lr, loss, score))
