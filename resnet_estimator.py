from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

tf.logging.set_verbosity(tf.logging.INFO)

from functools import partial
from glob import glob
import sys
import getopt
#import os
#home = os.path.expanduser("~")
#user = home.split('/')[-1]
#sys.path.append(home + '/tflib/')
from model import SR_net_model as model
from utils import *



def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    logits = model(features, mode)
    predictions = {
        # Can save anything needed into this dictionary.
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1, output_type=tf.int32),
        # Add `softmax_tensor` to the graph. It is used for PREDICT.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    labels = tf.cast(labels, tf.int32)
    oh = tf.one_hot(labels, 2)
    xen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=oh,logits=logits))  
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([xen_loss] + reg_losses)
    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Horovod: scale learning rate by the number of workers.
        optimizer = AdamaxOptimizer
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.piecewise_constant(global_step, params['boundaries'], params['values'])  
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = optimizer(learning_rate)
        
        tf.summary.scalar('train_accuracy', accuracy[1]) # output to TensorBoard
        
        # Horovod: add Horovod Distributed Optimizer.
        optimizer = hvd.DistributedOptimizer(optimizer)
        
        # Update batch norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss=loss,
                global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    
    # Horovod: calculate loss from all ranks
    loss = hvd.allreduce(loss)
    
    if params['test']:
        eval_metric_ops = {
            "test_accuracy": (hvd.allreduce(accuracy[0]), accuracy[1])} # Horovod: allreduce just accuracy value, not the update_op
    else:
        eval_metric_ops = {
            "valid_accuracy": (hvd.allreduce(accuracy[0]), accuracy[1])}    
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(argv):
    # Horovod: initialize Horovod.
    hvd.init()
    
    # Global parameters
    DB_FOLDER = '/media/nvme/Honza/DataBase/'
    ALGORITHM = 'MIPOD'
    PAYLOAD = '0.4'

    train_interval = 100
    eval_interval = 3000
    save_interval = 100
    train_batch_size = 64
    max_iter = 500000//(train_batch_size//32)
    eval_batch_size = 40
    num_of_threads = 5
    
    warm_start_checkpoint = None
    #warm_start_checkpoint = LOG_DIR + 'model.ckpt-323'
    load_checkpoint = 'last' #'model.ckpt-1288'
    test = False
    predict = False
    
    try:
        opts, args = getopt.getopt(argv[1:], 'l:g:p:a:test:w:c:predict:', \
                                   ['logdir=', 'gpus=', 'payload=', 'algorithm=', \
                                    'test=', 'warm_start=', 'checkpoint=', 'predict='])
    except getopt.GetoptError:
        print 'usage: ' + argv[0] \
        + ' -l <logdir> -g <gpu0,gpu1,...,gpuN> -p <payload> -a <algorithm> ' \
        + '-t <0 for training, 1 for test> -w <path_to_warmup_checkpoint> ' \
        + '-c <checkpoint_name> -pr <1 for predictions, 0 else>'
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-l', '--logdir'):
            LOG_DIR = arg
        elif opt in ('-g', '--gpus'):
            GPUs = np.array(arg.split(',')).astype(np.uint8)
        elif opt in ('-p', '--payload'):
            PAYLOAD = arg
        elif opt in ('-a', '--algorithm'):
            ALGORITHM = arg
        elif opt in ('--test'):
            test = bool(int(arg))
        elif opt in ('-w', '--warm_start'):
            warm_start_checkpoint = arg
        elif opt in ('-c', '--checkpoint'):
            load_checkpoint = arg
        elif opt in ('--predict'):
            predict = bool(int(arg))
        else:
            print 'Unknown opt ' + opt
            sys.exit()
       
    # Horovod: adjust number of steps based on number of GPUs.
    num_of_workers = hvd.size()
    worker_idx = hvd.rank()
    max_iter = max_iter//num_of_workers
    eval_interval = eval_interval//num_of_workers
    boundaries = [(400000//(train_batch_size//32))//num_of_workers]
    values = [0.001, 0.0001]
    values = [val*(train_batch_size//32)*num_of_workers for val in values]
            
    CoverValPath = DB_FOLDER + 'Cover_VAL/'
    StegoValPath = DB_FOLDER + ALGORITHM + '/' + PAYLOAD + '/Stego_' + ALGORITHM + '_' + PAYLOAD + '_VAL/'
    CoverTrnPath = DB_FOLDER + 'Cover_TRN/'
    StegoTrnPath = DB_FOLDER + ALGORITHM + '/' + PAYLOAD + '/Stego_' + ALGORITHM + '_' + PAYLOAD + '_TRN/'
    CoverTstPath = DB_FOLDER + 'Cover_TST/'
    StegoTstPath = DB_FOLDER + ALGORITHM + '/' + PAYLOAD + '/Stego_' + ALGORITHM + '_' + PAYLOAD + '_TST/'

    
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(GPUs[hvd.local_rank()])
    
    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    if not (test or predict):
        model_dir = LOG_DIR if hvd.rank() == 0 else None
    else: # if testing or predicting on multiple workers, load all of them
        model_dir = LOG_DIR

    # Create the Estimator
    resnet_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir,
        params={'boundaries': boundaries, 'values': values, 'test': test},
        config=tf.estimator.RunConfig(save_summary_steps=save_interval,
                                      save_checkpoints_steps=eval_interval,
                                      session_config=config,
                                     keep_checkpoint_max=10000),
        warm_start_from=warm_start_checkpoint)
    
    # If warm_start, training starts with global_step=0, 
    # use this for loading checkpoints from different folders or for transfer learning.
    # If specific checkpoint is to be loaded, checkpoint file will be updated
    if warm_start_checkpoint is not None:
        start = 0
        if hvd.rank() == 0: # it's enough to delete the file just once
            deleteCheckpoiontFile()
    elif load_checkpoint == 'last' or load_checkpoint is None:
        start = getLatestGlobalStep(LOG_DIR)
    else:
        start = int(load_checkpoint.split('-')[-1])
        if hvd.rank() == 0:
            updateCheckpointFile(LOG_DIR, load_checkpoint)
    if hvd.rank() == 0:        
        print('global step: ', start)
    
    # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
    # rank 0 to all other processes. This is necessary to ensure consistent
    # initialization of all workers when training is started with random weights or
    # restored from a checkpoint.
    bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
    
    if predict:
        predictions = resnet_classifier.predict(input_fn=partial(input_fn,
                                                                   CoverTstPath, StegoTstPath,
                                                                   eval_batch_size),
                                                predict_keys='classes', #can specify any key (or None) in predictions variable
                                                hooks=[bcast_hook])
        # Do whatever with predictions
        # Predictions is a generator now
        # Following line prints testing accuracy (if run on single GPU)
        test_ds_size = glob(
        print(np.sum(np.equal([p['classes'] for p in predictions], [0,1]*5000))/10000.)
        sys.exit()
    if test:
        test_results = resnet_classifier.evaluate(input_fn=partial(input_fn,
                                                                   CoverTstPath, StegoTstPath,
                                                                   eval_batch_size),
                                                  steps=None,
                                                  hooks=[bcast_hook])
        print(test_results)
        sys.exit()

    # Train the model
    for i in range(start,max_iter,eval_interval):
        
        resnet_classifier.train(
            input_fn=partial(input_fn,
                             CoverTrnPath, StegoTrnPath, 
                             train_batch_size, num_of_threads,
                             True),
            steps=eval_interval,
            hooks=[bcast_hook])
        if hvd.rank() == 0:
            deleteExtraCheckpoints(LOG_DIR, i+1)
        # Evaluate the model and print results
        eval_results = resnet_classifier.evaluate(input_fn=partial(input_fn,
                                                                 CoverValPath, StegoValPath, 
                                                                 eval_batch_size),
                                                 steps=None,
                                                 hooks=[bcast_hook])
        print(eval_results)

if __name__ == '__main__':
    tf.app.run(main, sys.argv)

