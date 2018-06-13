import tensorflow as tf
import os
import numpy as np
from scipy import misc
import random
import horovod.tensorflow as hvd

from tensorflow.python.training import optimizer
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops

# For AMSGrad
from tensorflow.python.eager import context
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope


class AMSGrad(optimizer.Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8, use_locking=False, name="AMSGrad"):
        super(AMSGrad, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        self._beta1_power = None
        self._beta2_power = None

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and context.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = variable_scope.variable(self._beta1, name="beta1_power", trainable=False)
                self._beta2_power = variable_scope.variable(self._beta2, name="beta2_power", trainable=False)
        # Create slots for the first and second moments.
        for v in var_list :
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "vhat", self._name)

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr)
        self._beta1_t = ops.convert_to_tensor(self._beta1)
        self._beta2_t = ops.convert_to_tensor(self._beta2)
        self._epsilon_t = ops.convert_to_tensor(self._epsilon)

    def _apply_dense(self, grad, var):
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_sqrt = math_ops.sqrt(vhat_t)

        var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _resource_apply_dense(self, grad, var):
        var = var.handle
        beta1_power = math_ops.cast(self._beta1_power, grad.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, grad.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, grad.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, grad.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, grad.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, grad.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m").handle
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v").handle
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat").handle
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_sqrt = math_ops.sqrt(vhat_t)

        var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_sqrt = math_ops.sqrt(vhat_t)
        var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(
            grad, var, indices, self._resource_scatter_add)

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(
                    self._beta1_power * self._beta1_t,
                    use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(
                    self._beta2_power * self._beta2_t,
                    use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                      name=name_scope)

class AdamaxOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adamax algorithm.
    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False, name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
        

def gen_flip_and_rot(cover_dir, stego_dir, im_name):
	cover = misc.imread(cover_dir + im_name)
	stego = misc.imread(stego_dir + im_name)
	batch = np.stack([cover,stego])
	rot = random.randint(0,3)
	if random.random() < 0.5:
		return [np.rot90(batch, rot, axes=[1,2]), np.array([0,1], dtype='uint8')]
	else:
		return [np.flip(np.rot90(batch, rot, axes=[1,2]), axis=2), np.array([0,1], dtype='uint8')]     
    
def gen_valid(cover_dir, stego_dir, im_name):
	cover = misc.imread(cover_dir + im_name)
	stego = misc.imread(stego_dir + im_name)
	batch = np.stack([cover,stego])
	return [batch, np.array([0, 1], dtype='uint8')]

def getLatestGlobalStep(LOG_DIR):
    # no directory
    if not os.path.exists(LOG_DIR):
        return 0
    checkpoints = [int(f.split('-')[-1].split('.')[0]) \
                   for f in os.listdir(LOG_DIR) if f.startswith('model.ckpt')]
    # no checkpoint files
    if not checkpoints:
        return 0
    global_step = max(checkpoints)
    to_file = open(LOG_DIR+'checkpoint', 'w')
    line = 'model_checkpoint_path: "model.ckpt-{}"'.format(global_step)
    to_file.write(line)
    to_file.close()
    return global_step

def updateCheckpointFile(LOG_DIR, checkpoint_name):
    if not os.path.isfile(LOG_DIR + 'checkpoint'):
        return 0
    from_file = open(LOG_DIR+'checkpoint')
    line = from_file.readline()
    from_file.close()
    splits = line.split('"')
    new_line = splits[0] + '"' + checkpoint_name + '"' + splits[-1]
    to_file = open(LOG_DIR + 'checkpoint', mode="w")
    to_file.write(new_line)
    to_file.close()
    
def deleteCheckpointFile(LOG_DIR):
    if os.path.isfile(LOG_DIR + 'checkpoint'):
        os.remove(LOG_DIR + 'checkpoint')
        
def deleteExtraCheckpoints(LOG_DIR, step):
    checkpoint = LOG_DIR + 'model.ckpt-' + str(step)
    os.remove(checkpoint + '.meta')
    os.remove(checkpoint + '.index')
    os.remove(checkpoint + '.data-00000-of-00001')

def input_fn(CoverPath, StegoPath, batch_size, num_of_threads=1, training=False):
    filenames = os.listdir(CoverPath)
    nb_data = len(os.listdir(StegoPath))
    assert len(filenames) != 0, "the cover directory '%s' is empty" % CoverPath
    assert nb_data != 0, "the stego directory '%s' is empty" % StegoPath
    assert len(filenames) == nb_data, "the cover directory and " + \
                                        "the stego directory don't " + \
                                        "have the same number of files " + \
                                        "respectively %d and %d" % (len(filenames), \
                                         nb_data)
    if training:
        f = gen_flip_and_rot
        shuffle_buffer_size = nb_data
        random.seed(5*(random.randint(1,nb_data)+hvd.rank()))
    else:
        f = gen_valid
    _input = f(CoverPath, StegoPath, filenames[0])
    shapes = [_i.shape for _i in _input]
    features_shape = [batch_size] + [s for s in shapes[0][1:]]
    # add color channel
    # should be of shape (2, height, width, color),
    # because we are using pair constraint
    if len(shapes[0]) < 4:
        features_shape += [1]
    labels_shape = [batch_size] + [s for s in shapes[1][1:]]
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    if not training:
        ds = ds.shard(hvd.size(), hvd.rank())
        ds = ds.take(len(filenames) // hvd.size()) # make sure all ranks have the same amount
    if training:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=5*(hvd.rank()+random.randint(0,nb_data)))
        ds = ds.repeat() # infinitely many data
    ds = ds.map(lambda filename : tf.py_func(f, [CoverPath, StegoPath, filename], [tf.uint8, tf.uint8]),
                 num_parallel_calls=num_of_threads)
    ds = ds.batch(batch_size//2) # divide by 2, because we already work with pairs and batch() adds 0-th dimension
    ds = ds.map(lambda x,y: (tf.reshape(x, features_shape), tf.reshape(y, labels_shape)), # reshape number of pairs into batch_size
                num_parallel_calls=num_of_threads).prefetch(buffer_size=num_of_threads*batch_size)
    if training:
        ds = ds.shuffle(buffer_size=num_of_threads*batch_size, seed=7*(hvd.rank()+random.randint(0,nb_data)))
    
    iterator = ds.make_one_shot_iterator()
    return iterator.get_next()