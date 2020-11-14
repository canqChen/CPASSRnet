
import numpy as np
import tensorflow as tf

from utils import cal_conv_pad, cal_deconv_pad

weight_init = tf.contrib.layers.xavier_initializer()
weight_regularizer = None


#####################################################################
# normalizations
#####################################################################

def batch_norm(x, scope="bn"):
    """batch normalization
    
    Arguments:
        x {[tensor]} -- input feature
    
    Keyword Arguments:
        scope {str} -- name of variable scope (default: {"bn"})
    
    Returns:
        [tensor] -- output feature after batch normalization
    """
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=scope)

def instance_norm(x, scope='in'):
    """instance normalization
    
    Arguments:
        x {[tensor]} -- input feature
    
    Keyword Arguments:
        scope {str} -- name of variable scope (default: {"in"})
    
    Returns:
        [tensor] -- output feature after instance normalization
    """
    return tf.contrib.layers.instance_norm(x, epsilon=1e-05, center=True, scale=True)

#####################################################################
# convolution blocks
#####################################################################

def conv_na(x, out_channels, ks=3, s=1, pad_type='reflect', sn=False, use_bias=False, norm=None, act=None, scope="conv2d"):
    """convolutional block
    
    Arguments:
        x {[tensor]} -- input feature
        out_channels {[int]} -- number of convoluted output channel
    
    Keyword Arguments:
        ks {int} -- convoluted kernel size (default: {3})
        s {int} -- convoluted strides (default: {1})
        pad_type {str} -- padding type, options: 'zero', 'reflect' (default: {'reflect'})
        sn {bool} -- use spectral normalization or not (default: {False})
        use_bias {bool} -- use bias or not (default: {False})
        norm {[str]} -- normalization type, options: 'instance', 'batch' (default: {None})
        act {[str]} -- activation type, options: 'relu', 'lrelu','sigmoid', 'tanh' (default: {None})
        scope {str} -- variable scope (default: {"conv2d"})
    
    Raises:
        TypeError: wrong norm name
        TypeError: wrong activation name
    
    Returns:
        [tensor] -- output features
    """
    b,h,w,c = x.get_shape().as_list()
    p_top, p_down, p_left, p_right = cal_conv_pad(h,w,s,ks)
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [p_top, p_down], [p_left, p_right], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [p_top, p_down], [p_left, p_right], [0, 0]], mode='REFLECT')
            
        out = 0
        if sn:
            weight = tf.get_variable('kernel', shape=[ks, ks, x.get_shape().as_list()[-1], out_channels])
            if use_bias:
                bias = tf.get_variable('bias', shape=[out_channels], initializer=tf.constant_initializer(0.0))
                out = tf.nn.conv2d(x, filter=spectral_norm(weight), strides=[1,s,s,1], padding='VALID') + bias
            else:
                out = tf.nn.conv2d(x, filter=spectral_norm(weight), strides=[1,s,s,1], padding='VALID')
        else:
            out = tf.layers.conv2d(x, out_channels, ks, strides=s, kernel_initializer=weight_init,padding='VALID',
                                    kernel_regularizer=weight_regularizer, use_bias=use_bias)
        if norm == 'instance':
            out = instance_norm(out)
        elif norm == 'batch':
            out = batch_norm(out)
        elif norm is not None:
            raise TypeError("No normalization: %s"%norm)

        if act == 'relu':
            out = relu(out)
        elif act == 'lrelu':
            out = lrelu(out)
        elif act== 'sigmoid':
            out = sigmoid(out)
        elif act == 'tanh':
            out = tanh(out)
        elif act == 'mish':
            out = mish(out)
        elif act is not None:
            raise TypeError("No activation: %s"%norm)
        
        return out

#####################################################################
# fully connected layer
#####################################################################

def fully_connected_na(input_, output_size, scope='fc', norm=None, act=None, reuse=False, stddev=0.02, bias_init_value=0.0, with_w=False):
    """fully connected layer
    
    Arguments:
        input_ {tensor} -- input feature with shape [B,N] or [B,H,W,C]
        output_size {integer} -- number of output node of the fully connected layer
    
    Keyword Arguments:
        scope {str} -- name of variable scope (default: {'fc'})
        reuse {bool} -- reuse the variable scope or not (default: {False})
        norm {str} -- normalization operation after weighted multiplication, options: 'instance', 'batch' (default: {None})
        act {str} -- activation after normalization, options: 'relu', 'lrelu', 'sigmoid', 'tanh' (default: {None})
        stddev {float} -- weight initialized standard deviation (default: {0.02})
        bias_init_value {float} -- bias initialized value (default: {0.0})
        with_w {bool} -- return the parameters (weight and bias) or not (default: {False})
    
    Returns:
        tensor -- shape of [B,output_size]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # if reuse:
        #     tf.get_variable_scope().reuse_variables()
        # else:
        #     assert tf.get_variable_scope().reuse is False

        if len(input_.get_shape().as_list()) > 2:
            input_ = flatten(input_)

        with tf.variable_scope(scope or "Linear"):
            matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                    tf.random_normal_initializer(stddev=stddev))
            bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_init_value))

            out = tf.matmul(input_, matrix) + bias

            if norm == 'instance':
                out = instance_norm(out)
            elif norm == 'batch':
                out = batch_norm(out)
            elif norm is not None:
                raise TypeError("No normalization: %s"%norm)

            if act == 'relu':
                out = relu(out)
            elif act == 'lrelu':
                out = lrelu(out)
            elif act== 'sigmoid':
                out = sigmoid(out)
            elif act == 'tanh':
                out = tanh(out)
            elif act is not None:
                raise TypeError("No activation: %s"%norm)

            if with_w:
                return out,matrix,bias
            else:
                return out
            

#####################################################################
# residual-related blocks
#####################################################################

# Residual-block v1
def resblock_V1(x, out_channels, ks=3, s=1, pad_type='reflect', sn=False, use_bias=False, norm=None, act=None, scope='resblock'):
    """residual block version 1:

    input----conv----norm----act----conv----+----output
          |_________________________________|
    
    Arguments:
        x {[tensor]} -- input feature
        out_channels {[int]} -- number of convoluted output channel
    
    Keyword Arguments:
        ks {int} -- convoluted kernel size (default: {3})
        s {int} -- convoluted strides (default: {1})
        pad_type {str} -- padding type, options: 'zero', 'reflect' (default: {'reflect'})
        sn {bool} -- use spectral normalization or not (default: {False})
        use_bias {bool} -- use bias or not (default: {False})
        norm {[str]} -- normalization type, options: 'instance', 'batch' (default: {None})
        act {[str]} -- activation type, options: 'relu', 'lrelu','sigmoid', 'tanh' (default: {None})
        scope {str} -- variable scope (default: {"resblock"})
    
    Returns:
        [tensor] -- output feature
    """
    _,_,_,c = x.get_shape().as_list()
    with tf.variable_scope(scope):
        y = conv_na(x, out_channels, ks=ks, s=s, pad_type=pad_type, sn=sn, use_bias=use_bias, norm=norm, act=act, scope='conv1')
        y = conv_na(y, out_channels, ks=ks, s=s, pad_type=pad_type, sn=sn, use_bias=use_bias, norm=norm, act=None, scope='conv2')
        if c==out_channels:
            return relu(x + y)
        else:
            return relu(y + conv_na(x, out_channels, ks=1, s=1, pad_type=pad_type, sn=sn, use_bias=None, norm=None, act=None, scope='rescale_conv'))

#####################################################################
# attention
#####################################################################

def CPAM(view, view_star, scope='dpeth_wise_attention'):
    b,h,w,nc = view.get_shape().as_list()
    with tf.variable_scope(scope):
        cat_feat = tf.concat([view, view_star],axis=-1)
        mask = conv_na(cat_feat, 1, ks=7, s=1,  act='relu', scope='attent_conv')
        attent_feat = mask*view_star
        beta = tf.get_variable('beta', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(value=0))
        output = beta*attent_feat + view
        return output

#####################################################################
# sampling blocks
#####################################################################

def max_pool(x, ps=2, s=2, padding='SAME', scope='maxpool'):
    return tf.nn.max_pool(x, ksize=[1, ps, ps, 1], strides=[1, s, s, 1], padding=padding, name=scope)

def upsample_conv_na(x, output_channels, ks=3, s=1, mode='bilinear', use_bias = False, scale_factor=2, out_h=None, out_w=None, sn=False, norm=None, act=None, scope='upsp_conv'):
    """Perform nearest upsample before convolution to avoid checkerboard artifacts. Usually be used for replacing deconv(transposed convolution).
    
    Arguments:
        x {[tensor]} -- input feature
        output_channels {[type]} -- [description]
    
    Keyword Arguments:
        ks {int} -- [description] (default: {3})
        s {int} -- [description] (default: {1})
        use_bias {bool} -- [description] (default: {False})
        scale_factor {int} -- [description] (default: {2})
        sn {bool} -- [description] (default: {False})
        norm {[type]} -- [description] (default: {None})
        act {[type]} -- [description] (default: {None})
        scope {str} -- [description] (default: {'upsp_conv'})
    
    Returns:
        [tensor] -- output feature
    """
    with tf.variable_scope(scope):
        _, h, w, _ = x.get_shape().as_list()
        new_size = [h * scale_factor, w * scale_factor] if out_h==None else [out_h, out_w]
        x = tf.image.resize_nearest_neighbor(x, size=new_size) if mode=='nearest' else tf.image.resize_bilinear(x, size=new_size)
        return conv_na(x, output_channels, ks=ks, s=1, use_bias=use_bias, sn=sn, norm=norm, act=act)


#####################################################################
# activation functions
#####################################################################
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

def relu(x):
    return tf.nn.relu(x, name='relu')

def tanh(x):
    return tf.tanh(x, name='tanh')

def sigmoid(x):
    return tf.nn.sigmoid(x, name='sigmoid')

def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))


#####################################################################
# loss functions
#####################################################################

def mae_loss(pred, target):
    """Mean absolute error: 
    error = mean(abs(in_ - target))
    
    Arguments:
        pred {[tensor]} -- predicted feature
        target {[tensor]} -- target feature, must have the same shape as the pred
    
    Returns:
        [type] -- [description]
    """
    return tf.reduce_mean(tf.abs(pred - target))

def mse_loss(pred, target):
    """Mean square error: 
    error = mean((in_ - target)**2)
    
    Arguments:
        pred {[tensor]} -- predicted feature
        target {[tensor]} -- target feature, must have the same shape as the pred
    
    Returns:
        [type] -- [description]
    """
    return tf.reduce_mean((pred-target)**2)

def bce_loss(logits, labels):
    """Binary cross entropy error: 
    
    Arguments:
        logits {[tensor]} -- predicted logits
        labels {[tensor]} -- labels, must have the same shape as the logits
    
    Returns:
        [type] -- [description]
    """
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def perceptual_loss(recon_feats=[], real_feats=[]):
    """Calculate perceptual loss
    
    Keyword Arguments:
        recon_feats {list} -- list contains several reconstructed features (default: {[]})
        real_feats {list} -- list contains several real img features (default: {[]})
    
    Returns:
        [scalar] -- value of loss
    """
    n_feat = len(real_feats)
    loss = []
    for i in range(n_feat):
        loss.append(tf.reduce_mean(tf.abs(real_feats[i]-recon_feats[i])))
    return sum(loss) / len(loss)