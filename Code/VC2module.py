import tensorflow as tf

def conv2d_layer(inputs,filters,kernel_size,strides,padding = 'same',activation = None,
    kernel_initializer = None,name = None):

    conv_layer = tf.layers.conv2d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return conv_layer

def gated_linear_layer(inputs, gates, name = None):

    activation = tf.multiply(x = inputs, y = tf.sigmoid(gates), name = name)

    return activation

def instance_norm_layer(inputs, epsilon = 1e-06, activation_fn = None, name = None):

    instance_norm_layer = tf.contrib.layers.instance_norm(
        inputs = inputs,
        epsilon = epsilon,
        activation_fn = activation_fn)

    return instance_norm_layer

def pixel_shuffler(inputs, shuffle_size = 2, name = None):

    n = tf.shape(inputs)[0]
    w = tf.shape(inputs)[1]
    c = inputs.get_shape().as_list()[2]

    oc = c // shuffle_size
    ow = w * shuffle_size

    outputs = tf.reshape(tensor = inputs, shape = [n, ow, oc], name = name)

    return outputs

def downsample2d_block(
    inputs,
    filters,
    kernel_size,
    strides,
    name_prefix = 'downsample2d_block_'):

    h1 = conv2d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv2d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

def residual1d_block(
    inputs,
    filters = 512,
    kernel_size = 3,
    strides = 1,
    name_prefix = 'residule_block_'):

    h1 = tf.layers.conv1d(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')
    h2 = tf.layers.conv1d(inputs = h1_glu, filters = filters // 2, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h2_conv')
    h2_norm = instance_norm_layer(inputs = h2, activation_fn = None, name = name_prefix + 'h2_norm')

    h3 = inputs + h2_norm

    return h3

def upsample2d_block(
    inputs,
    filters,
    kernel_size,
    strides,
    shuffle_size = 2,
    name_prefix = 'upsample1d_block_'):

    h1 = conv2d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_shuffle = pixel_shuffler(inputs = h1, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle')
    h1_norm = instance_norm_layer(inputs = h1_shuffle, activation_fn = None, name = name_prefix + 'h1_norm')

    h1_gates = conv2d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_shuffle_gates = pixel_shuffler(inputs = h1_gates, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_shuffle_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu


def generator(inputs, reuse = False, scope_name = 'generator'):

    #Expand the dimension for input and feed into 2D conv
    inputs = tf.expand_dims(inputs, -1)

        with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False
        #Downsample 2D->1D
        #Conv k5x15,c128,s1x1
        h1 = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [5,15], strides = [1,1], activation = None, name = 'h1_conv')
        #GLU
        h1_gates = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [5,15], strides = [1,1], activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(h1,h1_gates,name='h1_glu')

        #Conv k5x5 c256 s2x2
        d1 = downsample2d_block(h1_glu,256,[5,5],[2,2])
        #Conv k5x5 c512 s2x2
        d2 = downsample2d_block(d1,512,[5,5],[2,2])

        #Shape 2D->1D
        w_new = tf.shape(inputs)[2]/4
        bs = tf.shape(inpus)[0]
        reshape_layer = tf.reshape(d2,[bs,w_new,2304],name='Reshape_layer1')

        #1X1 Conv
        h2 = tf.layers.conv1d(reshape_layer,filters=256,kernel_size=1,strides=1,padding='same',name='h2_cov')

        # Six Residual blocks
        r1 = residual1d_block(h2)
        r2 = residual1d_block(r1)
        r3 = residual1d_block(r2)
        r4 = residual1d_block(r3)
        r5 = residual1d_block(r4)
        r6 = residual1d_block(r5)

        # Conv 1x1
        h3 = tf.layers.conv1d(r6,filters=2304,kernel_size=1,strides=1,padding='same')
        # Instance Norm
        h3_norm = tf.contrib.layers.instance_norm(h3)

        #Reshape 1D->2D
        reshape_layer2 = tf.reshape(h3_norm,[bs,256,w_new,9])

        # upsample
        u1 = upsample2d_block(reshape_layer2,filters=1024,kernel_size=[5,5],strides=[1,1])
        u2 = upsample2d_block(u1,filters=512,kernel_size=[5,5],strides=[1,1])

        # Conv k 5x5 c512 s1x1
        o1 = tf.layers.conv2d(u2,filters=35,kernel_size=[5,15],strides=[1,1],padding='same')

        return o1


def discriminator(inputs, reuse = False, scope_name = 'discriminator'):

    # inputs has shape [batch_size, num_features, time]
    # we need to add channel for 2D convolution [batch_size, num_features, time, 1]
    inputs = tf.expand_dims(inputs, -1)

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        # Conv k3x3 c256 s2x2
        h1 = tf.layers.conv2d(inputs,filters=128,kernel_size=[3,3],strides=[1,1],padding='same')

        # GLU
        h1_gates = tf.layers.conv2d(inputs,filters=128,kernel_size=[3,3],strides=[1,1],padding='same')
        h1_glu = gated_linear_layer(h1,h1_gates)

        # Four downsample layers
        d1 = downsample2d_block(h1_glu,filters = 256,kernel_size = [3, 3],strides = [2,2])
        d2 = downsample2d_block(d1,filters = 512,kernel_size = [3,3],strides = [2,2])
        d3 = downsample2d_block(d2,filters = 1024,kernel_size = [3,3],strides = [2,2])
        d4 = downsample2d_block(d3,filters = 1024,kernel_size = [1,5],strides = [1,1])

        # Conv k1x3 c1 s1x1
        o1 = tf.layers.conv2d(d4,filters=1,kernel_size=[1,3],strides=[1,1])
        //# TODO:
        o2
        return o2
