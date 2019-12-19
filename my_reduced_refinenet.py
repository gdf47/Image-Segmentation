from keras.layers import Input
from keras.models import Model
from keras.layers import Add, Lambda, SeparableConv2D, Concatenate, Dense
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
import keras.backend as K
import tensorflow as tf
from bolt.utils.model_building import load_keras_model, get_building_tools

def ResidualConvUnit(inputs,n_filters=16,kernel_size=16,name=''):

    net= Activation("relu")(inputs)
    net = Conv2D(n_filters, kernel_size, padding='same')(net)
    net = Activation("relu")(net)
    net = Conv2D(n_filters, kernel_size, padding='same')(net)
    net = Add()([net, inputs])

    return net


def ChainedResidualPooling(inputs, n_filters=16, name=''):

    net =Activation("relu")(inputs)
    net_out_1 = net

    net = Conv2D(n_filters, 3, padding='same')(inputs)
    #net = BatchNormalization()(net)
    net = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', data_format='channels_last')(net)
    net_out_2 = net

    net = Conv2D(n_filters, 3, padding='same')(net)
    #net = BatchNormalization()(net)
    net = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', data_format='channels_last')(net)
    net_out_3 = net

    net = Conv2D(n_filters, 3, padding='same')(net)
    #net = BatchNormalization()(net)
    net = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', data_format='channels_last')(net)
    net_out_4 = net

    # net = Conv2D(n_filters, 3, padding='same', name=name + 'conv4', kernel_initializer=kern_init,
    #              kernel_regularizer=kern_reg)(net)
    # #net = BatchNormalization()(net)
    # net = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same', name=name + 'pool4', data_format='channels_last')(net)
    # net_out_5 = net

    net = Add()([net_out_1, net_out_2, net_out_3, net_out_4])

    return net


def MultiResolutionFusion(high_inputs=None, low_inputs=None, n_filters=16, name=''):

    if low_inputs is None:  # RefineNet block 4
        return high_inputs

    else:
        conv_low = Conv2D(n_filters, 3, padding='same', activation= "relu")(low_inputs)
        #conv_low = BatchNormalization()(conv_low)
        conv_high = Conv2D(n_filters, 3, padding='same', activation= "relu")(high_inputs)
        #conv_high = BatchNormalization()(conv_high)
        conv_low_up = Lambda(lambda x: tf.image.resize_bilinear(x, (2 * K.int_shape(conv_low)[1], 2 * K.int_shape(conv_low)[2])))(conv_low) #UpSampling2D(size=2, interpolation='bilinear', name=name+'up')
        #conv_low_up = Lambda(lambda x: tf.image.resize_bilinear(x, (2 * K.int_shape(conv_high)[1], 2 * K.int_shape(conv_high)[2])))(conv_low)

    return Add()([conv_low_up, conv_high])


def IFS_Refinenet_4CLS(n_classes=4, load_model=None):
    net = {}
    net["input"] = Input(shape=(128, 192, 3), name="image")
    net["input_64x96"] = Input(shape=(64, 96, 3), name="image_half")
    net["input_32x48"] = Input(shape=(32, 48, 3), name="image_quart")
    net["input_16x24"] = Input(shape=(16, 24, 3), name="image_eighth")

    net["conv1_1"]= Conv2D(filters=int(16), kernel_size=1, activation="relu", padding="same", name="conv2d_1_128x192")(net["input"])
    net["conv1_2"] = Conv2D(filters=int(16), kernel_size=1, activation="relu", padding="same", name="conv2d_2_64x96")(net["input_64x96"])
    net["conv1_3"] = Conv2D(filters=int(16), kernel_size=1, activation="relu", padding="same", name="conv2d_3_32x48")(net["input_32x48"])
    net["conv1_4"] = Conv2D(filters=int(32), kernel_size=1, activation="relu", padding="same", name="conv2d_4_16x24")(net["input_16x24"])

    rcu_high = ResidualConvUnit(net["conv1_4"], n_filters=32, name='rb_{}_rcu_h1_128x192')
    rcu_high = ResidualConvUnit(rcu_high, n_filters=32, name='rb_{}_rcu_h2_128x192')
    fuse_128x192 = MultiResolutionFusion(high_inputs=rcu_high, low_inputs=None,  n_filters=32, name='rb_{}_mrf_128x192')
    fuse_pooling_128x192 = ChainedResidualPooling(fuse_128x192, n_filters=32, name='rb_{}_crp_128x192')
    low_0 = ResidualConvUnit(fuse_pooling_128x192, n_filters=32, name='rb_{}_rcu_o1_128x192')


    rcu_high = ResidualConvUnit(net["conv1_3"], n_filters=16, name='rb_{}_rcu_h1_64x96')
    rcu_high = ResidualConvUnit(rcu_high, n_filters=16, name='rb_{}_rcu_h2_64x96')
    rcu_low = ResidualConvUnit(low_0, n_filters=32, name='rb_{}_rcu_l1_64x96')
    rcu_low = ResidualConvUnit(rcu_low, n_filters=32, name='rb_{}_rcu_l2_64x96')
    fuse = MultiResolutionFusion(high_inputs=rcu_high, low_inputs=rcu_low, n_filters=16, name='rb_{}_mrf_64x96')
    fuse_pooling = ChainedResidualPooling(fuse, n_filters=16, name='rb_{}_crp_64x96')
    low_1 = ResidualConvUnit(fuse_pooling, n_filters=16, name='rb_{}_64x96')

    rcu_high = ResidualConvUnit(net["conv1_2"], n_filters=16, name='rb_{}_rcu_h1_32x48')
    rcu_high = ResidualConvUnit(rcu_high, n_filters=16, name='rb_{}_rcu_h2_32x48')
    rcu_low = ResidualConvUnit(low_1, n_filters=16, name='rb_{}_rcu_l1_32x48')
    rcu_low = ResidualConvUnit(rcu_low, n_filters=16, name='rb_{}_rcu_l2_32x48')
    fuse = MultiResolutionFusion(high_inputs=rcu_high, low_inputs=rcu_low, n_filters=16, name='rb_{}_mrf_32x48')
    fuse_pooling = ChainedResidualPooling(fuse, n_filters=16, name='rb_{}_crp_32x48')
    low_2 = ResidualConvUnit(fuse_pooling, n_filters=16, name='rb_{}_32x48')

    rcu_high = ResidualConvUnit(net["conv1_1"], n_filters=16, name='rb_{}_rcu_h1_16x24')
    rcu_high = ResidualConvUnit(rcu_high, n_filters=16, name='rb_{}_rcu_h2_16x24')
    rcu_low = ResidualConvUnit(low_2, n_filters=16, name='rb_{}_rcu_l1_16x24')
    rcu_low = ResidualConvUnit(rcu_low, n_filters=16, name='rb_{}_rcu_l2_16x24')
    fuse = MultiResolutionFusion(high_inputs=rcu_high, low_inputs=rcu_low, n_filters=16, name='rb_{}_mrf_16x24')
    fuse_pooling = ChainedResidualPooling(fuse, n_filters=16, name='rb_{}_crp_16x24')
    low_3 = ResidualConvUnit(fuse_pooling, n_filters=16, name='rb_{}_16x24')

    netz = ResidualConvUnit(low_3, n_filters=16, name='rf_rcu_o1_')
    netz = ResidualConvUnit(netz, n_filters=16, name='rf_rcu_o2_')
    net["last_conv"] = Conv2D(filters=int(n_classes), kernel_size=1, activation="relu", padding="same")(netz)
    #net["upsample"] = Lambda(lambda x: tf.image.resize_bilinear(x, (128, 192)), name='rf_up_o')(net)

    cls_shape = (n_classes, -1) if K.image_dim_ordering() == "th" else (-1, n_classes)
    net["act"] = Activation("softmax")(net["last_conv"])
    net["predictions"] = Reshape(cls_shape, name="ground_truth")(net["act"])

    model = Model(input=net["input"], output=[net["predictions"]])

    if load_model is not None:
        load_keras_model(model, load_model)

    model.summary()

    return model
