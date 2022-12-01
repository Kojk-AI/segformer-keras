import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import cv2

def overlap_patch_merging_module(input, num_filters, kernel_size, stride):
    output = layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=stride, padding='same', use_bias=False)(input)
    output = layers.LayerNormalization()(output)
    return output


def reducer_module(input, dim_factor):
    b, h, w, c = input.shape
    reduction_factor = dim_factor * dim_factor
    output = layers.Reshape((h*w//reduction_factor, c*reduction_factor))(input)
    output = layers.Dense(c)(output)
    output = layers.Reshape((h//dim_factor, w//dim_factor, c))(output)
    return output

def cross_attention_module(query, key, value, emb_size, num_heads, i):
    # Generate cross-attention outputs: [batch_size, latent_dim, projection_dim].
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=emb_size//num_heads)(
        query, key, value
        )
    
    #Layer norm and skip connection
    attention_output = layers.LayerNormalization(
        epsilon=1e-6
    )(query + attention_output)

    return attention_output

def mixmlp_module(input, expansion_factor):
    b, h, w, c = input.shape
    output = layers.Conv2D(c, kernel_size=1)(input)
    output = layers.DepthwiseConv2D(kernel_size=3, padding="same", depth_multiplier=expansion_factor, activation=tf.keras.activations.gelu)(output)
    output = layers.Conv2D(c, kernel_size=1)(output)
    return output

def encoder_stage(input, kernel_size, stride, emb_size, reduction_ratio, mlp_expansion, num_heads, depth):

    input = overlap_patch_merging_module(input, emb_size, kernel_size, stride)
    input = layers.LayerNormalization()(input)
    #Encoder block
    for i in range(depth):
        #Attention
        reduced_input = reducer_module(input, reduction_ratio)
        input = cross_attention_module(input, reduced_input, reduced_input, emb_size, num_heads, i)

        #Mix MLP
        residual = mixmlp_module(input, mlp_expansion)
        input = tfa.layers.StochasticDepth()([input, residual])
        input = layers.LayerNormalization()(input)

    return input

def decoder_stage(input, decoder_channels, scale_factor):

    output = layers.Conv2D(decoder_channels, kernel_size=1)(input)
    output = layers.UpSampling2D(scale_factor, interpolation='bilinear')(output)

    return output

def segformer(num_classes, kernel_sizes, strides, emb_sizes, reduction_ratios, mlp_expansions, num_heads, depths, decoder_channels, scale_factors, input_shape):

    outputs = []

    inputs=layers.Input(input_shape)
    input = inputs

    for kernel_size, stride, emb_size, reduction_ratio, mlp_expansion, num_head, depth, scale_factor in zip(kernel_sizes, strides, emb_sizes, reduction_ratios, mlp_expansions, num_heads, depths, scale_factors):
        input = encoder_stage(input, kernel_size, stride, emb_size, reduction_ratio, mlp_expansion, num_head, depth)
        output = decoder_stage(input, decoder_channels, scale_factor)
        outputs.append(output)

    # print(outputs)
    output = layers.Concatenate()(outputs)
    output = layers.Conv2D(decoder_channels, kernel_size=1, use_bias=False, activation='relu')(output)
    output = layers.BatchNormalization()(output)

    output = layers.Conv2D(num_classes, kernel_size=1, activation='softmax')(output)

    return keras.Model(inputs=inputs, outputs=output)