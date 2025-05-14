import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D # type: ignore
from tensorflow.keras import regularizers, optimizers # type: ignore
from tensorflow.keras import backend as K # type: ignore

import numpy as np
import pandas as pd
    #system
import os
import sys
import csv

import math
import random
import matplotlib
import matplotlib.pyplot as plt

def get_backbone(input_shape, 
                 backbone_name='custom', 
                 backbone_net_weights='imagenet',
                 num_blocks=4, initial_filters=64, filter_increment=2,
                 **kwargs):
    """
    Function to select backbone architecture with customizable number of blocks and filter increments.
    Arguments:
        - input_tensor: Input tensor for the backbone model.
        - backbone_name: The name of the backbone model to use.
        - num_blocks: The number of convolutional blocks (default is 4).
        - initial_filters: The number of filters for the first block (default is 64).
        - filter_increment: The multiplier to increase the filters after each block (default is 2).
    Returns:
        - x: Output tensor from the backbone.
    """
    inputs = keras.Input(shape=input_shape, name='input_layer')

    # Check if the input image needs resizing or RGB conversion
    x = inputs
    if input_shape[0] < 32 or input_shape[1] < 32 or input_shape[2] != 3:
        # Resize the input if the dimensions are less than 32x32
        print("adding resize layer")
        x = keras.layers.Lambda(lambda x: tf.image.resize(x, [32, 32]), name='Img_resize')(x)
        
        # Convert to RGB if the input is grayscale (i.e., channels != 3)
        if input_shape[2] != 3:
            print("adding grayscale to rgb layer")
            x = keras.layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x), name='Img_grayscale_to_rgb')(x)

    # Create the model using the input and the final output
    base_input = keras.Model(inputs=inputs, outputs=x, name='base_input')


    # Custom backbone architecture
    if backbone_name == 'custom':
        x = base_input.output  # Start with the input tensor

        # Loop over the number of blocks
        for i in range(num_blocks):
            # Calculate the number of filters for the current block
            filters = initial_filters * (filter_increment ** i)

            # Block name prefix for each block
            block_name = f'block{i + 1}'

            # First Conv2D + BatchNormalization
            x = keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv1')(x)
            x = keras.layers.BatchNormalization()(x)

            # Second Conv2D + BatchNormalization
            x = keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same', name=f'{block_name}_conv2')(x)
            x = keras.layers.BatchNormalization()(x)

            # MaxPooling layer
            x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name=f'{block_name}_pool')(x)

        # Define the final model
        model = keras.Model(inputs=base_input.input, outputs=x, name='custom_backbone')

    elif backbone_name in [model for model in dir(keras.applications) if callable(getattr(keras.applications, model))]:
        backbone = getattr(keras.applications, backbone_name)
        model = backbone(include_top=False, 
                         input_tensor=base_input.output,
                         weights=backbone_net_weights
                         )
    else:
        raise ValueError("Unknown backbone architecture specified.")
    
    return model

def squash(s, axis=-1, name="squash"):
    """
    non-linear squashing function to manipulate the length of the capsule vectors
    :param s: input tensor containing capsule vectors
    :param axis: If `axis` is a Python integer, the input is considered a batch of vectors,
      and `axis` determines the axis in `tensor` over which to compute squash.
    :return: a Tensor with same shape as input vectors
    """
    with tf.name_scope(name):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                         keepdims=True)
        safe_norm = tf.sqrt(squared_norm + keras.backend.epsilon())
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
        
def safe_norm(s, axis=-1, keepdims=False, name="safe_norm"):
    """
    Safe computation of vector 2-norm
    :param s: input tensor
    :param axis: If `axis` is a Python integer, the input is considered a batch 
      of vectors, and `axis` determines the axis in `tensor` over which to 
      compute vector norms.
    :param keepdims: If True, the axis indicated in `axis` are kept with size 1.
      Otherwise, the dimensions in `axis` are removed from the output shape.
    :param name: The name of the op.
    :return: A `Tensor` of the same type as tensor, containing the vector norms. 
      If `keepdims` is True then the rank of output is equal to
      the rank of `tensor`. If `axis` is an integer, the rank of `output` is 
      one less than the rank of `tensor`.
    """
    with tf.name_scope(name):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                 keepdims=keepdims)
        return tf.sqrt(squared_norm + keras.backend.epsilon())    
        
class SecondaryCapsule(keras.layers.Layer):
    """
    The Secondary Capsule layer With Dynamic Routing Algorithm. 
    input shape = [None, input_num_capsule, input_dim_capsule] 
    output shape = [None, num_capsule, dim_capsule]
    :param n_caps: number of capsules in this layer
    :param n_dims: dimension of the output vectors of the capsules in this layer
    """
    def __init__(self, n_caps, n_dims, routings=2, **kwargs):
        super().__init__(**kwargs)
        self.n_caps = n_caps
        self.n_dims = n_dims
        self.routings = routings
    def build(self, batch_input_shape):
        # transformation matrix
        self.W = self.add_weight(
            name="W", 
            shape=(1, batch_input_shape[1], self.n_caps, self.n_dims, batch_input_shape[2]),
            initializer=keras.initializers.RandomNormal(stddev=0.1))
        super().build(batch_input_shape)
    def call(self, X):
        # predict output vector
        batch_size = tf.shape(X)[0]
        caps1_n_caps = tf.shape(X)[1] 
        W_tiled = tf.tile(self.W, [batch_size, 1, 1, 1, 1])
        caps1_output_expanded = tf.expand_dims(X, -1, name="caps1_output_expanded")
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_tile")
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, self.n_caps, 1, 1], name="caps1_output_tiled")
        caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")
        
        # routing by agreement
        # routing weights
        raw_weights = tf.zeros([batch_size, caps1_n_caps, self.n_caps, 1, 1],
                       dtype=np.float32, name="raw_weights")
        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            routing_weights = tf.nn.softmax(raw_weights, axis=2, name="routing_weights")
            weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name="weighted_predictions")
            weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True, name="weighted_sum")
            caps2_output_round_1 = squash(weighted_sum, axis=-2, name="caps2_output_round_")
            caps2_output_squeezed = tf.squeeze(caps2_output_round_1, axis=[1,4], name="caps2_output_squeezed")
            if i < self.routings - 1:
                caps2_output_round_1_tiled = tf.tile(
                                        caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
                                        name="caps2_output_tiled_round_")
                agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                      transpose_a=True, name="agreement")
                raw_weights_round_2 = tf.add(raw_weights, agreement,
                             name="raw_weights_round_")
                raw_weights = raw_weights_round_2
        return caps2_output_squeezed
    def compute_output_shape(self, batch_input_shape):
        return (batch_input_shape[0], self.n_caps, self.n_dims)
    def get_config(self): ### custom layers to be serializable as part of a Functional model
        base_config = super().get_config()
        return {**base_config, 
                "n_caps": self.n_caps, 
                "n_dims": self.n_dims,
                "routings": self.routings}
    
class ML_Class_Capsule(keras.layers.Layer):
    """
    The Class Capsule layer With Dynamic Routing Algorithm. 
    input shape = [None, input_num_capsule, input_dim_capsule] 
    output shape = [None, num_capsule, dim_capsule]
    :param n_caps: number of capsules in this layer
    :param n_dims: dimension of the output vectors of the capsules in this layer
    """
    def __init__(self, n_caps_lvl:list, n_dims:list, routings=2, **kwargs):
        super().__init__(**kwargs)
        self.n_caps_lvl = n_caps_lvl
        self.n_dims = n_dims
        self.routings = routings

    def build(self, batch_input_shape):
        # transformation matrix
        self.W = [self.add_weight(
                    name=f"W_LVL_{k}", 
                    shape=(1, batch_input_shape[1], v, self.n_dims[k], batch_input_shape[2]),
                    initializer=keras.initializers.RandomNormal(stddev=0.1), trainable=True
            ) for k,v in enumerate(self.n_caps_lvl)]
        
        super().build(batch_input_shape)

    def call(self, X):
        # predict output vector
        batch_size = tf.shape(X)[0]
        caps1_n_caps = tf.shape(X)[1] 
        W_tiled = [tf.tile(W, [batch_size, 1, 1, 1, 1]) for W in self.W]
        caps1_output_expanded = tf.expand_dims(X, -1, name="caps1_output_expanded")
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_tile")
        caps1_output_tiled = [tf.tile(caps1_output_tile, [1, 1, n_caps, 1, 1], name=f"caps1_output_tiled_LVL_{k}") for k,n_caps in enumerate(self.n_caps_lvl)]
        caps2_predicted = [tf.matmul(W_tiled[k], caps1_output_tiled[k], name=f"caps2_predicted_LVL_{k}") for k,W in enumerate(W_tiled)]
        
        # routing by agreement
            # routing weights
        raw_weights = [tf.zeros([batch_size, caps1_n_caps, n_caps, 1, 1], dtype=np.float32, name=f"raw_weights_LVL_{k}") for k,n_caps in enumerate(self.n_caps_lvl)]

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):

            routing_weights = [tf.nn.softmax(raw_weights[k], axis=2, name=f"routing_weights_LVL_{k}") for k in range(len(raw_weights))]
            weighted_predictions = [tf.multiply(routing_weights[k], caps2_predicted[k], name=f"weighted_predictions_LVL_{k}") for k in range(len(raw_weights))]
            weighted_sum = [tf.reduce_sum(weighted_predictions[k], axis=1, keepdims=True, name=f"weighted_sum_LVL_{k}") for k in range(len(weighted_predictions))]

            caps2_output_round_1 = [squash(weighted_sum[k], axis=-2, name=f"caps2_output_round_{k}") for k in range(len(weighted_sum))]

            caps2_output_squeezed = [tf.squeeze(caps2_output_round_1[k], axis=[1,4], name=f"caps2_output_squeezed_LVL_{k}") for k in range(len(caps2_output_round_1))]

            if i < self.routings - 1:
                caps2_output_round_1_tiled = [tf.tile(
                                        caps2_output_round_1[k], [1, caps1_n_caps, 1, 1, 1],
                                        name=f"caps2_output_tiled_round_LVL_{k}") for k in range(len(caps2_output_round_1))]

                agreement = [tf.matmul(caps2_predicted[k], caps2_output_round_1_tiled[k], transpose_a=True, name=f"agreement_LVL_{k}") for k in range(len(caps2_predicted))]

                raw_weights_round_2 = [tf.add(raw_weights, agreement, name=f"raw_weights_round_LVL_{k}") for k, (raw_weights,agreement) in enumerate(zip(raw_weights,agreement))]
                raw_weights = raw_weights_round_2
        return caps2_output_squeezed
    def compute_output_shape(self, batch_input_shape):
        return (batch_input_shape[0], self.n_caps_lvl, self.n_dims)
    def get_config(self): ### custom layers to be serializable as part of a Functional model
        base_config = super().get_config()
        return {**base_config, 
                "n_caps": self.n_caps_lvl, 
                "n_dims": self.n_dims,
                "routings": self.routings}
                
class LengthLayer(keras.layers.Layer):
    """
    Compute the length of capsule vectors.
    inputs: shape=[None, num_capsule, dim_vector]
    output: shape=[None, num_capsule]
    """
    def call(self, X):
        logits = safe_norm(X, axis=-1, name="logits")
        # return tf.math.divide(logits,tf.reshape(tf.reduce_sum(logits,-1),(-1,1),name='reshape'),name='Normalizing_Probability')
        return logits
    def compute_output_shape(self, batch_input_shape): # in case the layer modifies the shape of its input, 
                                                        #you should specify here the shape transformation logic.
                                                        #This allows Keras to do automatic shape inference.
        return (batch_input_shape[0], batch_input_shape[1])
          
class MarginLoss(keras.losses.Loss):
    """
    Compute margin loss.
    y_true shape [None, n_classes] 
    y_pred shape [None, num_capsule] = [None, n_classes]
    """
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_=0.5, **kwargs):
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_
        super().__init__(**kwargs)
        
    def call(self, y_true, y_proba):
        present_error_raw = tf.square(tf.maximum(0., self.m_plus - y_proba), name="present_error_raw")
        absent_error_raw = tf.square(tf.maximum(0., y_proba - self.m_minus), name="absent_error_raw")
        L = tf.add(y_true * present_error_raw, self.lambda_ * (1.0 - y_true) * absent_error_raw,
           name="L")
        total_marginloss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
        return total_marginloss
    
    def get_config(self): ### custom layers to be serializable as part of a Functional model
        base_config = super().get_config()
        return {**base_config, 
                "m_plus": self.m_plus,
                "m_minus": self.m_minus,
                "lambda_": self.lambda_}
                
class HDR_Class_Capsule(keras.layers.Layer):
    """
    Hierarchical Dynamic Routing Capsule Layer
    Modified Class Capsule layer with Dynamic Routing Algorithm that incorporates
    hierarchical information flow.
    
    Key Features:
        - Hierarchical Information Flow: Incorporates information from previous levels
        - Adaptive Weighting: Uses learnable parameter Alpha_h to balance information sources
        - Level-wise Processing: Maintains separate routing for each hierarchical level
        - Dimension Consistency: Ensures proper dimension matching between levels
        - Dynamic Routing: Preserves the core routing-by-agreement mechanism

    Output shape = [None, num_capsule, dim_capsule]
    """
    def __init__(self, n_caps_lvl:list, n_dims:list, routings=2, **kwargs):
        super().__init__(**kwargs)
        self.n_caps_lvl = n_caps_lvl
        self.n_dims = n_dims
        self.routings = routings

    def build(self, batch_input_shape):
        # Transformation matrices for each level
        self.W = [self.add_weight(
            name=f"W_LVL_{k}", 
            shape=(1, batch_input_shape[1], v, self.n_dims[k], batch_input_shape[2]),
            initializer=keras.initializers.RandomNormal(stddev=0.1),
            trainable=True
        ) for k,v in enumerate(self.n_caps_lvl)]
        
        # Additional weights for processing hierarchical inputs (levels > 0)
        self.W_h = [self.add_weight(
            name=f"W_H_LVL_{k}", 
            shape=(1, self.n_caps_lvl[k-1], self.n_caps_lvl[k], self.n_dims[k], self.n_dims[k-1]),
            initializer=keras.initializers.RandomNormal(stddev=0.1),
            trainable=True
        ) for k in range(1, len(self.n_caps_lvl))]
        
        # Learnable alpha parameter for each level
        self.alphas = [self.add_weight(
            name=f"alpha_LVL_{k}",
            shape=(1,),
            initializer=keras.initializers.Constant(0.5),
            trainable=True
        ) for k in range(1, len(self.n_caps_lvl))]
        
        super().build(batch_input_shape)

    def call(self, X):
        batch_size = tf.shape(X)[0]
        caps1_n_caps = tf.shape(X)[1]
        level_outputs = []
        
        for h_level, n_caps in enumerate(self.n_caps_lvl):
            # Process base input
            W_tiled = tf.tile(self.W[h_level], [batch_size, 1, 1, 1, 1])
            caps1_output_expanded = tf.expand_dims(X, -1)
            caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2)
            caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, n_caps, 1, 1])
            caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled)
            
            if h_level > 0:
                # Process previous level's output
                prev_output = level_outputs[-1]  # Get previous level output
                
                # Calculate predictions from previous level
                W_h_tiled = tf.tile(self.W_h[h_level-1], [batch_size, 1, 1, 1, 1])
                
                # Reshape prev_output to match dimensions
                prev_output = tf.expand_dims(prev_output, axis=-1)  # Add dim for matmul
                prev_output = tf.expand_dims(prev_output, axis=2)   # Add dim for current caps
                
                # Generate predictions from previous level
                h_predicted = tf.matmul(W_h_tiled, prev_output)
                
                # Get the target shape from caps2_predicted
                target_shape = tf.shape(caps2_predicted)
                
                # Reshape h_predicted to match exactly
                h_predicted_shape = tf.shape(h_predicted)
                repeat_dims = [
                    1,  # batch dimension
                    target_shape[1] // h_predicted_shape[1],  # spatial dimension
                    1,  # capsule dimension
                    1,  # feature dimension
                    1   # last dimension
                ]
                
                # Ensure repeat_dims are valid
                repeat_dims = [tf.maximum(1, dim) for dim in repeat_dims]
                
                # Tile h_predicted to match caps2_predicted dimensions exactly
                h_predicted = tf.tile(h_predicted, repeat_dims)
                
                # Pad or slice if necessary to match exactly
                target_shape_static = caps2_predicted.shape
                current_shape = tf.shape(h_predicted)
                
                # Pad if necessary
                paddings = [[0, 0],  # batch dimension
                           [0, target_shape[1] - current_shape[1]],  # spatial dimension
                           [0, 0],  # capsule dimension
                           [0, 0],  # feature dimension
                           [0, 0]]  # last dimension
                
                h_predicted = tf.pad(h_predicted, paddings)
                
                # Combine predictions using learnable alpha
                alpha = tf.sigmoid(self.alphas[h_level-1])  # Ensure alpha is between 0 and 1
                caps2_predicted = alpha * caps2_predicted + (1 - alpha) * h_predicted
            
            # Initialize routing weights
            raw_weights = tf.zeros([batch_size, caps1_n_caps, n_caps, 1, 1],
                                 dtype=np.float32, name=f"raw_weights_LVL_{h_level}")
            
            # Routing by agreement
            for i in range(self.routings):
                routing_weights = tf.nn.softmax(raw_weights, axis=2)
                weighted_predictions = tf.multiply(routing_weights, caps2_predicted)
                weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True)
                caps2_output = squash(weighted_sum, axis=-2)
                
                if i < self.routings - 1:
                    caps2_output_tiled = tf.tile(caps2_output, [1, caps1_n_caps, 1, 1, 1])
                    agreement = tf.matmul(caps2_predicted, caps2_output_tiled, transpose_a=True)
                    raw_weights = tf.add(raw_weights, agreement)
            
            # Store output for this level
            level_output = tf.squeeze(caps2_output, axis=[1,4])
            level_outputs.append(level_output)
        
        return level_outputs

    def compute_output_shape(self, batch_input_shape):
        return [(batch_input_shape[0], n_caps, dim) 
                for n_caps, dim in zip(self.n_caps_lvl, self.n_dims)]
                
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 
                "n_caps_lvl": self.n_caps_lvl,
                "n_dims": self.n_dims,
                "routings": self.routings}

class HD_SecondaryCapsule(keras.layers.Layer):
    """
    Hierarchical DEEP Capsule Layer with direct tensor operations.
    Implements the same logic as HD_CapsNet but with more efficient tensor operations.
    Use Dynamic Routing Algorithm to route information between hierarchical levels.
    
    Args:
        n_caps_lvl: List of number of capsules for each hierarchical level
        n_dims: List of capsule dimensions for each hierarchical level
        routings: Number of routing iterations (default: 2)
    """
    def __init__(self, n_caps_lvl: list, n_dims: list, routings=2, **kwargs):
        super().__init__(**kwargs)
        self.n_caps_lvl = n_caps_lvl
        self.n_dims = n_dims
        self.routings = routings
        
    def build(self, batch_input_shape):
        self.input_caps = batch_input_shape[1]
        self.input_dim = batch_input_shape[2]
        
        # Create transformation matrices for each level
        self.W = []
        for level, (n_caps, n_dims) in enumerate(zip(self.n_caps_lvl, self.n_dims)):
            if level == 0:
                # First level transforms from input_dim
                W_level = self.add_weight(
                    name=f"W_level_{level}", 
                    shape=(1, batch_input_shape[1], n_caps, n_dims, self.input_dim),
                    initializer=keras.initializers.RandomNormal(stddev=0.1)
                )
            else:
                # Calculate new number of primary capsules for this level
                new_n_caps = (self.input_caps * self.input_dim) // self.n_dims[level-1]
                # Total capsules will be new_n_caps + previous level capsules
                prev_total_caps = new_n_caps + self.n_caps_lvl[level-1]
                
                W_level = self.add_weight(
                    name=f"W_level_{level}", 
                    shape=(1, prev_total_caps, n_caps, n_dims, self.n_dims[level-1]),
                    initializer=keras.initializers.RandomNormal(stddev=0.1)
                )
            self.W.append(W_level)
                
        super().build(batch_input_shape)
        
    def route_level(self, X, W, n_caps):
        """
        Performs routing by agreement for a single level.
        
        Args:
            X: Input tensor
            W: Transformation matrix for this level
            n_caps: Number of output capsules for this level
        """
        batch_size = tf.shape(X)[0]
        caps1_n_caps = tf.shape(X)[1]
        
        # Transform input capsules
        W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1])
        caps1_output_expanded = tf.expand_dims(X, -1)
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2)
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, n_caps, 1, 1])
        caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled)
        
        # Initialize routing weights
        raw_weights = tf.zeros([batch_size, caps1_n_caps, n_caps, 1, 1],
                             dtype=np.float32)
        
        # Routing iterations
        for i in range(self.routings):
            routing_weights = tf.nn.softmax(raw_weights, axis=2)
            weighted_predictions = tf.multiply(routing_weights, caps2_predicted)
            weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True)
            caps2_output_round_1 = squash(weighted_sum, axis=-2)
            
            if i < self.routings - 1:
                caps2_output_round_1_tiled = tf.tile(
                    caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1]
                )
                agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                                    transpose_a=True)
                raw_weights = tf.add(raw_weights, agreement)
                
        return tf.squeeze(caps2_output_round_1, axis=[1,4])
        
    def call(self, X):
        """
        Forward pass of the layer.
        Implements hierarchical routing with skip connections using direct tensor operations.
        
        Args:
            X: Input tensor of primary capsules
            
        Returns:
            List of capsule outputs for each hierarchical level
        """
        s_caps = []
        batch_size = tf.shape(X)[0]
        
        for level, (n_caps, n_dims) in enumerate(zip(self.n_caps_lvl, self.n_dims)):
            if level == 0:
                # First level - direct routing from primary capsules
                s_caps.append(self.route_level(X, self.W[level], n_caps))
            else:
                # Calculate new number of primary capsules
                new_n_caps = (self.input_caps * self.input_dim) // self.n_dims[level-1]
                
                # Reshape primary capsules to match current level dimension
                p_caps_lvl = tf.reshape(X, [batch_size, new_n_caps, self.n_dims[level-1]])
                
                # Concatenate with previous level output
                skip = tf.concat([p_caps_lvl, s_caps[-1]], axis=1)
                
                s_caps.append(self.route_level(skip, self.W[level], n_caps))
                
        return s_caps
        
    def compute_output_shape(self, batch_input_shape):
        """
        Computes the output shape of the layer.
        
        Returns:
            List of output shapes for each hierarchical level
        """
        return [(batch_input_shape[0], n_caps, n_dims) 
                for n_caps, n_dims in zip(self.n_caps_lvl, self.n_dims)]
                
    def get_config(self):
        """
        Gets the config of the layer for serialization.
        """
        base_config = super().get_config()
        return {**base_config,
                "n_caps_lvl": self.n_caps_lvl,
                "n_dims": self.n_dims,
                "routings": self.routings}

def ML_CapsNet(input_shape,
                num_classes: list,
                PCaps_dim: int, 
                SCaps_dim: list,
                backbone_name='custom', 
                backbone_net_weights=None,
                num_blocks=4,  # for custom backbone
                initial_filters=64,  # for custom backbone
                filter_increment=2, # for custom backbone
                routing_iterations=2,
                **kwargs):
    """
    This is a Hierarchical Deep Capsule Network (HD-CapsNet) model architecture.
    Secondary Capsules are formed by stacking from Coarse to Fine level.
    Output of the primary capsules are fed to the secondary capsules for each level.
    """
    model_name = kwargs.get('model_name', 'ML-CapsNet')
    base_model = get_backbone(input_shape, 
                              backbone_name, 
                              backbone_net_weights,
                              num_blocks,  # for custom backbone
                              initial_filters,  # for custom backbone
                              filter_increment, # for custom backbone
                              )

    x = base_model.output # Output of the backbone model (Feature Extraction)


    ### Layer: Reshape to n-D primary capsules ###
    
    reshape = keras.layers.Reshape((int((tf.reduce_prod(x.shape[1:]).numpy())/PCaps_dim), PCaps_dim), name="reshape_layer")(x)
    p_caps = keras.layers.Lambda(squash, name='p_caps')(reshape)

    ### Layer: Secondary Capsules for hierarchical levels###
    ml_caps =  ML_Class_Capsule(n_caps_lvl=num_classes, n_dims=SCaps_dim, routings=routing_iterations, name=f"ml_caps")(p_caps)
    outputs = [LengthLayer(name=f'Out_L_{i}')(LVL) for i,LVL in enumerate(ml_caps) ] # prediction for each level
   
    model = keras.Model(inputs= [base_model.input],
                        outputs= outputs,
                        name=model_name)

    ## Return Model
    return model

def HDR_CapsNet(input_shape,
                num_classes: list,
                PCaps_dim: int, 
                SCaps_dim: list,
                backbone_name='custom', 
                backbone_net_weights=None,
                num_blocks=4,  # for custom backbone
                initial_filters=64,  # for custom backbone
                filter_increment=2, # for custom backbone
                routing_iterations=2,
                **kwargs):
    """
    This is a Hierarchical Deep Capsule Network (HD-CapsNet) model architecture. Similar to ML-CapsNet but the Secondary Capsules are changed. So, it takes the output of the primary capsules and predicts the output for each class.
    So, it is similar to HD-CapsNet.
    
    """
    model_name = kwargs.get('model_name', 'DML-CapsNet')
    base_model = get_backbone(input_shape, 
                              backbone_name, 
                              backbone_net_weights,
                              num_blocks,  # for custom backbone
                              initial_filters,  # for custom backbone
                              filter_increment, # for custom backbone
                              )

    x = base_model.output # Output of the backbone model (Feature Extraction)


    ### Layer: Reshape to n-D primary capsules ###
    
    reshape = keras.layers.Reshape((int((tf.reduce_prod(x.shape[1:]).numpy())/PCaps_dim), PCaps_dim), name="reshape_layer")(x)
    p_caps = keras.layers.Lambda(squash, name='p_caps')(reshape)

    ### Layer: Secondary Capsules for hierarchical levels###
    dml_caps =  HDR_Class_Capsule(n_caps_lvl=num_classes, n_dims=SCaps_dim, routings=routing_iterations, name=f"dml_caps")(p_caps)
    outputs = [LengthLayer(name=f'Out_L_{i}')(LVL) for i,LVL in enumerate(dml_caps) ] # prediction for each level
   
    model = keras.Model(inputs= [base_model.input],
                        outputs= outputs,
                        name=model_name)

    ## Return Model
    return model

def BUH_CapsNet(input_shape,
                num_classes: list,
                PCaps_dim: int, 
                SCaps_dim: list,
                backbone_name='custom', 
                backbone_net_weights=None,
                num_blocks=4,  # for custom backbone
                initial_filters=64,  # for custom backbone
                filter_increment=2, # for custom backbone
                routing_iterations=2,
                **kwargs
                ): ## Name should be changed to BUH-CapsNet
    """
    This is a Bottom-up Hierarchical Deep Capsule Network (BHD-CapsNet) model architecture.
    Secondary Capsules are formed by stacking from Fine to Coarse level.
    Output of the primary capsules are fed to the secondary capsules of the last hierarchy.
    The output of the secondary capsules are fed to the secondary capsules of the previous hierarchy.
    """
    
    model_name = kwargs.get('model_name', 'BUH-CapsNet')
    base_model = get_backbone(input_shape, 
                              backbone_name, 
                              backbone_net_weights,
                              num_blocks,  # for custom backbone
                              initial_filters,  # for custom backbone
                              filter_increment, # for custom backbone
                              )

    x = base_model.output # Output of the backbone model (Feature Extraction)


    ### Layer-3: Reshape to 8D primary capsules ###
    
    reshape = keras.layers.Reshape((int((tf.reduce_prod(x.shape[1:]).numpy())/PCaps_dim), PCaps_dim), name="reshape_layer")(x)
    p_caps = keras.layers.Lambda(squash, name='p_caps')(reshape)

    ### Layer-4: Secondary Capsules for hierarchical levels### 
    s_caps = [] # fine to coarse level secondary capsules
    pred = [] # prediction for each level
    # Reverse loop over the indices of SCaps_dim
    for i in [i for i, j in enumerate(SCaps_dim)][::-1]: # Reverse loop over the indices of SCaps_dim (BUH-CapsNet is bottom-up)
        if len(SCaps_dim) - 1 == i:
            # For the last element, apply to p_caps
            s_caps.append(SecondaryCapsule(n_caps=num_classes[i], n_dims=SCaps_dim[i], routings=routing_iterations, name=f"s_caps_l_{i}")(p_caps))
        else:
            # For all other elements, apply to the last element of s_caps
            s_caps.append(SecondaryCapsule(n_caps=num_classes[i], n_dims=SCaps_dim[i], routings=routing_iterations, name=f"s_caps_l_{i}")(s_caps[-1]))
        
        # Append the prediction from the LengthLayer for this level
        pred.append(LengthLayer(name=f'Out_L_{i}')(s_caps[-1]))

    pred = pred[::-1] # Reverse the prediction list to get the correct order coarse to fine
    
    ### Building Keras Model ###
    model = keras.Model(inputs= [base_model.input],
                        outputs= pred,
                        name=model_name)    
    return model

def HD_CapsNet(input_shape,
                num_classes: list,
                PCaps_dim: int, 
                SCaps_dim: list,
                backbone_name='custom', 
                backbone_net_weights=None,
                num_blocks=4,  # for custom backbone
                initial_filters=64,  # for custom backbone
                filter_increment=2, # for custom backbone
                routing_iterations=2,
                **kwargs):
    """
    This is a Hierarchical Deep Capsule Network (HD-CapsNet) model architecture.
    Secondary Capsules are formed by stacking from Coarse to Fine level.
    Output of the primary capsules are fed to the secondary capsules of the first hierarchy.
    The output of the secondary capsules are fed to the secondary capsules of the next hierarchy Using Skip Connections (Primary + Secondary Capsules of the previous hierarchy).
    """
    model_name = kwargs.get('model_name', 'HD-CapsNet')
    base_model = get_backbone(input_shape, 
                              backbone_name, 
                              backbone_net_weights,
                              num_blocks,  # for custom backbone
                              initial_filters,  # for custom backbone
                              filter_increment, # for custom backbone
                              )

    x = base_model.output # Output of the backbone model (Feature Extraction)


    ### Layer: Reshape to n-D primary capsules ###
    
    reshape = keras.layers.Reshape((int((tf.reduce_prod(x.shape[1:]).numpy())/PCaps_dim), PCaps_dim), name="reshape_layer")(x)
    p_caps = keras.layers.Lambda(squash, name='p_caps')(reshape)

    ### Layer: Secondary Capsules for hierarchical levels### 
    s_caps = [] # fine to coarse level secondary capsules
    outputs = [] # prediction for each level
    # Loop over the indices of SCaps_dim
    for LVL, DIM in enumerate(SCaps_dim):
        if LVL <= 0:
            s_caps.append(SecondaryCapsule(n_caps=num_classes[LVL], n_dims=DIM, routings=routing_iterations, name=f"s_caps_L_{LVL}")(p_caps))
        else:
            p_caps_lvl = keras.layers.Reshape((int((tf.reduce_prod(p_caps.shape[1:]).numpy())/s_caps[-1].shape[-1]), s_caps[-1].shape[-1]), name=f"skip_{LVL}")(p_caps)

            skip = keras.layers.Concatenate(axis=1, name=f"skip_connection_{LVL}")([p_caps_lvl, s_caps[-1]])
            
            s_caps.append(SecondaryCapsule(n_caps=num_classes[LVL], n_dims=DIM, routings=routing_iterations, name=f"s_caps_L_{LVL}")(skip))
        
        outputs.append(LengthLayer(name=f'Out_L_{LVL}')(s_caps[-1]))
   
    model = keras.Model(inputs= [base_model.input],
                        outputs= outputs,
                        name=model_name)

    ## Return Model
    return model

def HD_CapsNet_Eff(input_shape,
                       num_classes: list,
                       PCaps_dim: int, 
                       SCaps_dim: list,
                       backbone_name='custom', 
                       backbone_net_weights=None,
                       num_blocks=4,
                       initial_filters=64,
                       filter_increment=2,
                       routing_iterations=2,
                       **kwargs):
    """
    Modified HD_CapsNet using the more efficient HD_SecondaryCapsule layer.

    - This is same as HD_CapsNet but uses the HD_SecondaryCapsule layer.
    - The HD_SecondaryCapsule is a custom layer that implements the HD_CapsNet logic with direct tensor operations.
        - This layer is more efficient than the SecondaryCapsule layer.
    
    Args:
        input_shape: Shape of input images
        num_classes: List of number of classes at each hierarchical level
        PCaps_dim: Dimension of primary capsules
        SCaps_dim: List of dimensions for secondary capsules at each level
        backbone_name: Name of backbone network
        backbone_net_weights: Pre-trained weights for backbone
        num_blocks: Number of blocks in custom backbone
        initial_filters: Initial number of filters in custom backbone
        filter_increment: Filter increment factor in custom backbone
        routing_iterations: Number of routing iterations
    """
    # saved_args = {**locals()}  # Updated to make a copy per loco.loop
    # print("saved_args is", saved_args)
    model_name = kwargs.get('model_name', 'HD-CapsNet')
    
    # Get backbone
    base_model = get_backbone(input_shape, 
                            backbone_name, 
                            backbone_net_weights,
                            num_blocks,
                            initial_filters,
                            filter_increment)

    x = base_model.output

    # Reshape to primary capsules
    reshape = keras.layers.Reshape(
        (int((tf.reduce_prod(x.shape[1:]).numpy())/PCaps_dim), PCaps_dim),
        name="reshape_layer")(x)
    p_caps = keras.layers.Lambda(squash, name='p_caps')(reshape)

    # Single layer that implements HD_CapsNet logic with direct tensor operations
    hierarchical_caps = HD_SecondaryCapsule(
        n_caps_lvl=num_classes,
        n_dims=SCaps_dim,
        routings=routing_iterations,
        name="hd_caps"
    )(p_caps)

    # Generate outputs for each level
    outputs = [LengthLayer(name=f'Out_L_{i}')(caps) 
              for i, caps in enumerate(hierarchical_caps)]

    model = keras.Model(
        inputs=[base_model.input],
        outputs=outputs,
        name=model_name
    )
    return model

class PrimaryCaps(keras.layers.Layer):
    def __init__(self, caps_dim=16, no_filter=32, activation='sigmoid', **kwargs):
        super().__init__(**kwargs)
        self.caps_dim = caps_dim
        self.no_filter = no_filter
        self.activation = activation
        self.pose_map = keras.layers.Conv2D(no_filter * caps_dim, 1)
        self.active_map = keras.layers.Conv2D(no_filter, 1, activation=activation)

    def call(self, x):
        pose = self.pose_map(x)  # (b,h,w,caps_dim * no_filter)
        active = self.active_map(x)  # (b,h,w,no_filter)
        return pose, active

     
    def get_config(self): ### custom layers to be serializable as part of a Functional model
        base_config = super().get_config()
        return {**base_config, 
            "caps_dim": self.caps_dim,
            "no_filter": self.no_filter,
            "activation": self.activation}

def m_step(r, votes, active, bv, ba, lambda_, epsilon=1e-7, variance_epsilon=1e-2, name="m_step"):
    """
    M-step in EM routing. 
    Parameters:
        - r: responsibility matrix from E-step.
        - votes: the predicted votes from the lower capsules.
        - active: activity from previous layers.
        - bv, ba: biases for votes and activities.
        - lambda_: a dynamic scaling factor for the sigmoid function.
        - config: configuration dictionary containing hyperparameters.
    """
    with tf.name_scope(name):
        # Update responsibilities with activation
        r = r * active
        # Sum of responsibilities
        r_sum = tf.reduce_sum(r, axis=-3, keepdims=True)  # (b, h, w, 1, onc, 1)

        # Compute mean (mu) for each capsule
        mu = tf.reduce_sum(r * votes, axis=-3, keepdims=True) / (r_sum + epsilon)  # (b, h, w, 1, onc, pose_dim)

        # Compute variance (sigma) with an offset to avoid sigma=0
        sigma = tf.sqrt(tf.reduce_sum(r * (votes - mu) ** 2, axis=-3, keepdims=True) / (r_sum + epsilon) + variance_epsilon)  # (b, h, w, 1, onc, pose_dim)

        # Compute log-likelihood term for activity update
        l_h = (bv + tf.math.log(sigma + epsilon)) * r_sum  # (b, h, w, 1, onc, pose_dim)

        # Update activity using sigmoid function
        active = tf.nn.sigmoid(lambda_ * (ba - tf.reduce_sum(l_h, axis=-1, keepdims=True)))  # (b, h, w, 1, onc, 1)

        return mu, sigma, active

def e_step(mu, sigma, active, votes, name="e_step"):
    with tf.name_scope(name):
        p_unit0 = -tf.reduce_sum(tf.math.log(sigma + 1e-7), axis=-1, keepdims=True)
        p_unit1 = -tf.reduce_sum((votes-mu)**2/(2*sigma**2 + 1e-7), axis=-1, keepdims=True) #(b,h,w,9xnc,onc,1)
        logp = p_unit0 + p_unit1
        r = tf.nn.softmax(tf.math.log(active + 1e-7) + logp, axis=-2) #(b,h,w,9xnc,onc,1)
        return r

def kernel_tile(x, ksize, strides, name="kernel_tile"):
    with tf.name_scope(name):
        nc = x.shape[-1]
        kernel = np.zeros((ksize, ksize, x.shape[-1], ksize * ksize))
        for i in range(ksize):
            for j in range(ksize):
                kernel[i, j, :, ksize*i + j ] = 1.
        kernel = tf.cast(kernel, 'float32')
        x = tf.nn.depthwise_conv2d(x, kernel, [1, strides, strides, 1], 'VALID') #(b, h, w, nc * 9)
        b, h, w, _ = x.shape
        x = tf.reshape(x, [-1, h, w, ksize * ksize, nc]) #(b,h,w,9,nc)
        return x

class Gen_Votes(keras.layers.Layer):
    def __init__(self, onc, pose_dim=16):
        """
        Parameters:
            - onc: Output number of capsules.
            - pose_dim: Dimensionality of the capsule pose vector, typically 16 (4x4 matrix).
        """
        super().__init__()
        self.onc = onc
        self.pose_dim = pose_dim  # Pose dimensions are now a parameter
    
    def build(self, shape):
        # Get the number of capsules (nc) from the input shape
        _, nc, _ = shape
        # Initialize weights with shape [1, nc, onc, pose_dim_height, pose_dim_width]
        self.w = tf.Variable(tf.random.normal([1, nc, self.onc, self.pose_dim // 4, self.pose_dim // 4], stddev=1.0), trainable=True)
    
    def call(self, x):
        """
        Generate votes for routing.
        Parameters:
            - x: Input tensor (bhw, 9xnc, pose_dim).
        """
        bhw, c, _ = x.shape
        # Reshape to match the shape of weights
        x = tf.reshape(x, [-1, c, 1, self.pose_dim // 4, self.pose_dim // 4])
        # Compute votes: element-wise multiplication with weights
        v = self.w * x  # (bhw, 9xnc, onc, 4, 4)
        # Reshape to flatten the pose matrix into a vector
        v = tf.reshape(v, [-1, c, self.onc, self.pose_dim])  # (bhw, 9xnc, onc, pose_dim)
        return v

    def get_config(self): ### custom layers to be serializable as part of a Functional model
        base_config = super().get_config()
        return {**base_config, 
            "onc": self.onc,
            "pose_dim": self.pose_dim}


def em_routing(votes, active, bv, ba, n_routing, pose_dim=16, name="em_routing"):
    
    """
    EM Routing function.
    Parameters:
        - votes: The votes tensor of shape (b, h, w, 9xnc, onc, pose_dim).
        - active: Activation tensor from the previous layer.
        - bv, ba: Bias variables for votes and activation.
        - n_routing: Number of routing iterations.
        - pose_dim: Dimensionality of the capsule pose vector, default is 16.
    """
    with tf.name_scope(name):
        # Check if the votes are from convolution capsules or class capsules based on dimensions
        if len(votes.shape) == 6:
            # If votes have 6 dimensions, this means it's coming from a convolutional capsule layer, where we still have spatial dimensions (h and w).
            b, h, w, inc, onc, _ = votes.shape
        else:
            b, inc, onc, _ = votes.shape

        # Reshape active for broadcasting
        active = active[..., None, None]  # (b, h, w, 9xnc) --> (b, h, w, 9xnc, 1, 1)
        bv, ba = bv[..., None, :, None], ba[..., None, :, None]  # (1, 1, 1, 1, nc, 1)
        
        # Initialize the assignment probability tensor
        r = tf.ones([inc, onc, 1]) / onc  # (9xnc, onc, 1)
        
        # Base lambda value for dynamic scaling during routing iterations
        base_lambda = 0.01
        for i in range(n_routing):
            lambda_ = base_lambda * (1 - 0.95 ** (i + 1))
            # M-step of routing
            mu, sigma, active_prime = m_step(r, votes, active, bv, ba, lambda_)
            # E-step of routing, update assignment probabilities
            if i < n_routing - 1:
                r = e_step(mu, sigma, active_prime, votes)  # (b, h, w, 9xnc, onc, 1)
        
        # Reshape the output pose tensor based on the layer type (ConvCaps or ClassCaps)
        if len(mu.shape) == 6:  # Convolution capsules
            # If mu has 6 dimensions, this means it's coming from a convolutional capsule layer, where we still have spatial dimensions (h and w).
            pose = tf.reshape(mu, [-1, h, w, pose_dim * onc])
        else:  # Class capsules
            pose = tf.reshape(mu, [-1, pose_dim * onc])
        
        active_prime = tf.squeeze(active_prime, axis=[-3, -1])
        return pose, active_prime


class ConvCaps(keras.layers.Layer):
    def __init__(self, onc, ksize, strides, n_routing=3, pose_dim=16, **kwargs):
        """
        Parameters:
            - onc: Output number of capsules.
            - ksize: Kernel size for the convolutional operation.
            - strides: Stride size for the convolution.
            - n_routing: Number of routing iterations, default is 3.
            - pose_dim: Dimensionality of the capsule pose vector, default is 16.
        """
        super().__init__(**kwargs)
        self.onc = onc
        self.ksize = ksize
        self.strides = strides
        self.n_routing = n_routing
        self.pose_dim = pose_dim

        # Instantiate Gen_Votes with output capsule count (onc) and pose_dim
        self.gen_votes = Gen_Votes(onc, pose_dim)

        # Initialize bias variables for votes and activation
        self.bv = self.add_weight(
            shape=[1, 1, 1, onc],
            initializer=keras.initializers.TruncatedNormal(stddev=1.0),
            trainable=True
        )
        self.ba = self.add_weight(
            shape=[1, 1, 1, onc],
            initializer=keras.initializers.TruncatedNormal(stddev=1.0),
            trainable=True
        )
    
    def call(self, inputs):
        pose, active = inputs

        # Kernel tiling for pose with specified kernel size and stride
        pose = kernel_tile(pose, self.ksize, self.strides)  # (b, h, w, ksize², pose_dim * nc)
        b, h, w, kernel_area, c = pose.shape
        pose = tf.reshape(pose, [-1, kernel_area * c // self.pose_dim, self.pose_dim])  # (bhw, ksize² * nc, pose_dim)

        # Kernel tiling for active capsules
        active = kernel_tile(active, self.ksize, self.strides)  # (b, h, w, ksize², nc)
        b, h, w, _, nc = active.shape
        active = tf.reshape(active, [-1, h, w, self.ksize * self.ksize * nc])  # (b, h, w, ksize² * nc)

        # Generate votes for routing
        votes = self.gen_votes(pose)  # (bhw, ksize² * nc, onc, pose_dim)
        votes = tf.reshape(votes, [-1, h, w, self.ksize * self.ksize * nc, self.onc, self.pose_dim])  # (b, h, w, ksize² * nc, onc, pose_dim)

        # Perform EM routing with the generated votes
        pose, active = em_routing(votes, active, self.bv, self.ba, self.n_routing, self.pose_dim)
        return pose, active
    
        
    def get_config(self): ### custom layers to be serializable as part of a Functional model
        base_config = super().get_config()
        return {**base_config, 
            "onc": self.onc,
            "ksize": self.ksize,
            "strides": self.strides,
            "n_routing": self.n_routing,
            "pose_dim": self.pose_dim}

class ClassCaps(keras.layers.Layer):
    def __init__(self, num_class, n_routing=3, caps_dim=16, use_coord_add=True,**kwargs):
        super().__init__(**kwargs)
        self.num_class = num_class
        self.gen_votes = Gen_Votes(self.num_class)
        self.bv = self.add_weight(
            shape=[1, self.num_class],
            initializer=keras.initializers.TruncatedNormal(stddev=1.0),
            trainable=True
        )
        self.ba = self.add_weight(
            shape=[1, self.num_class],
            initializer=keras.initializers.TruncatedNormal(stddev=1.0),
            trainable=True
        )
        self.n_routing = n_routing
        self.use_coord_add = use_coord_add
        self.caps_dim = caps_dim  # Capsule dimension (e.g., 16 for 4x4 pose matrix)

    def call(self, inputs):
        pose, active = inputs
        b, h, w, c = pose.shape

        # Reshape pose for vote generation
        pose = tf.reshape(pose, [-1, c // self.caps_dim, self.caps_dim])  # (bhw, nc, caps_dim)

        # Generate votes
        votes = self.gen_votes(pose)  # (bhw, nc, k, self.caps_dim)
        votes = tf.reshape(votes, [-1, h, w, c // self.caps_dim, self.num_class, self.caps_dim])  # (b, h, w, nc, k, caps_dim)

        # Add coordinate information if specified
        if self.use_coord_add:
            votes = coord_add(votes, self.caps_dim)  # Adjust to caps_dim dynamically

        # Reshape votes and activations
        votes = tf.reshape(votes, [-1, h * w * c // self.caps_dim, self.num_class, self.caps_dim])  # (b, hwxnc, k, caps_dim)
        active = tf.reshape(active, [-1, h * w * c // self.caps_dim])  # (b, hwxnc)

        # Perform EM routing
        pose, active = em_routing(votes, active, self.bv, self.ba, self.n_routing)

        return pose, active
        
    def get_config(self): ### custom layers to be serializable as part of a Functional model
        base_config = super().get_config()
        return {**base_config, 
            "num_class": self.num_class,
            "n_routing": self.n_routing,
            "use_coord_add": self.use_coord_add,
            "caps_dim": self.caps_dim}


def coord_add(votes, caps_dim, name="coord_add"):
    """
    Adds coordinate information to votes.
    Parameters:
        - votes: Tensor of votes with shape (b, h, w, nc, k, caps_dim).
        - caps_dim: Dimensionality of the capsule's pose matrix (e.g., 16 for a 4x4 matrix).
    """
    with tf.name_scope(name):
        b, h, w, _, _, _ = votes.shape

        # Coordinate addition based on the height (h)
        coord_h = tf.reshape((tf.range(h, dtype=tf.float32) + 0.5), [1, h, 1, 1, 1]) / h
        coord_h = tf.stack([coord_h] + [tf.zeros_like(coord_h) for _ in range(caps_dim - 1)], axis=-1)  # Adjust to caps_dim dynamically

        # Coordinate addition based on the width (w)
        coord_w = tf.reshape((tf.range(w, dtype=tf.float32) + 0.5), [1, 1, w, 1, 1]) / w
        coord_w = tf.stack([tf.zeros_like(coord_w)] + [coord_w] + [tf.zeros_like(coord_w) for _ in range(caps_dim - 2)], axis=-1)

        # Add the coordinates to the votes
        votes = votes + coord_h + coord_w

        return votes

def HD_CapsNet_EM(input_shape,
                    num_classes: list,
                    PCaps_dim: int, 
                    SCaps_dim: list,
                    backbone_name='custom', 
                    backbone_net_weights=None,
                    no_filter=32,  # for primary capsules
                    n_conv_caps=2,  # for conv capsules
                    num_blocks=4,  # for custom backbone
                    initial_filters=64,  # for custom backbone
                    filter_increment=2, # for custom backbone
                    routing_iterations=2,
                    **kwargs):
    """
    This is a Hierarchical Deep Capsule Network (HD-CapsNet) model architecture With EM Routing.
    Secondary Capsules are formed by stacking from Coarse to Fine level.
    Output of the primary capsules are fed to the secondary capsules of the first hierarchy.
    The output of the secondary capsules are fed to the secondary capsules of the next hierarchy Using Skip Connections (Primary + Secondary Capsules of the previous hierarchy).
    """
    model_name = kwargs.get('model_name', 'HD-CapsNet-EM')
    if backbone_name == 'custom':
        input = keras.layers.Input(shape=input_shape, name='input')
        x = keras.layers.Conv2D(32, 5, 2, 'same', activation = 'relu')(input)
        base_model = keras.Model(inputs=input, outputs=x, name='custom_backbone')
    else:
        base_model = get_backbone(input_shape, 
                                backbone_name, 
                                backbone_net_weights,
                                num_blocks,  # for custom backbone
                                initial_filters,  # for custom backbone
                                filter_increment, # for custom backbone
                                )

    x = base_model.output # Output of the backbone model (Feature Extraction)


    ### Layer: Reshape to n-D primary capsules ###
    
    pose, active = PrimaryCaps(caps_dim=SCaps_dim[0], no_filter=no_filter, name = 'primary_caps')(x)
    for i in range(n_conv_caps):
        pose, active = ConvCaps(onc=no_filter, ksize=3, strides=1, n_routing=routing_iterations, pose_dim=SCaps_dim[0],name=f'Conv_Caps_{i}')([pose, active])
    ### Layer: Secondary Capsules for hierarchical levels### 
    outputs = [] # prediction for each level
    # Loop over the indices of SCaps_dim
    for LVL, DIM in enumerate(SCaps_dim):
        class_pose, class_active = ClassCaps(num_class=num_classes[LVL], n_routing=3, caps_dim=SCaps_dim[0],name=f'Class_Caps_{LVL}')([pose, active])
        outputs.append(class_active)
        # outputs.append(LengthLayer(name=f'Out_L_{LVL}')(class_actives[-1]))
   
    model = keras.Model(inputs= [base_model.input],
                        outputs= outputs,
                        name=model_name)

    ## Return Model
    return model
