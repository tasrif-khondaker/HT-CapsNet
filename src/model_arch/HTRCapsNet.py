import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D # type: ignore
from tensorflow.keras import regularizers, optimizers # type: ignore
from tensorflow.keras import backend as K # type: ignore

import numpy as np
import pandas as pd # type: ignore
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
        
class LengthLayer(keras.layers.Layer):
    """
    Compute the length of capsule vectors.
    inputs: shape=[None, num_capsule, dim_vector]
    output: shape=[None, num_capsule]
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def call(self, X,training=None):
        logits = safe_norm(X, axis=-1, name="logits")
        if not training:
            logits = tf.nn.softmax(logits)
            # logits = tf.math.divide(logits,tf.reshape(tf.reduce_sum(logits,-1),(-1,1),name='reshape'),name='Normalizing_Probability')
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

class HTR_Capsule(keras.layers.Layer):
    """
    Hierarchical Taxonomy-aware Routing (HTR) Capsule Layer
    
    Key innovations:
    1. Taxonomy-guided Attention: Uses taxonomy matrix to weight routing coefficients
    2. Level-wise Consistency Regularization: Enforces parent-child relationships
    3. Adaptive Hierarchical Agreement: Adjusts routing based on level depth
    4. Cross-level Information Flow: Enables bidirectional information exchange
    """
    def __init__(self, 
                 n_caps_lvl: list, 
                 n_dims: list, 
                 taxonomy: list, 
                 routings=2,
                 taxonomy_temperature: float = 10.0,  # Controls the sharpness of sigmoid
                 mask_threshold_high: float = 0.9,   # Upper bound for mask values
                 mask_threshold_low: float = 0.1,    # Lower bound for mask values
                 mask_temperature: float = 0.5,   # Temperature for routing softmax
                 mask_center: float = 0.5,       # Center for mask sigmoid
                 num_heads=16,  # Number of attention heads
                 key_dim=32,   # Dimension of key projection
                 **kwargs):
        super().__init__(**kwargs)
        # Convert inputs to regular Python lists if they're numpy arrays or tensors
        self.n_caps_lvl = [int(x) for x in n_caps_lvl]
        self.n_dims = [int(x) for x in n_dims]
        # Handle nested taxonomy structure
        self.taxonomy_data = []
        for level_matrix in taxonomy:
            level_data = []
            for row in level_matrix:
                level_data.append([float(x) for x in row])
            self.taxonomy_data.append(level_data)
        # Store routing hyperparameters
        self.routings = int(routings)

        self.taxonomy_temperature = float(taxonomy_temperature)
        self.mask_range = float(mask_threshold_high - mask_threshold_low)
        self.mask_threshold_low = float(mask_threshold_low)
        self.mask_temperature = float(mask_temperature)
        self.mask_center = float(mask_center)

        # Initialize lists for weights
        self.W = []
        self.h_gates = []
        self.dim_transforms = []

        # Attention parameters
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention_layers = []
        self.layer_norms = []

    def build(self, batch_input_shape):
        # Convert taxonomy to TensorFlow constants during build
        self.taxonomy = [tf.constant(t, dtype=tf.float32) for t in self.taxonomy_data]
        
        self.batch_size = batch_input_shape[0]
        self.input_caps = int(batch_input_shape[1])
        self.input_dim = int(batch_input_shape[2])
        
        # Create transformation matrices for each level
        for level, (n_caps, n_dims) in enumerate(zip(self.n_caps_lvl, self.n_dims)):
            if level == 0:
                # First level transforms from input_dim
                W_level = self.add_weight(
                    name=f"W_level_{level}", 
                    shape=(1, self.input_caps, n_caps, n_dims, self.input_dim),
                    initializer=keras.initializers.RandomNormal(stddev=0.1),
                    trainable=True
                )
            else:
                # Calculate new number of primary capsules for this level
                new_n_caps = (self.input_caps * self.input_dim) // self.n_dims[level-1]
                prev_total_caps = new_n_caps + self.n_caps_lvl[level-1]
                
                W_level = self.add_weight(
                    name=f"W_level_{level}", 
                    shape=(1, prev_total_caps, n_caps, n_dims, self.n_dims[level-1]),
                    initializer=keras.initializers.RandomNormal(stddev=0.1),
                    trainable=True
                )
                
                # Add hierarchical gate
                h_gate = self.add_weight(
                    name=f"h_gate_level_{level}",
                    shape=(1, n_caps, self.n_caps_lvl[level-1]),
                    initializer=keras.initializers.Constant(0.5),
                    trainable=True
                )
                self.h_gates.append(h_gate)
                
                # Add dimension transformation matrix
                dim_transform = self.add_weight(
                    name=f"dim_transform_{level}",
                    shape=(self.n_dims[level-1], n_dims),
                    initializer=keras.initializers.RandomNormal(stddev=0.1),
                    trainable=True
                )
                self.dim_transforms.append(dim_transform)
            
            # Add attention layer
            attention = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.key_dim,
                value_dim=self.key_dim,
                name=f'mha_level_{level}'
            )
            # Add Layer Normalization
            layer_norm = keras.layers.LayerNormalization(epsilon=1e-6, name=f"layer_norm_{level}")

            self.W.append(W_level)
            self.attention_layers.append(attention)
            self.layer_norms.append(layer_norm)
        
        super().build(batch_input_shape)

    def taxonomy_guided_routing(self, raw_weights, level, prev_predictions=None):
        """
        Taxonomy-guided routing.
        
        Args:
            raw_weights: shape [batch_size, num_input_caps, num_output_caps, 1, 1]
            level: current hierarchy level
        """
        if level == 0:
            return tf.nn.softmax(raw_weights, axis=2)
        
        batch_size = tf.shape(raw_weights)[0]
        num_input_caps = int(raw_weights.shape[1])
        num_output_caps = int(raw_weights.shape[2])
        
        # Get taxonomy mask for current level
        taxonomy_mask = self.taxonomy[level-1]  # [parent_classes, child_classes]
        parent_classes = int(taxonomy_mask.shape[0])
        
        repeats = num_input_caps // parent_classes

        if prev_predictions is not None:
            # Get activation strengths: [batch_size, parent_classes]
            prev_activations = safe_norm(prev_predictions, axis=-1)
            prev_activations = tf.nn.softmax(prev_activations, axis=-1)
            
            # Reshape for broadcasting: [batch_size, parent_classes, 1]
            prev_activations = tf.expand_dims(prev_activations, axis=2)
            
            # Expand taxonomy: [1, parent_classes, child_classes]
            taxonomy_expanded = tf.expand_dims(taxonomy_mask, axis=0)
            
            # Broadcast multiply: [batch_size, parent_classes, child_classes]
            weighted_taxonomy = taxonomy_expanded * prev_activations
        else:
            # If no previous predictions, just use taxonomy mask
            weighted_taxonomy = tf.expand_dims(taxonomy_mask, axis=0)  # [1, parent_classes, child_classes]
            weighted_taxonomy = tf.tile(weighted_taxonomy, [batch_size, 1, 1])  # [batch_size, parent_classes, child_classes]

        # Apply sigmoid with temperature to get soft routing coefficients
        soft_taxonomy = self.mask_range * tf.sigmoid(
            self.taxonomy_temperature * (weighted_taxonomy - self.mask_center)
        ) + self.mask_threshold_low  # [batch_size, parent_classes, child_classes]
        
        # Add dimensions for routing
        soft_taxonomy = tf.expand_dims(tf.expand_dims(soft_taxonomy, axis=-1), axis=-1)
        # Shape: [batch_size, parent_classes, child_classes, 1, 1]
        
        # Repeat for each capsule group
        # First create the repeating pattern
        replicated_mask = tf.tile(soft_taxonomy, [1, repeats, 1, 1, 1])
        
        # Handle remaining capsules
        remaining = num_input_caps - (repeats * parent_classes)
        if remaining > 0:
            remainder_slice = soft_taxonomy[:, :1, :, :, :]  # Take just the first pattern
            remainder_mask = tf.tile(remainder_slice, [1, remaining, 1, 1, 1])
            extended_mask = tf.concat([replicated_mask, remainder_mask], axis=1)
        else:
            extended_mask = replicated_mask
            
        # Now the extended_mask shape matches raw_weights: [batch_size, num_input_caps, num_output_caps, 1, 1]
        masked_weights = raw_weights * extended_mask
        # Apply temperature and softmax
        routing_weights = tf.nn.softmax(masked_weights * self.mask_temperature, axis=2)
        
        return routing_weights
    
    def hierarchical_agreement(self, votes, prev_predictions, level):
        """Modified hierarchical agreement with static shape handling"""
        if level == 0:
            return votes
        
        # Get static shapes
        batch_size = tf.shape(votes)[0]
        num_input_caps = int(votes.shape[1])
        num_output_caps = int(votes.shape[2])
        output_dim = int(votes.shape[3])
        
        # Transform previous predictions
        prev_transformed = tf.matmul(prev_predictions, self.dim_transforms[level-1])
        
        # Compute similarities
        votes_reshaped = tf.reshape(tf.squeeze(votes, axis=-1), 
                                  [batch_size, num_input_caps * num_output_caps, output_dim])
        
        agreement = tf.matmul(votes_reshaped, tf.transpose(prev_transformed, [0, 2, 1]))
        agreement = tf.reshape(agreement, 
                             [batch_size, num_input_caps, num_output_caps, prev_predictions.shape[1]])
        
        # Apply hierarchical gating
        h_gate = tf.expand_dims(self.h_gates[level-1], axis=1)
        gated = agreement * h_gate
        
        # Generate consistency scores
        consistency = tf.sigmoid(tf.reduce_sum(gated, axis=-1, keepdims=True))
        consistency = tf.expand_dims(consistency, axis=-1)
        
        return votes * consistency
    
    def route_level(self, X, W, n_caps, level, prev_predictions=None):
        """Modified routing with static shape handling"""
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
                             dtype=tf.float32)
        
        
        # Apply hierarchical agreement
        if prev_predictions is not None:
            caps2_predicted = self.hierarchical_agreement(
                caps2_predicted, prev_predictions, level
            )
        for i in range(self.routings):
            # Get routing weights
            routing_weights = self.taxonomy_guided_routing(raw_weights, level, prev_predictions)
            
            # Apply routing
            weighted_predictions = routing_weights * caps2_predicted
            weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True)
            caps2_output = squash(weighted_sum, axis=-2)
            
            if i < self.routings - 1:
                # Update agreement
                caps2_output_tiled = tf.tile(caps2_output, [1, caps1_n_caps, 1, 1, 1])
                agreement = tf.matmul(caps2_predicted, caps2_output_tiled, transpose_a=True)
                raw_weights += agreement
        
        output = tf.squeeze(caps2_output, axis=[1,4])  # [batch_size, n_caps, n_dims]
        if prev_predictions is not None:
            attended_output = self.attention_layers[level](output, prev_predictions, prev_predictions)
        else:
            attended_output = self.attention_layers[level](output, output, output)
        
        return self.layer_norms[level](output + attended_output)
    
    def call(self, X):
        """Modified call method with proper shape handling"""
        s_caps = []
        batch_size = tf.shape(X)[0]
        
        for level, (n_caps, n_dims) in enumerate(zip(self.n_caps_lvl, self.n_dims)):
            if level == 0:
                s_caps.append(self.route_level(X, self.W[level], n_caps, level))
            else:
                # Handle skip connections
                new_n_caps = (self.input_caps * self.input_dim) // self.n_dims[level-1]
                p_caps_lvl = tf.reshape(X, [batch_size, new_n_caps, self.n_dims[level-1]])
                skip = tf.concat([p_caps_lvl, s_caps[-1]], axis=1)
                
                s_caps.append(self.route_level(
                    skip, self.W[level], n_caps, level,
                    prev_predictions=s_caps[-1]
                ))
        
        return s_caps

    def compute_output_shape(self, batch_input_shape):
        return [(batch_input_shape[0], n_caps, n_dims) 
                for n_caps, n_dims in zip(self.n_caps_lvl, self.n_dims)]

    def get_config(self):
        """Properly handle serialization of the layer"""
        config = super().get_config()
        config.update({
            'n_caps_lvl': self.n_caps_lvl,
            'n_dims': self.n_dims,
            'taxonomy': self.taxonomy_data,  # Already nested Python lists
            'routings': self.routings,
            'taxonomy_temperature': self.taxonomy_temperature,
            'mask_threshold_high': self.mask_threshold_low + self.mask_range,
            'mask_threshold_low': self.mask_threshold_low,
            'mask_temperature': self.mask_temperature,
            'mask_center': self.mask_center
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Properly handle deserialization of the layer"""
        return cls(**config)



def HTRCapsNet_model(input_shape,
                       num_classes: list,
                       PCaps_dim: int, 
                       SCaps_dim: list,
                       taxonomy: list,
                       backbone_name='custom', 
                       backbone_net_weights=None,
                       num_blocks=4,
                       initial_filters=64,
                       filter_increment=2,
                       routing_iterations=2,
                       taxonomy_temperature: float = 5.0,
                       mask_threshold_high: float = 0.9,
                       mask_threshold_low: float = 0.1,
                       mask_temperature: float = 0.5,
                       mask_center: float = 0.5,
                       num_heads=16,
                       key_dim=32,
                       **kwargs):
    # saved_args = {**locals()}  # Updated to make a copy per loco.loop
    # print("Input Given to HTRCapsNet_model(): ", saved_args)
    model_name = kwargs.get('model_name', 'HTR-CapsNet')
    
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
    hierarchical_caps = HTR_Capsule(
        n_caps_lvl=num_classes,
        n_dims=SCaps_dim,
        taxonomy=taxonomy,
        routings=routing_iterations,
        taxonomy_temperature=taxonomy_temperature,
        mask_threshold_high=mask_threshold_high,
        mask_threshold_low=mask_threshold_low,
        mask_temperature=mask_temperature,
        mask_center = mask_center,       # Center for mask sigmoid
        num_heads = num_heads,  # Number of attention heads
        key_dim = key_dim,   # Dimension of key projection
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
    