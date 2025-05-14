import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from typing import List, Tuple, Union, Optional
from treelib import Tree

class HTRCapsuleCAM:
    """
    Specialized Class Activation Mapping for HTR Capsule Networks.
    Visualizes activations directly from capsule vectors and routing weights.
    """
    def __init__(self, model: keras.Model, taxonomy: list, input_shape: tuple):
        """
        Initialize Capsule CAM handler.
        
        Args:
            model: Trained HTRCapsNet model
            taxonomy: Hierarchical taxonomy matrix
            input_shape: Shape of input images
        """
        self.model = model
        self.taxonomy = taxonomy
        self.input_shape = input_shape
        self.htr_layer = self._get_htr_capsule_layer()
        
        # Create intermediate models for activation extraction
        self.activation_model = self._create_activation_model()
        
    def _get_htr_capsule_layer(self) -> keras.layers.Layer:
        """Get the HTR capsule layer."""
        for layer in self.model.layers:
            if 'htr_capsule' in layer.__class__.__name__.lower() or 'hd_caps' in layer.name:
                return layer
        raise ValueError("Could not find HTR capsule layer")
    
    def _create_activation_model(self) -> keras.Model:
        """Create model to extract intermediate activations."""
        primary_caps_input = self.model.get_layer('p_caps').output
        htr_caps_outputs = self.htr_layer(primary_caps_input)
        
        return keras.Model(
            inputs=self.model.input,
            outputs=[primary_caps_input, htr_caps_outputs]
        )

    @tf.function
    def compute_capsule_importance(self,
                                images: tf.Tensor,
                                level: int,
                                target_class: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute importance scores for capsules at a specific level.
        
        Args:
            images: Input images tensor
            level: Target hierarchy level
            target_class: Target class index
            
        Returns:
            Tuple of (primary capsule activations, capsule vectors, routing weights)
        """
        with tf.GradientTape() as tape:
            # Get primary capsules and hierarchical capsule outputs
            primary_caps, htr_outputs = self.activation_model(images)
            
            # Get output for target level and class
            level_output = htr_outputs[level]
            class_output = level_output[:, target_class, :]
            
            # Watch primary capsules for gradient computation
            tape.watch(primary_caps)
        
        # Compute gradients w.r.t primary capsules
        grads = tape.gradient(class_output, primary_caps)
        
        # Compute importance scores
        importance = tf.reduce_mean(tf.abs(grads), axis=-1)
        
        return primary_caps, level_output, importance

    def compute_hierarchical_attention(self,
                                    images: tf.Tensor,
                                    level: int,
                                    target_class: int) -> tf.Tensor:
        """
        Compute hierarchical attention maps using taxonomy relationships.
        
        Args:
            images: Input images
            level: Target hierarchy level
            target_class: Target class index
            
        Returns:
            Attention maps incorporating hierarchical information
        """
        # Get capsule activations and importance
        primary_caps, level_caps, importance = self.compute_capsule_importance(
            images, level, target_class)
        
        # Get taxonomy mask for current level
        if level > 0:
            taxonomy_mask = tf.constant(self.taxonomy[level-1], dtype=tf.float32)
            parent_mask = taxonomy_mask[target_class]
            
            # Incorporate parent relationship into attention
            parent_importance = self.compute_capsule_importance(
                images, level-1, tf.argmax(parent_mask))[2]
            
            # Combine child and parent importance
            importance = 0.7 * importance + 0.3 * parent_importance
        
        return importance

    def generate_capsule_cam(self,
                           images: np.ndarray,
                           level: int,
                           target_class: int,
                           upsample: bool = True) -> np.ndarray:
        """
        Generate CAM from capsule activations.
        
        Args:
            images: Input images
            level: Target hierarchy level
            target_class: Target class index
            upsample: Whether to upsample map to input size
            
        Returns:
            Activation map for the specified class
        """
        # Convert inputs to tensors
        images = tf.cast(images, tf.float32)
        
        # Get hierarchical attention
        attention = self.compute_hierarchical_attention(
            images, level, target_class)
        
        # Reshape attention to spatial dimensions
        feature_size = int(np.sqrt(attention.shape[1]))
        cam = tf.reshape(attention, [-1, feature_size, feature_size])
        
        if upsample:
            # Resize to input dimensions
            cam = tf.image.resize(
                cam[..., tf.newaxis],
                (self.input_shape[0], self.input_shape[1]),
                method='bilinear'
            )
            cam = tf.squeeze(cam)
        
        # Normalize
        cam = tf.maximum(cam, 0)
        cam = (cam - tf.reduce_min(cam)) / (
            tf.reduce_max(cam) - tf.reduce_min(cam) + keras.backend.epsilon()
        )
        
        return cam.numpy()

    def visualize_capsule_attention(self,
                                  image: np.ndarray,
                                  level: int,
                                  target_class: int,
                                  alpha: float = 0.4) -> np.ndarray:
        """
        Visualize capsule attention maps.
        
        Args:
            image: Input image
            level: Target hierarchy level
            target_class: Target class index
            alpha: Overlay transparency
            
        Returns:
            Visualization with attention overlay
        """
        # Generate attention map
        cam = self.generate_capsule_cam(
            image[np.newaxis, ...], level, target_class)
        
        # Create heatmap
        heatmap = np.uint8(255 * cam[0])
        colored_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay on image
        overlay = (
            (1 - alpha) * image + 
            alpha * colored_map.astype(np.float32)
        )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay

    def analyze_capsule_hierarchy(self,
                                images: np.ndarray,
                                target_classes: List[int],
                                return_attention: bool = False
                                ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        Analyze capsule activations across hierarchy levels.
        
        Args:
            images: Input images
            target_classes: Target class for each level
            return_attention: Whether to return attention maps
            
        Returns:
            List of visualizations and optionally attention maps
        """
        visualizations = []
        attention_maps = []
        
        for level, target_class in enumerate(target_classes):
            # Generate visualization
            vis = self.visualize_capsule_attention(
                images[0], level, target_class)
            visualizations.append(vis)
            
            if return_attention:
                # Get attention map
                attention = self.generate_capsule_cam(
                    images, level, target_class)
                attention_maps.append(attention)
        
        if return_attention:
            return visualizations, attention_maps
        return visualizations

    def compare_level_activations(self,
                                images: np.ndarray,
                                target_classes: List[int]
                                ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Compare activations between hierarchical levels.
        
        Args:
            images: Input images
            target_classes: Target class for each level
            
        Returns:
            Tuple of (attention maps, correlation matrix)
        """
        # Get attention maps for all levels
        _, attention_maps = self.analyze_capsule_hierarchy(
            images, target_classes, return_attention=True)
        
        # Compute correlation matrix
        n_levels = len(attention_maps)
        correlation_matrix = np.zeros((n_levels, n_levels))
        
        for i in range(n_levels):
            for j in range(n_levels):
                if i != j:
                    # Compute correlation between levels
                    correlation = tf.reduce_mean(tf.image.ssim(
                        attention_maps[i][..., tf.newaxis],
                        attention_maps[j][..., tf.newaxis],
                        1.0
                    ))
                    correlation_matrix[i, j] = correlation.numpy()
        
        return attention_maps, correlation_matrix