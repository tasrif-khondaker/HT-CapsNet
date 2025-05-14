import time
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
import psutil
from keras_flops import get_flops

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def identify_capsule_layer_type(layer):
    """Identify the type of capsule layer based on attributes rather than class type"""
    if hasattr(layer, 'n_caps_lvl') and hasattr(layer, 'n_dims'):
        if hasattr(layer, 'taxonomy'):
            return 'HTR_Capsule'
        elif hasattr(layer, 'alphas'):
            return 'HDR_Capsule'
        else:
            return 'ML_Capsule'
    elif hasattr(layer, 'n_caps') and hasattr(layer, 'n_dims'):
        return 'Secondary_Capsule'
    elif hasattr(layer, 'length'):
        return 'Length_Layer'
    return 'Other'

def estimate_layer_flops(layer):
    """Estimate FLOPs for different layer types including capsule layers"""
    
    if isinstance(layer, tf.keras.layers.Conv2D):
        # Conv2D FLOPs = 2 * H * W * Cin * Cout * kernel_size^2
        input_shape = layer.input_shape[1:]  # Exclude batch dimension
        output_shape = layer.output_shape[1:]
        kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
        flops = 2 * output_shape[0] * output_shape[1] * input_shape[-1] * layer.filters * kernel_size
        return flops
        
    elif isinstance(layer, tf.keras.layers.Dense):
        # Dense FLOPs = 2 * input_features * output_features
        flops = 2 * layer.input_shape[-1] * layer.units
        return flops
        
    else:
        layer_type = identify_capsule_layer_type(layer)
        
        if layer_type == 'Secondary_Capsule':
            # Single level capsule
            input_caps = layer.input_shape[1]
            output_caps = layer.n_caps
            input_dims = layer.input_shape[2]
            output_dims = layer.n_dims
            routing_iterations = layer.routings
            
            # Calculate FLOPs for capsule operations
            transform_flops = input_caps * output_caps * input_dims * output_dims * 2
            
            routing_flops = routing_iterations * (
                input_caps * output_caps * 3 +  # Softmax
                input_caps * output_caps * output_dims * 2 +  # Agreement
                input_caps * output_caps * output_dims +  # Weighted sum
                output_caps * output_dims * 4  # Squash function
            )
            
            return transform_flops + routing_flops
            
        elif layer_type in ['ML_Capsule', 'HDR_Capsule', 'HTR_Capsule']:
            input_caps = layer.input_shape[1]
            input_dims = layer.input_shape[2]
            total_flops = 0
            
            for level, (output_caps, output_dims) in enumerate(zip(layer.n_caps_lvl, layer.n_dims)):
                # Basic capsule operations
                transform_flops = input_caps * output_caps * input_dims * output_dims * 2
                
                routing_flops = layer.routings * (
                    input_caps * output_caps * 3 +  # Softmax
                    input_caps * output_caps * output_dims * 2 +  # Agreement
                    input_caps * output_caps * output_dims +  # Weighted sum
                    output_caps * output_dims * 4  # Squash function
                )
                
                # Additional operations for specific types
                if level > 0:
                    if layer_type == 'HDR_Capsule':
                        hierarchical_flops = (
                            output_caps * layer.n_caps_lvl[level-1] * output_dims * 2 +
                            output_caps * layer.n_caps_lvl[level-1] * 3
                        )
                        routing_flops += hierarchical_flops
                    elif layer_type == 'HTR_Capsule':
                        # Additional operations for taxonomy-guided routing
                        taxonomy_flops = (
                            input_caps * output_caps * 2 +  # Mask application
                            output_caps * output_caps * layer.key_dim * 3 +  # Attention
                            output_caps * output_dims * 4  # Layer norm
                        )
                        routing_flops += taxonomy_flops
                
                total_flops += transform_flops + routing_flops
                
                # Update dimensions for next level
                input_caps = output_caps
                input_dims = output_dims
                
            return total_flops
            
        elif layer_type == 'Length_Layer':
            input_shape = layer.input_shape
            return input_shape[1] * input_shape[2] * 2  # multiply and sqrt operations
            
        elif hasattr(layer, 'layers'):  # For layers containing other layers
            return sum(estimate_layer_flops(l) for l in layer.layers)
            
    return 0

def calculate_total_flops(model):
    """Calculate total FLOPs for the entire model"""
    total_flops = 0
    flops_breakdown = {}
    
    for layer in model.layers:
        layer_flops = estimate_layer_flops(layer)
        if layer_flops > 0:
            flops_breakdown[layer.name] = layer_flops
            total_flops += layer_flops
    
    return total_flops, flops_breakdown

class ModelAnalyzer:
    def __init__(self, model):
        self.model = model
        self.complexity_metrics = None
        
    def analyze(self):
        """Analyze model complexity and generate detailed metrics"""
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        total_flops, flops_breakdown = calculate_total_flops(self.model)
        
        # Calculate percentage breakdown
        flops_percentage = {name: (flops/total_flops)*100 for name, flops in flops_breakdown.items()}
        
        self.complexity_metrics = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'total_flops': total_flops,
            'flops_breakdown': flops_breakdown,
            'flops_percentage': flops_percentage
        }
        
    def print_summary(self):
        """Print detailed summary of model complexity"""
        if self.complexity_metrics is None:
            self.analyze()
            
        print("\nModel Complexity Analysis")
        print("="*50)
        print(f"Total Parameters: {self.complexity_metrics['total_params']:,}")
        print(f"Trainable Parameters: {self.complexity_metrics['trainable_params']:,}")
        print(f"Non-trainable Parameters: {self.complexity_metrics['non_trainable_params']:,}")
        print(f"Total FLOPs: {self.complexity_metrics['total_flops']:,}")
        
        print("\nFLOPs breakdown by layer:")
        print("-"*50)
        for layer_name, flops in self.complexity_metrics['flops_breakdown'].items():
            percentage = self.complexity_metrics['flops_percentage'][layer_name]
            print(f"{layer_name}: {flops:,} FLOPs ({percentage:.2f}%)")
            
    def plot_flops_distribution(self, save_path=None):
        """Create visualization of FLOPs distribution across layers"""
        if self.complexity_metrics is None:
            self.analyze()
            
        # Prepare data for plotting
        layers = list(self.complexity_metrics['flops_percentage'].keys())
        percentages = list(self.complexity_metrics['flops_percentage'].values())
        
        # Create figure
        plt.figure(figsize=(12, 6))
        bars = plt.bar(layers, percentages)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Percentage of Total FLOPs')
        plt.title('Distribution of Computational Complexity Across Layers')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def get_flops_keras(self):
        flops = get_flops(self.model, batch_size=1)
        return {f'flops in M': f'{flops/1e6} M', f'flops in G': f'{flops/1e9} G'}