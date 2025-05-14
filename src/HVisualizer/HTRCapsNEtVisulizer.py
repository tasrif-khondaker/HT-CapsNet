import numpy as np
import tensorflow as tf
from tensorflow import keras
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict
import pandas as pd


class HTRCapsNetVisualizer:
    """
    A class for visualizing HTRCapsNet features using UMAP
    """
    def __init__(self, model: keras.Model, dataset: tf.data.Dataset):
        """
        Initialize the visualizer
        
        Args:
            model: Trained HTRCapsNet model
            dataset: TensorFlow dataset containing the data to visualize
        """
        self.model = model
        self.dataset = dataset
        self.feature_extractor = None
        self.umap_reducer = None
        self.embeddings = {}
        self.labels = {}
        
    def create_feature_extractor(self) -> None:
        """Create a model that outputs capsule features from all levels"""
        # Get the p_caps layer output
        p_caps_layer = self.model.get_layer('p_caps').output
        
        # Get the HTR capsule layer
        htr_caps_layer = self.model.get_layer('hd_caps')
        
        # Create a model that outputs primary caps and all hierarchical level features
        self.feature_extractor = keras.Model(
            inputs=self.model.input,
            outputs=[p_caps_layer] + htr_caps_layer.output
        )
        
    def extract_features(self, batch_size: int = 32) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Extract features from all capsule levels
        
        Args:
            batch_size: Batch size for feature extraction
            
        Returns:
            features: Dictionary containing features from each level
            labels: Dictionary containing labels from each level
        """
        if self.feature_extractor is None:
            self.create_feature_extractor()
            
        features = {
            'primary': [],
            'level_0': [],
            'level_1': [],
            'level_2': []
        }
        
        labels = {
            'level_0': [],
            'level_1': [],
            'level_2': []
        }
        
        # Extract features and labels batch by batch
        for x, y in self.dataset:
            # Extract features
            batch_features = self.feature_extractor(x)
            
            # Store primary capsule features
            features['primary'].append(batch_features[0].numpy())
            
            # Store hierarchical level features
            for i, level_features in enumerate(batch_features[1:]):
                features[f'level_{i}'].append(level_features.numpy())
                
            # Store labels for each level
            for i, level_labels in enumerate(y):
                labels[f'level_{i}'].append(level_labels.numpy())
                
        # Concatenate batches
        for key in features:
            features[key] = np.concatenate(features[key], axis=0)
            
        for key in labels:
            labels[key] = np.concatenate(labels[key], axis=0)
            
        return features, labels
    
    def compute_umap(self, features: Dict[str, np.ndarray], 
                    n_neighbors: int = 15, 
                    min_dist: float = 0.1,
                    n_components: int = 2,
                    random_state: int = 42) -> Dict[str, np.ndarray]:
        """
        Compute UMAP embeddings for all feature levels
        
        Args:
            features: Dictionary containing features from each level
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            n_components: Number of components for UMAP projection
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing UMAP embeddings for each level
        """
        embeddings = {}
        
        for level, level_features in features.items():
            # Reshape features if needed
            if len(level_features.shape) > 2:
                level_features = level_features.reshape(level_features.shape[0], -1)
                
            # Standardize features
            scaler = StandardScaler()
            level_features = scaler.fit_transform(level_features)
            
            # Compute UMAP
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                random_state=random_state
            )
            
            embeddings[level] = reducer.fit_transform(level_features)
            
        return embeddings
    
    def plot_umap(self, embeddings: Dict[str, np.ndarray], 
                 labels: Dict[str, np.ndarray],
                 figsize: Tuple[int, int] = (20, 5),
                 point_size: int = 5,
                 alpha: float = 0.6) -> None:
        """
        Plot UMAP embeddings for all levels
        
        Args:
            embeddings: Dictionary containing UMAP embeddings
            labels: Dictionary containing labels
            figsize: Figure size
            point_size: Size of scatter points
            alpha: Transparency of points
        """
        n_plots = len(embeddings)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        for i, (level, embedding) in enumerate(embeddings.items()):
            ax = axes[i]
            
            if level == 'primary':
                # For primary capsules, color by first level labels
                scatter = ax.scatter(
                    embedding[:, 0], 
                    embedding[:, 1],
                    c=np.argmax(labels['level_0'], axis=1),
                    cmap='tab20',
                    s=point_size,
                    alpha=alpha
                )
                ax.set_title(f'Primary Capsules\nColored by Level 0 Labels')
            else:
                level_idx = int(level.split('_')[1])
                scatter = ax.scatter(
                    embedding[:, 0], 
                    embedding[:, 1],
                    c=np.argmax(labels[f'level_{level_idx}'], axis=1),
                    cmap='tab20',
                    s=point_size,
                    alpha=alpha
                )
                ax.set_title(f'Level {level_idx} Capsules')
                
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            plt.colorbar(scatter, ax=ax)
            
        plt.tight_layout()
        return fig
    
    def visualize_hierarchy(self, save_path: str = None) -> None:
        """
        Extract features, compute UMAP, and create visualization
        
        Args:
            save_path: Path to save the visualization
        """
        # Extract features and labels
        features, labels = self.extract_features()
        
        # Compute UMAP embeddings
        embeddings = self.compute_umap(features)
        
        # Create visualization
        fig = self.plot_umap(embeddings, labels)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    visualizer = HTRCapsNetVisualizer(model, ds_test)
    visualizer.visualize_hierarchy(save_path='htr_capsnet_umap.png')