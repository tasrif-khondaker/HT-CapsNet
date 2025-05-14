import numpy as np
import tensorflow as tf
from tensorflow import keras
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict
import pandas as pd
from scipy.spatial import ConvexHull
class HTRClassCenterVisualizer:
    """
    A class for visualizing class centers and sample distributions in HTRCapsNet
    """
    def __init__(self, model: keras.Model, dataset: tf.data.Dataset, class_names: Dict[str, List[str]]):
        """
        Initialize the visualizer
        
        Args:
            model: Trained HTRCapsNet model
            dataset: TensorFlow dataset containing the data to visualize
            class_names: Dictionary containing class names for each level
                        e.g., {'level_0': ['transport', 'animal'], 
                              'level_1': ['sky', 'water', 'road', ...]}
        """
        self.model = model
        self.dataset = dataset
        self.class_names = class_names
        self.feature_extractor = None
        self.umap_reducer = None
        self.features = None
        self.labels = None
        self.embeddings = None
        
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
        
    def compute_class_centers(self, embeddings: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Compute class centers and sample indices for each class
        
        Args:
            embeddings: UMAP embeddings
            labels: One-hot encoded labels
            
        Returns:
            centers: Array of class centers
            class_samples: List of sample indices for each class
        """
        n_classes = labels.shape[1]
        centers = np.zeros((n_classes, 2))  # 2D UMAP embeddings
        class_samples = []
        
        for i in range(n_classes):
            class_mask = labels[:, i] == 1
            class_points = embeddings[class_mask]
            centers[i] = np.mean(class_points, axis=0)
            class_samples.append(np.where(class_mask)[0])
            
        return centers, class_samples
    
    def compute_confidence_ellipse(self, points: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute confidence ellipse for a set of points
        
        Args:
            points: 2D points array
            confidence: Confidence level for the ellipse
            
        Returns:
            eigenvals: Eigenvalues of the covariance matrix
            eigenvecs: Eigenvectors of the covariance matrix
        """
        cov = np.cov(points, rowvar=False)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Scale eigenvalues for desired confidence level
        chi2_val = 5.991  # 95% confidence for 2 degrees of freedom
        scaling = np.sqrt(chi2_val)
        eigenvals = scaling * np.sqrt(eigenvals)
        
        return eigenvals, eigenvecs
    
    def plot_class_centers(self, 
                          level: str,
                          embeddings: np.ndarray,
                          labels: np.ndarray,
                          ax: plt.Axes,
                          plot_samples: bool = True,
                          plot_ellipse: bool = True,
                          sample_alpha: float = 0.3,
                          center_size: int = 150) -> None:
        """
        Plot class centers and sample distributions for a single level
        
        Args:
            level: Hierarchy level name
            embeddings: UMAP embeddings
            labels: One-hot encoded labels
            ax: Matplotlib axes
            plot_samples: Whether to plot individual samples
            plot_ellipse: Whether to plot confidence ellipses
            sample_alpha: Transparency of sample points
            center_size: Size of center markers
        """
        centers, class_samples = self.compute_class_centers(embeddings, labels)
        n_classes = labels.shape[1]
        
        # Create colormap
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
        
        # Plot samples and ellipses for each class
        for i in range(n_classes):
            class_points = embeddings[class_samples[i]]
            color = colors[i]
            
            if plot_samples:
                ax.scatter(class_points[:, 0], class_points[:, 1],
                          c=[color], alpha=sample_alpha, label=self.class_names[level][i])
            
            if plot_ellipse and len(class_points) > 2:
                eigenvals, eigenvecs = self.compute_confidence_ellipse(class_points)
                
                # Create ellipse points
                theta = np.linspace(0, 2*np.pi, 100)
                ellipse = np.array([eigenvals[0] * np.cos(theta), eigenvals[1] * np.sin(theta)])
                ellipse = np.dot(eigenvecs, ellipse)
                ellipse += centers[i, :, np.newaxis]
                
                ax.plot(ellipse[0], ellipse[1], c=color, linestyle='--', alpha=0.5)
            
            # Plot class center
            ax.scatter(centers[i, 0], centers[i, 1], c=[color], 
                      marker='*', s=center_size, edgecolor='black', linewidth=1,
                      label=f'{self.class_names[level][i]} (center)')
        
        ax.set_title(f'Level {level.split("_")[1]} Class Centers and Distributions')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
    def visualize_all_levels(self, 
                            save_path: str = None,
                            figsize: Tuple[int, int] = (25, 6),
                            plot_samples: bool = True,
                            plot_ellipse: bool = True) -> None:
        """
        Create visualizations for all hierarchical levels
        
        Args:
            save_path: Path to save the visualization
            figsize: Figure size
            plot_samples: Whether to plot individual samples
            plot_ellipse: Whether to plot confidence ellipses
        """
        if self.feature_extractor is None:
            self.create_feature_extractor()
        
        # Extract features and labels
        features = []
        labels = []
        
        for x, y in self.dataset:
            batch_features = self.feature_extractor(x)
            features.append([f.numpy() for f in batch_features])
            labels.append([l.numpy() for l in y])
        
        # Concatenate batches
        self.features = [np.concatenate([batch[i] for batch in features], axis=0) 
                        for i in range(len(features[0]))]
        self.labels = [np.concatenate([batch[i] for batch in labels], axis=0) 
                      for i in range(len(labels[0]))]
        
        # Create UMAP embeddings
        self.embeddings = []
        for level_features in self.features[1:]:  # Skip primary capsules
            # Reshape and standardize
            level_features = level_features.reshape(level_features.shape[0], -1)
            scaler = StandardScaler()
            level_features = scaler.fit_transform(level_features)
            
            # Compute UMAP
            reducer = umap.UMAP(random_state=42)
            self.embeddings.append(reducer.fit_transform(level_features))
        
        # Create visualization
        fig, axes = plt.subplots(1, len(self.embeddings), figsize=figsize)
        
        for i, (embedding, level_labels) in enumerate(zip(self.embeddings, self.labels)):
            level = f'level_{i}'
            self.plot_class_centers(
                level=level,
                embeddings=embedding,
                labels=level_labels,
                ax=axes[i],
                plot_samples=plot_samples,
                plot_ellipse=plot_ellipse
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
    def get_class_statistics(self) -> pd.DataFrame:
        """
        Compute statistics about class distributions
        
        Returns:
            DataFrame containing statistics for each class at each level
        """
        stats = []
        
        for i, (embedding, level_labels) in enumerate(zip(self.embeddings, self.labels)):
            centers, class_samples = self.compute_class_centers(embedding, level_labels)
            
            for j in range(level_labels.shape[1]):
                class_points = embedding[class_samples[j]]
                
                # Compute distances to center
                distances = np.sqrt(np.sum((class_points - centers[j]) ** 2, axis=1))
                
                stats.append({
                    'Level': f'level_{i}',
                    'Class': self.class_names[f'level_{i}'][j],
                    'Num_Samples': len(class_points),
                    'Mean_Distance_to_Center': np.mean(distances),
                    'Std_Distance_to_Center': np.std(distances),
                    'Min_Distance_to_Center': np.min(distances),
                    'Max_Distance_to_Center': np.max(distances)
                })
        
        return pd.DataFrame(stats)

if __name__ == "__main__":
    
    # Define class names for each level
    class_names = {
        'level_0': ['transport', 'animal'],
        'level_1': ['sky', 'water', 'road', 'reptile', 'bird', 'pet', 'medium'],
        'level_2': ['airplane', 'ship', 'automobile', 'bird', 'cat', 'deer', 'frog', 'horse', 'dog', 'truck']
    }

    # Create the visualizer
    visualizer = HTRClassCenterVisualizer(model, ds_test.take(10), class_names)

    # Generate visualizations
    visualizer.visualize_all_levels(
        save_path='htr_capsnet_class_centers.png',
        plot_samples=True,
        plot_ellipse=True
    )

    # Get statistical analysis of class distributions
    stats_df = visualizer.get_class_statistics()
    print(stats_df)