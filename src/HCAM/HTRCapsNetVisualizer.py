import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import cv2

class HTRCapsNetVisualizer:
    """
    Visualization tools for HTRCapsNet activation analysis and interpretation.
    Provides comprehensive plotting capabilities for capsule network analysis.
    """
    def __init__(self, capsule_cam: 'HTRCapsuleCAM', class_names: Dict[int, List[str]]):
        """
        Initialize visualizer with CAM generator and class labels.
        
        Args:
            capsule_cam: Initialized HTRCapsuleCAM instance
            class_names: Dictionary mapping level to list of class names
        """
        self.cam = capsule_cam
        self.class_names = class_names
        self.colors = plt.cm.jet(np.linspace(0, 1, 256))
    
    def plot_single_level_analysis(self,
                                 image: np.ndarray,
                                 level: int,
                                 target_class: int,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Plot detailed analysis for a single hierarchical level.
        
        Args:
            image: Input image
            level: Hierarchical level to analyze
            target_class: Target class index
            save_path: Optional path to save visualization
            figsize: Figure size (width, height)
        """
        # Create figure with grid
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 3, figure=fig)
        
        # Plot original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Plot attention heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        attention_map = self.cam.generate_capsule_cam(
            image[np.newaxis, ...], level, target_class)
        im = ax2.imshow(attention_map[0], cmap='jet')
        ax2.set_title(f'Attention Map - Level {level}\n'
                     f'Class: {self.class_names[level][target_class]}')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2)
        
        # Plot overlay
        ax3 = fig.add_subplot(gs[0, 2])
        overlay = self.cam.visualize_capsule_attention(
            image, level, target_class)
        ax3.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax3.set_title('Attention Overlay')
        ax3.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_hierarchical_analysis(self,
                                 image: np.ndarray,
                                 target_classes: List[int],
                                 save_path: Optional[str] = None,
                                 show_correlation: bool = True) -> None:
        """
        Plot comprehensive hierarchical analysis.
        
        Args:
            image: Input image
            target_classes: List of target classes for each level
            save_path: Optional path to save visualization
            show_correlation: Whether to show correlation matrix
        """
        n_levels = len(target_classes)
        fig_width = 15
        fig_height = 4 * (n_levels + (1 if show_correlation else 0))
        
        # Create figure
        fig = plt.figure(figsize=(fig_width, fig_height))
        n_rows = n_levels + (1 if show_correlation else 0)
        gs = GridSpec(n_rows, 4, figure=fig)
        
        # Get all visualizations and correlation data
        attention_maps, correlation_matrix = self.cam.compare_level_activations(
            image[np.newaxis, ...], target_classes)
        
        # Plot hierarchical levels
        for level in range(n_levels):
            # Original image
            ax1 = fig.add_subplot(gs[level, 0])
            ax1.imshow(image)
            ax1.set_title(f'Level {level} Input')
            ax1.axis('off')
            
            # Attention heatmap
            ax2 = fig.add_subplot(gs[level, 1])
            im = ax2.imshow(attention_maps[level][0], cmap='jet')
            ax2.set_title(f'Attention Map\n'
                         f'Class: {self.class_names[level][target_classes[level]]}')
            ax2.axis('off')
            plt.colorbar(im, ax=ax2)
            
            # Attention overlay
            ax3 = fig.add_subplot(gs[level, 2])
            overlay = self.cam.visualize_capsule_attention(
                image, level, target_classes[level])
            ax3.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax3.set_title('Overlay')
            ax3.axis('off')
            
            # Activation histogram
            ax4 = fig.add_subplot(gs[level, 3])
            ax4.hist(attention_maps[level][0].flatten(), bins=50)
            ax4.set_title('Activation Distribution')
            ax4.set_xlabel('Activation Strength')
            ax4.set_ylabel('Frequency')
        
        # Plot correlation matrix if requested
        if show_correlation:
            ax_corr = fig.add_subplot(gs[-1, :2])
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       cmap='coolwarm',
                       center=0,
                       ax=ax_corr)
            ax_corr.set_title('Level Correlation Matrix')
            ax_corr.set_xlabel('Level')
            ax_corr.set_ylabel('Level')
            
            # Add taxonomy visualization
            ax_tax = fig.add_subplot(gs[-1, 2:])
            self._plot_taxonomy_tree(ax_tax, target_classes)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_taxonomy_tree(self, 
                          ax: plt.Axes, 
                          target_classes: List[int]) -> None:
        """Plot taxonomy tree with highlighted classes."""
        def _add_node(x: float, y: float, 
                     level: int, class_idx: int, 
                     parent_x: Optional[float] = None, 
                     parent_y: Optional[float] = None) -> None:
            # Draw connection to parent
            if parent_x is not None:
                ax.plot([parent_x, x], [parent_y, y], 'k-', alpha=0.5)
            
            # Draw node
            is_target = class_idx == target_classes[level]
            color = 'red' if is_target else 'lightgray'
            ax.scatter(x, y, c=color, s=100)
            
            # Add label
            label = self.class_names[level][class_idx]
            ax.annotate(label, (x, y), xytext=(5, 5), 
                       textcoords='offset points')
        
        # Clear and set up axes
        ax.clear()
        ax.set_title('Taxonomy Structure')
        ax.axis('off')
        
        # Plot nodes level by level
        n_levels = len(self.class_names)
        for level in range(n_levels):
            n_classes = len(self.class_names[level])
            y = 1 - (level / (n_levels - 1))
            
            for class_idx in range(n_classes):
                x = (class_idx + 1) / (n_classes + 1)
                
                # Find parent if not root
                if level > 0:
                    parent_class = target_classes[level - 1]
                    parent_n_classes = len(self.class_names[level - 1])
                    parent_x = (parent_class + 1) / (parent_n_classes + 1)
                    parent_y = 1 - ((level - 1) / (n_levels - 1))
                else:
                    parent_x = parent_y = None
                
                _add_node(x, y, level, class_idx, parent_x, parent_y)

    def plot_activation_comparison(self,
                                images: List[np.ndarray],
                                level: int,
                                target_class: int,
                                save_path: Optional[str] = None) -> None:
        """
        Compare activations across multiple images for the same class.
        
        Args:
            images: List of input images
            level: Target hierarchy level
            target_class: Target class index
            save_path: Optional path to save visualization
        """
        n_images = len(images)
        fig = plt.figure(figsize=(4 * n_images, 12))
        gs = GridSpec(3, n_images, figure=fig)
        
        for i, image in enumerate(images):
            # Original image
            ax1 = fig.add_subplot(gs[0, i])
            ax1.imshow(image)
            ax1.set_title(f'Image {i+1}')
            ax1.axis('off')
            
            # Attention map
            ax2 = fig.add_subplot(gs[1, i])
            attention_map = self.cam.generate_capsule_cam(
                image[np.newaxis, ...], level, target_class)
            im = ax2.imshow(attention_map[0], cmap='jet')
            ax2.set_title('Attention Map')
            ax2.axis('off')
            
            # Overlay
            ax3 = fig.add_subplot(gs[2, i])
            overlay = self.cam.visualize_capsule_attention(
                image, level, target_class)
            ax3.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax3.set_title('Overlay')
            ax3.axis('off')
        
        plt.suptitle(f'Level {level} - Class: {self.class_names[level][target_class]}',
                    fontsize=16, y=1.02)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()