import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch
from typing import List, Dict, Tuple
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class SimplexETFVisualizer:
    def __init__(self, 
                model: tf.keras.Model,
                dataset: tf.data.Dataset,
                class_names: Dict[str, List[str]],
                taxonomy: List[List[List[float]]]):
        self.model = model
        self.dataset = dataset
        self.class_names = class_names
        self.taxonomy = taxonomy
        self.feature_extractor = None
        self.features = None
        self.labels = None
        
    def create_feature_extractor(self):
        htr_caps_layer = self.model.get_layer('hd_caps')
        self.feature_extractor = tf.keras.Model(
            inputs=self.model.input,
            outputs=htr_caps_layer.output
        )
        
    def extract_features(self):
        if self.feature_extractor is None:
            self.create_feature_extractor()
            
        features = []
        labels = []
        
        for x, y in self.dataset:
            batch_features = self.feature_extractor(x)
            features.append([f.numpy() for f in batch_features])
            labels.append([l.numpy() for l in y])
            
        self.features = [np.concatenate([batch[i] for batch in features], axis=0)
                        for i in range(len(features[0]))]
        self.labels = [np.concatenate([batch[i] for batch in labels], axis=0)
                      for i in range(len(labels[0]))]

    def compute_circle_positions(self, num_levels: int) -> List[float]:
        """Compute radii for nested circles"""
        base_radius = 1.0
        radius_step = 2.0
        return [base_radius + i * radius_step for i in range(num_levels)]  # Reverse order

    def get_parent_children_mapping(self, level: int) -> Dict[int, List[int]]:
        """Get mapping of parent indices to their children indices"""
        if level == 0:
            return {}
            
        parent_children = {}
        taxonomy_matrix = self.taxonomy[level-1]
        
        for parent_idx in range(len(taxonomy_matrix)):
            children = [i for i, is_child in enumerate(taxonomy_matrix[parent_idx]) if is_child == 1]
            if children:
                parent_children[parent_idx] = children
                
        return parent_children

    def compute_class_positions(self, level: int, radius: float, 
                            parent_positions: np.ndarray = None) -> np.ndarray:
        """Compute positions for classes at given level with equal spacing"""
        n_classes = len(self.class_names[f'level_{level}'])
        positions = np.zeros((n_classes, 2))
        
        if level == 0:
            # Root level - split into two halves with equal spacing
            angles = np.array([0, np.pi])  # transport at right, animal at left
            positions[:, 0] = radius * np.cos(angles)
            positions[:, 1] = radius * np.sin(angles)
        else:
            parent_children = self.get_parent_children_mapping(level)
            used_positions = set()  # Track used positions to avoid overlap
            
            # First, determine total arc length available for each parent
            total_children = sum(len(children) for children in parent_children.values())
            angle_per_child = 2 * np.pi / total_children
            
            # Process each parent's children
            current_angle = 0
            for parent_idx, children in sorted(parent_children.items()):
                parent_pos = parent_positions[parent_idx]
                parent_angle = np.arctan2(parent_pos[1], parent_pos[0])
                
                # Place children with equal spacing
                for child_idx in children:
                    angle = current_angle
                    positions[child_idx] = [
                        radius * np.cos(angle),
                        radius * np.sin(angle)
                    ]
                    current_angle += angle_per_child
        
        return positions

    def process_features_for_level(self, level_features: np.ndarray, 
                                 center_positions: np.ndarray,
                                 labels: np.ndarray) -> np.ndarray:
        """Process and position features relative to their class centers"""
        if len(level_features.shape) > 2:
            level_features = level_features.reshape(level_features.shape[0], -1)
        
        # Standardize features
        scaler = StandardScaler()
        features_std = scaler.fit_transform(level_features)
        
        # Initialize output positions
        n_samples = len(level_features)
        positions = np.zeros((n_samples, 2))
        
        # For each sample, calculate its relative position
        for i in range(n_samples):
            class_idx = np.argmax(labels[i])
            center_pos = center_positions[class_idx]
            
            # Calculate relative position based on feature values
            feature_norm = np.linalg.norm(features_std[i])
            relative_distance = 0.3 * feature_norm / np.max(np.linalg.norm(features_std, axis=1))
            
            # Add some randomness to avoid overlapping
            angle = np.random.uniform(0, 2*np.pi)
            offset = relative_distance * np.array([np.cos(angle), np.sin(angle)])
            
            positions[i] = center_pos + offset
            
        return positions

    def draw_connection(self, ax: plt.Axes, start: np.ndarray, end: np.ndarray,
                       color: str = 'gray', alpha: float = 0.3):
        """Draw a connection line between two points"""
        con = ConnectionPatch(
            xyA=start, xyB=end,
            coordsA='data', coordsB='data',
            axesA=ax, axesB=ax,
            color=color, alpha=alpha,
            linewidth=1.0
        )
        ax.add_artist(con)

    def add_label(self, ax: plt.Axes, position: np.ndarray, label: str):
        """Add a label with white background"""
        ax.text(position[0], position[1], label,
                horizontalalignment='center',
                verticalalignment='center',
                bbox=dict(facecolor='white',
                         alpha=0.7,
                         edgecolor='none',
                         pad=1))

    def visualize(self, figsize=(15, 15), point_size=20, center_size=200,
                 sample_alpha=0.3, save_path=None):
        """Create the complete visualization"""
        if self.features is None:
            self.extract_features()
            
        num_levels = len(self.features)
        radii = self.compute_circle_positions(num_levels)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw circles
        for radius in radii:
            circle = Circle((0, 0), radius, fill=False, 
                          color='gray', linestyle='--', alpha=0.5)
            ax.add_artist(circle)
        
        # Initialize color maps based on the number of classes
        num_classes = sum(len(names) for names in self.class_names.values())
        cmap = plt.cm.get_cmap('tab20', num_classes)
        colors = [cmap(i) for i in range(num_classes)]
        
        # Compute and store class positions for all levels
        class_positions = []
        for level in range(num_levels):
            positions = self.compute_class_positions(
                level, radii[level],
                class_positions[-1] if level > 0 else None
            )
            class_positions.append(positions)
        
        # Draw connections first (bottom layer)
        for level in range(num_levels - 1):
            parent_children = self.get_parent_children_mapping(level + 1)
            for parent_idx, children in parent_children.items():
                parent_pos = class_positions[level][parent_idx]
                for child_idx in children:
                    child_pos = class_positions[level + 1][child_idx]
                    self.draw_connection(ax, parent_pos, child_pos)
        
        # Draw features and class centers
        for level in range(num_levels):
            positions = class_positions[level]
            features_pos = self.process_features_for_level(
                self.features[level],
                positions,
                self.labels[level]
            )
            
            # Plot features
            for class_idx in range(len(self.class_names[f'level_{level}'])):
                color = colors[class_idx]
                mask = np.argmax(self.labels[level], axis=1) == class_idx
                
                # Plot samples
                ax.scatter(
                    features_pos[mask, 0],
                    features_pos[mask, 1],
                    c=[color], s=point_size, alpha=sample_alpha
                )
                
                # Plot class center
                ax.scatter(
                    positions[class_idx, 0],
                    positions[class_idx, 1],
                    c=[color], marker='*', s=center_size,
                    edgecolor='black', linewidth=1
                )
                
                # Add label
                self.add_label(
                    ax, positions[class_idx],
                    self.class_names[f'level_{level}'][class_idx]
                )
        
        # Set plot properties
        ax.set_aspect('equal')
        margin = 0.5
        ax.set_xlim([-max(radii)-margin, max(radii)+margin])
        ax.set_ylim([-max(radii)-margin, max(radii)+margin])
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.title('Hierarchical Class Structure with Feature Distributions')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
if __name__ == '__main__':
    
    # Create and use visualizer
    visualizer = SimplexETFVisualizer(
        model=model,
        dataset=ds_test.take(10),
        class_names={f'level_{i}':v for i,v in enumerate(dataset.labels_name)},
        taxonomy=dataset.taxonomy
    )

    fig, ax = visualizer.visualize(
        figsize=(30, 30),
        point_size=40,
        center_size=200,
        sample_alpha=0.9,
        save_path='hierarchy_visualization.png'
    )