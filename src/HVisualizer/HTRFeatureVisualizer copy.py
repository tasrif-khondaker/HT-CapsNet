import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import seaborn as sns

class HTRFeatureVisualizer:
    def __init__(self, model, label_tree, class_names=None, feature_dim=2):
        self.model = model
        self.label_tree = label_tree
        self.feature_dim = feature_dim
        self.pca = PCA(n_components=feature_dim)
        self.class_names = class_names or {}

    def extract_features_and_predictions(self, dataset):
        """Extract capsule features and model predictions"""
        # Find the HTR_Capsule layer
        htr_layer = None
        for layer in self.model.layers:
            if 'htr_capsule' in layer.__class__.__name__.lower() or 'hd_caps' in layer.name:
                htr_layer = layer
                break
                
        if htr_layer is None:
            raise ValueError("Could not find HTR_Capsule layer in the model")
        
        # Create intermediate model for features
        input_layer = self.model.get_layer(name='input_layer').input
        capsule_vectors = htr_layer.output
        intermediate_model = tf.keras.Model(inputs=input_layer, outputs=capsule_vectors)
        
        # Process batches
        features = []
        true_labels = []
        predictions = []
        
        for x, y in dataset:
            # Get features
            batch_features = intermediate_model.predict(x)
            if not isinstance(batch_features, list):
                batch_features = [batch_features]
                
            # Get predictions
            batch_preds = self.model.predict(x)
            if not isinstance(batch_preds, list):
                batch_preds = [batch_preds]
            
            if not features:
                features = [[] for _ in range(len(batch_features))]
                predictions = [[] for _ in range(len(batch_preds))]
                
            for i, bf in enumerate(batch_features):
                features[i].append(bf)
            for i, bp in enumerate(batch_preds):
                predictions[i].append(bp)
            true_labels.append(y)
            
        features = [np.concatenate(f) for f in features]
        predictions = [np.concatenate(p) for p in predictions]
        true_labels = [np.concatenate([l[i] for l in true_labels]) for i in range(len(true_labels[0]))]
        
        return features, true_labels, predictions
    
    def reduce_dimensions(self, features):
        """Reduce dimensionality of features for visualization"""
        reduced_features = []
        for level_features in features:
            orig_shape = level_features.shape
            reshaped = level_features.reshape(orig_shape[0], -1)
            reduced = self.pca.fit_transform(reshaped)
            reduced_features.append(reduced)
            print(f"Level {len(reduced_features)-1} explained variance ratio: {self.pca.explained_variance_ratio_}")
        return reduced_features

    def plot_hierarchical_features(self, features, true_labels, predictions, 
                                 figsize=(15, 15), show_samples=True, 
                                 show_centers=True, show_connections=True,
                                 show_hulls=True, min_radius=0.3, max_radius=1.0,
                                 show_class_names=True, 
                                 class_name_fontsize=8
                                 ):
        """Plot hierarchical features showing misclassifications"""
        fig, ax = plt.subplots(figsize=figsize)
        n_levels = len(features)
        
        # Define radii for each level (inner to outer)
        radius_step = (max_radius - min_radius) / (n_levels - 1) if n_levels > 1 else 0
        level_radii = [min_radius + i * radius_step for i in range(n_levels)]
        
        # Color maps for different levels
        color_maps = [plt.cm.tab20(np.linspace(0, 1, len(np.unique(np.argmax(true_labels[i], axis=1))))) 
                     for i in range(n_levels)]
        
        # Plot from outer to inner to ensure proper layering
        centers = []
        for level in range(n_levels-1, -1, -1):
            # Convert labels to indices
            true_level_labels = np.argmax(true_labels[level], axis=1)
            pred_level_labels = np.argmax(predictions[level], axis=1)
            unique_classes = np.unique(true_level_labels)
            n_classes = len(unique_classes)
            
            # Calculate angles for class centers
            angles = np.linspace(0, 2*np.pi, n_classes, endpoint=False)
            radius = level_radii[level]
            
            # Calculate and store class centers
            level_centers = {}
            for i, class_idx in enumerate(unique_classes):
                angle = angles[i]
                center = np.array([radius * np.cos(angle), radius * np.sin(angle)])
                level_centers[class_idx] = center
                
                if show_centers:
                    ax.scatter(center[0], center[1], c=[color_maps[level][i]], 
                             marker='*', s=200, edgecolor='black', zorder=5)
                                # Add class name if enabled
                    if show_class_names:
                        # Get class name from the dictionary if available, otherwise use default format
                        if f'level_{level}' in self.class_names and len(self.class_names[f'level_{level}']) > class_idx:
                            class_name = self.class_names[f'level_{level}'][class_idx]
                        else:
                            class_name = f'L{level}_{class_idx}'
                        
                        # Adjust label position slightly above the center point
                        label_pos = center + np.array([0, 0.1])  # Offset label slightly above center
                        self.add_class_label(ax, label_pos, class_name, class_name_fontsize)
                        
                # Plot samples if enabled
                if show_samples:
                    # Correct classifications
                    correct_mask = (true_level_labels == class_idx) & (true_level_labels == pred_level_labels)
                    if np.any(correct_mask):
                        class_features = features[level][correct_mask]
                        
                        # Scale and translate features
                        scaled_features = class_features / (np.max(np.abs(class_features)) + 1e-10)
                        sector_size = radius * 0.15
                        scaled_features = scaled_features * sector_size
                        rotated_features = np.column_stack([
                            scaled_features[:, 0] * np.cos(angle) - scaled_features[:, 1] * np.sin(angle),
                            scaled_features[:, 0] * np.sin(angle) + scaled_features[:, 1] * np.cos(angle)
                        ])
                        translated_features = rotated_features + center
                        
                        # Plot correct samples
                        ax.scatter(translated_features[:, 0], translated_features[:, 1],
                                 c=[color_maps[level][i]], alpha=0.6, s=30, zorder=3)
                    
                    # Misclassifications
                    misclass_mask = (true_level_labels == class_idx) & (true_level_labels != pred_level_labels)
                    if np.any(misclass_mask):
                        misclass_features = features[level][misclass_mask]
                        pred_classes = pred_level_labels[misclass_mask]
                        
                        # For each misclassified sample, draw it closer to predicted class
                        for feat, pred_class in zip(misclass_features, pred_classes):
                            pred_angle = angles[pred_class]
                            pred_center = np.array([radius * np.cos(pred_angle), 
                                                  radius * np.sin(pred_angle)])
                            
                            # Scale and position between true and predicted centers
                            scaled_feat = feat / (np.max(np.abs(feat)) + 1e-10) * sector_size
                            rotated_feat = np.array([
                                scaled_feat[0] * np.cos(angle) - scaled_feat[1] * np.sin(angle),
                                scaled_feat[0] * np.sin(angle) + scaled_feat[1] * np.cos(angle)
                            ])
                            
                            # Position sample between true and predicted centers
                            mid_point = center + 0.7 * (pred_center - center)  # Bias towards predicted class
                            translated_feat = rotated_feat + mid_point
                            
                            # Plot misclassified sample with special marking
                            ax.scatter(translated_feat[0], translated_feat[1],
                                     c=[color_maps[level][i]], marker='x', s=100, 
                                     linewidth=2, 
                                    #  edgecolor='red', 
                                     zorder=4)
                            
                            # Draw line to predicted center
                            ax.plot([translated_feat[0], pred_center[0]],
                                  [translated_feat[1], pred_center[1]],
                                  'r--', alpha=0.3, zorder=2)
                    
                    # Plot convex hull if enabled
                    if show_hulls:
                        class_features = features[level][true_level_labels == class_idx]
                        if len(class_features) >= 3:
                            scaled_features = class_features / (np.max(np.abs(class_features)) + 1e-10)
                            scaled_features = scaled_features * sector_size
                            rotated_features = np.column_stack([
                                scaled_features[:, 0] * np.cos(angle) - scaled_features[:, 1] * np.sin(angle),
                                scaled_features[:, 0] * np.sin(angle) + scaled_features[:, 1] * np.cos(angle)
                            ])
                            translated_features = rotated_features + center
                            
                            try:
                                hull = ConvexHull(translated_features)
                                for simplex in hull.simplices:
                                    ax.plot(translated_features[simplex, 0], 
                                          translated_features[simplex, 1],
                                          'k-', alpha=0.3, zorder=2)
                            except:
                                continue
            
            centers.insert(0, level_centers)
            
            # Draw circle for this level
            circle = plt.Circle((0, 0), radius, fill=False, linestyle='--', 
                              color='gray', alpha=0.5, zorder=1)
            ax.add_artist(circle)
            
            # Add level label with accuracy
            angle = -np.pi/4
            label_x = (radius + 0.05) * np.cos(angle)
            label_y = (radius + 0.05) * np.sin(angle)
            accuracy = np.mean(true_level_labels == pred_level_labels) * 100
            ax.text(label_x, label_y, f'Level {level}\nAcc: {accuracy:.1f}%', 
                   horizontalalignment='left', verticalalignment='bottom',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Draw parent-child connections
        if show_connections:
            self._draw_hierarchy_connections(centers, ax)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Update legend to use class names
        legend_elements = []
        for level in range(n_levels):
            unique_classes = np.unique(np.argmax(true_labels[level], axis=1))
            for i, class_idx in enumerate(unique_classes):
                if f'level_{level}' in self.class_names and len(self.class_names[f'level_{level}']) > class_idx:
                    label = self.class_names[f'level_{level}'][class_idx]
                else:
                    label = f'L{level}_{class_idx}'
                    
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=color_maps[level][i],
                                                label=label,
                                                markersize=10))
        
        ax.legend(handles=legend_elements, loc='center left', 
                bbox_to_anchor=(1, 0.5), title='Classes')
        
        plt.tight_layout()
        return fig, ax
        
    def _draw_hierarchy_connections(self, centers, ax):
        """Draw connections between parent and child classes using parent-child mapping"""
        for level in range(1, len(centers)):
            # Get parent-child mappings for this level
            parent_children = self.get_parent_children_mapping(level-1)
            
            # Draw connections for each parent to its children
            for parent_idx, children in parent_children.items():
                if parent_idx in centers[level-1]:  # Check if parent center exists
                    parent_center = centers[level-1][parent_idx]
                    
                    # Draw lines to each child
                    for child_idx in children:
                        if child_idx in centers[level]:  # Check if child center exists
                            child_center = centers[level][child_idx]
                            ax.plot([parent_center[0], child_center[0]],
                                [parent_center[1], child_center[1]],
                                'k-', alpha=0.2, zorder=1)

    def get_parent_children_mapping(self, level: int) -> dict:
        """
        Get mapping of parent indices to their children indices based on taxonomy structure.
        
        Args:
            level: The parent level to get mappings for
            
        Returns:
            Dictionary mapping parent indices to lists of their children indices
        """
        parent_children = {}
        
        # Get all nodes at current level
        parent_nodes = [node for node in self.label_tree.all_nodes() 
                    if node.identifier.startswith(f'L{level}_')]
        
        # For each parent node, find its children
        for parent_node in parent_nodes:
            parent_idx = int(parent_node.identifier.split('_')[1])
            children = self.label_tree.children(parent_node.identifier)
            
            if children:
                # Extract child indices from identifiers
                children_idx = [int(child.identifier.split('_')[1]) for child in children]
                parent_children[parent_idx] = children_idx
                
        return parent_children
    
    def add_class_label(self, ax, pos, label, fontsize=8):
        """Add a label with white background at the specified position.
        
        Args:
            ax: Matplotlib axis
            pos: (x, y) position for the label
            label: Text to display
            fontsize: Font size for the label
        """
        ax.text(pos[0], pos[1], label,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=fontsize,
                bbox=dict(facecolor='white',
                        alpha=0.7,
                        edgecolor='none',
                        pad=1))