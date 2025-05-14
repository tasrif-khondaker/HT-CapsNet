import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
import seaborn as sns

class HTRFeatureVisualizer:
    def __init__(self, model, label_tree, class_names=None, feature_dim=2):
        self.model = model
        self.label_tree = label_tree
        self.feature_dim = feature_dim
        self.pca = PCA(n_components=feature_dim)
        self.umap = umap.UMAP()
        self.tsne = TSNE(n_components=feature_dim)
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
    
    def reduce_dimensions(self, features, method='pca'):
        """Reduce dimensionality of features for visualization"""
        reduced_features = []
        for level_features in features:
            orig_shape = level_features.shape
            reshaped = level_features.reshape(orig_shape[0], -1)
            if method == 'pca':
                reduced = self.pca.fit_transform(reshaped)
            elif method == 'umap':
                reduced = self.umap.fit_transform(reshaped)
            elif method == 'tsne':
                reduced = self.tsne.fit_transform(reshaped)
            reduced = self.pca.fit_transform(reshaped)
            reduced_features.append(reduced)
            print(f"Level {len(reduced_features)-1} explained variance ratio: {self.pca.explained_variance_ratio_}")
        return reduced_features

    def plot_hierarchical_features(self, features, true_labels, predictions, 
                                 figsize=(15, 15), show_samples=True, 
                                 show_centers=True, show_connections=True,
                                 show_hulls=True, min_radius=0.3, max_radius=1.0,
                                 show_class_names=True,
                                 show_levels=True, show_level_acc=False,
                                 show_class_legend=False,
                                 show_lines_to_true_center=False, 
                                 show_lines_to_pred_center=True,
                                 class_legend_ncol=1,
                                 class_name_fontsize=8
                                 ):
        """Plot hierarchical features showing misclassifications"""
        fig, ax = plt.subplots(figsize=figsize)
        n_levels = len(features)
        
        # Define radii for each level (inner to outer)
        radius_step = (max_radius - min_radius) / (n_levels - 1) if n_levels > 1 else 0
        level_radii = [min_radius + i * radius_step for i in range(n_levels)]
        
        # Color maps for different levels
        color_maps = []
        for level in range(n_levels):
            level_classes = self.get_level_classes(level)
            n_classes = len(level_classes)
            color_maps.append(plt.cm.tab20(np.linspace(0, 1, n_classes)))
        
        # Plot from outer to inner to ensure proper layering
        centers = []
        for level in range(n_levels-1, -1, -1):
            level_classes = self.get_level_classes(level)
            # Convert labels to indices
            true_level_labels = np.argmax(true_labels[level], axis=1)
            pred_level_labels = np.argmax(predictions[level], axis=1)
            
            # Calculate angles for class centers, grouping children by parent
            class_angles = self.calculate_class_angles(level, len(level_classes), level_classes)
            radius = level_radii[level]

            sector_size = radius * 0.15

            # Calculate and store class centers
            level_centers = {}
            for i, class_idx in enumerate(level_classes):
                angle = class_angles[class_idx]
                center = np.array([radius * np.cos(angle), radius * np.sin(angle)])
                level_centers[class_idx] = center
                
                if show_centers:
                    # Get class name from the dictionary if available, otherwise use default format
                    if f'level_{level}' in self.class_names and len(self.class_names[f'level_{level}']) > class_idx:
                        class_name = self.class_names[f'level_{level}'][class_idx]
                    else:
                        class_name = f'L{level}_{class_idx}'

                    ax.scatter(center[0], center[1], c=[color_maps[level][i]], 
                            marker='*', s=200, edgecolor='black', zorder=5,
                            label=class_name)

                    # Add class name if enabled
                    if show_class_names:
                        # Add label with consistent positioning
                        self.add_class_label(ax, center, angle, class_name, radius, class_name_fontsize)
                    
                        
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
                            pred_angle = class_angles[pred_class]
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
                            if show_lines_to_pred_center:
                                ax.plot([translated_feat[0], pred_center[0]],
                                    [translated_feat[1], pred_center[1]],
                                    'r--', alpha=0.3, zorder=2)
                            
                                # Draw line to True center
                            if show_lines_to_true_center:
                                ax.plot([translated_feat[0], center[0]],
                                    [translated_feat[1], center[1]],
                                    'g--', alpha=0.3, zorder=2)
                
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
            circle = plt.Circle((0, 0), radius, fill=False, linestyle='-', 
                              color='gray', alpha=0.5, zorder=1)
            ax.add_artist(circle)
            
            # Add level label with accuracy
            if show_levels:
                angle = -np.pi/4
                label_x = (radius + 0.05) * np.cos(angle)
                label_y = (radius + 0.05) * np.sin(angle)
                if show_level_acc:
                    accuracy = np.mean(true_level_labels == pred_level_labels) * 100
                ax.text(label_x, label_y, f'Level {level}\nAcc: {accuracy:.1f}%' if show_level_acc else f'Level {level}', 
                    horizontalalignment='left', verticalalignment='bottom',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Draw parent-child connections
        if show_connections:
            self._draw_hierarchy_connections(centers, ax)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        if show_class_legend:
            ax.legend(
                        title='Class Names',
                        loc='center left',
                        bbox_to_anchor=(1, 0.5),  # Positions the legend to the right of the plot
                        ncol=class_legend_ncol,  # Initially set to 1 column; it will adjust if needed
                        frameon=True
                        )   
        
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
    
    def add_class_label(self, ax, center, angle, label, radius, fontsize=8):
        """Add a label with white background at a position relative to the center point.
        
        Args:
            ax: Matplotlib axis
            center: (x, y) center point of the class
            angle: Angle of the center point (in radians)
            label: Text to display
            radius: Radius of the current level circle
            fontsize: Font size for the label
        """
        # Use a consistent offset relative to the radius
        offset_ratio = 0.05  # 20% of radius
        offset = radius * offset_ratio
        
        # Calculate label position with consistent radius
        label_x = center[0] + np.cos(angle) * offset
        label_y = center[1] + np.sin(angle) * offset
        
        # Determine text alignment based on position relative to center
        # This ensures text flows away from the circle on both sides
        if label_x >= 0:
            ha = 'left'
        else:
            ha = 'right'
        
        # Add the text with no rotation (horizontal)
        ax.text(label_x, label_y, label,
                horizontalalignment=ha,
                verticalalignment='center',
                rotation=0,  # Always horizontal
                fontsize=fontsize,
                bbox=dict(facecolor='white',
                        alpha=0.7,
                        edgecolor='none',
                        pad=1))

    def calculate_class_angles(self, level, n_classes, true_level_labels):
        """Calculate angles for class centers, maintaining hierarchical grouping at all levels.
        
        Args:
            level: Current hierarchy level
            n_classes: Number of classes at this level
            true_level_labels: True labels for this level
            
        Returns:
            Dictionary mapping class indices to their angles
        """
        # Get all unique class indices from the true labels
        all_classes = set(range(max(np.max(true_level_labels) + 1, n_classes)))
        
        # Get hierarchically grouped classes
        grouped_classes = self.get_class_hierarchy_groups(level)
        
        # Collect all classes that are in groups
        classes_in_groups = set()
        for group in grouped_classes:
            classes_in_groups.update(group)
        
        # Find classes not in any group
        ungrouped_classes = all_classes - classes_in_groups
        
        # Calculate angles
        angles = {}
        total_angle = 2 * np.pi
        current_angle = 0
        
        # Calculate total segments needed
        total_segments = sum(len(group) for group in grouped_classes) + len(ungrouped_classes)
        angle_per_segment = total_angle / total_segments if total_segments > 0 else total_angle
        
        # Assign angles to grouped classes
        for group in grouped_classes:
            group_size = len(group)
            group_angle = group_size * angle_per_segment
            
            # Distribute classes within their group's angle range
            for i, class_idx in enumerate(group):
                class_angle = current_angle + (i + 0.5) * (group_angle / group_size)
                angles[class_idx] = class_angle
            
            current_angle += group_angle
        
        # Handle ungrouped classes
        if ungrouped_classes:
            remaining_angle = total_angle - current_angle
            angle_step = remaining_angle / len(ungrouped_classes)
            for i, class_idx in enumerate(sorted(ungrouped_classes)):
                angles[class_idx] = current_angle + i * angle_step
        
        # Final check to ensure all classes have angles
        for class_idx in all_classes:
            if class_idx not in angles:
                angles[class_idx] = current_angle
                current_angle += angle_per_segment
        
        return angles
    
    def get_class_hierarchy_groups(self, level):
        """Get hierarchically grouped classes for the given level.
        
        This method recursively groups classes based on their ancestors,
        ensuring classes with common ancestors stay together at all levels.
        
        Args:
            level: Current hierarchy level
            
        Returns:
            List of lists, where each inner list contains class indices that should be grouped together
        """
        if level == 0:
            # Root level - each class is its own group
            parent_nodes = [node for node in self.label_tree.all_nodes() 
                        if node.identifier.startswith(f'L{level}_')]
            return [[int(node.identifier.split('_')[1])] for node in parent_nodes]
        
        # Get groups from previous level
        parent_groups = self.get_class_hierarchy_groups(level - 1)
        
        # For each parent group, get its children while maintaining the grouping
        child_groups = []
        for parent_group in parent_groups:
            group_children = []
            for parent_idx in parent_group:
                parent_id = f'L{level-1}_{parent_idx}'
                if self.label_tree.contains(parent_id):
                    children = self.label_tree.children(parent_id)
                    if children:
                        # Add children indices to the group
                        group_children.extend([int(child.identifier.split('_')[1]) 
                                            for child in children])
            if group_children:
                child_groups.append(group_children)
        
        return child_groups
    
    def get_level_classes(self, level):
        """Get all possible classes at a given level based on the taxonomy tree.
        
        Args:
            level: Level in the hierarchy
            
        Returns:
            List of class indices at that level
        """
        # Get all nodes at this level from the tree
        level_nodes = [node for node in self.label_tree.all_nodes() 
                    if node.identifier.startswith(f'L{level}_')]
        
        # Extract class indices from node identifiers
        class_indices = [int(node.identifier.split('_')[1]) for node in level_nodes]
        
        return sorted(class_indices)