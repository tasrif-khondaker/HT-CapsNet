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
        self.class_names = self._process_class_names(class_names)
        
    def _process_class_names(self, class_names):
        """Convert class names to dictionary format if provided as list"""
        if class_names is None:
            return {}
            
        # If class_names is already a dictionary with 'level_X' keys, return as is
        if isinstance(class_names, dict) and all(k.startswith('level_') for k in class_names.keys()):
            return class_names
            
        # If class_names is a list of lists, convert to dictionary
        if isinstance(class_names, list):
            return {f'level_{i}': {j: name for j, name in enumerate(level_names)} 
                   for i, level_names in enumerate(class_names)}
        
        return {}

    def get_class_name(self, level, class_idx):
        """Get class name for given level and index"""
        if not self.class_names:
            return f'L{level}_{class_idx}'
            
        level_names = self.class_names.get(f'level_{level}', {})
        if isinstance(level_names, dict):
            return level_names.get(class_idx, f'L{level}_{class_idx}')
        elif isinstance(level_names, list) and class_idx < len(level_names):
            return level_names[class_idx]
        return f'L{level}_{class_idx}'

    # [Previous methods remain unchanged: get_parent_children_mapping, 
    #  extract_features_and_predictions, reduce_dimensions]
    
    def get_parent_children_mapping(self, level):
        """Get mapping of parent indices to their children indices"""
        parent_children = {}
        
        # Get all nodes at current level
        current_level_nodes = [node for node in self.label_tree.all_nodes() 
                             if node.identifier.startswith(f'L{level}_')]
        
        # For each node, find its children
        for node in current_level_nodes:
            parent_idx = int(node.identifier.split('_')[1])
            children = self.label_tree.children(node.identifier)
            if children:
                children_idx = [int(child.identifier.split('_')[1]) for child in children]
                parent_children[parent_idx] = children_idx
                
        return parent_children
    
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

    def add_class_label(self, ax, position, label, fontsize=10):
        """Add a label with white background at the class center"""
        ax.text(position[0], position[1], label,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=fontsize,
                bbox=dict(facecolor='white',
                         alpha=0.7,
                         edgecolor='none',
                         pad=2))

    def plot_hierarchical_features(self, features, true_labels, predictions, 
                                 figsize=(15, 15), show_samples=True, 
                                 show_centers=True, show_connections=True,
                                 show_hulls=True, min_radius=0.3, max_radius=1.0):
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
                    
                    # Add class label at center
                    class_name = self.get_class_name(level, class_idx)
                    self.add_class_label(ax, center, class_name)
                
                # [Rest of the plotting code remains unchanged]
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
                        
                        for feat, pred_class in zip(misclass_features, pred_classes):
                            pred_angle = angles[pred_class]
                            pred_center = np.array([radius * np.cos(pred_angle), 
                                                  radius * np.sin(pred_angle)])
                            
                            scaled_feat = feat / (np.max(np.abs(feat)) + 1e-10) * sector_size
                            rotated_feat = np.array([
                                scaled_feat[0] * np.cos(angle) - scaled_feat[1] * np.sin(angle),
                                scaled_feat[0] * np.sin(angle) + scaled_feat[1] * np.cos(angle)
                            ])
                            
                            mid_point = center + 0.7 * (pred_center - center)
                            translated_feat = rotated_feat + mid_point
                            
                            ax.scatter(translated_feat[0], translated_feat[1],
                                     c=[color_maps[level][i]], marker='x', s=100, 
                                     linewidth=2, zorder=4)
                            
                            ax.plot([translated_feat[0], pred_center[0]],
                                  [translated_feat[1], pred_center[1]],
                                  'r--', alpha=0.3, zorder=2)
                    
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
        
        # Draw hierarchical connections with improved visibility
        if show_connections:
            # Draw connections from inner to outer levels
            for level in range(n_levels - 1):
                parent_children = self.get_parent_children_mapping(level)
                
                for parent_idx, children in parent_children.items():
                    if parent_idx in centers[level]:
                        parent_center = centers[level][parent_idx]
                        parent_name = self.get_class_name(level, parent_idx)
                        
                        for child_idx in children:
                            if child_idx in centers[level + 1]:
                                child_center = centers[level + 1][child_idx]
                                child_name = self.get_class_name(level + 1, child_idx)
                                
                                # Draw connection line
                                ax.plot([parent_center[0], child_center[0]],
                                      [parent_center[1], child_center[1]],
                                      color='gray',
                                      linestyle=connection_style,
                                      alpha=connection_alpha,
                                      linewidth=connection_width,
                                      zorder=1)
                                
                                # Optionally add arrow to show direction
                                mid_point = (parent_center + child_center) / 2
                                dx = child_center[0] - parent_center[0]
                                dy = child_center[1] - parent_center[1]
                                ax.arrow(mid_point[0] - dx/4, mid_point[1] - dy/4,
                                       dx/8, dy/8,
                                       head_width=0.05,
                                       head_length=0.1,
                                       fc='gray',
                                       ec='gray',
                                       alpha=connection_alpha,
                                       zorder=1)
        
        # Add legend
        legend_elements = []
        for level in range(n_levels):
            unique_classes = np.unique(np.argmax(true_labels[level], axis=1))
            for i, class_idx in enumerate(unique_classes):
                class_name = self.get_class_name(level, class_idx)
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=color_maps[level][i],
                                                label=class_name,
                                                markersize=10))
        
        ax.legend(handles=legend_elements, loc='center left', 
                 bbox_to_anchor=(1, 0.5), title='Classes')
        
        plt.tight_layout()
        return fig, ax