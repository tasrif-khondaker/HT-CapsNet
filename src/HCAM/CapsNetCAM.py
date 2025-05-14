import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def visualize_capsnet_cam(model, image, layer_name='hd_caps', alpha=0.4):
    """
    Generate and visualize Class Activation Maps for HTRCapsNet model.
    
    Args:
        model: HTRCapsNet model
        image: Input image of shape (height, width, channels)
        layer_name: Name of the capsule layer to visualize
        alpha: Transparency factor for heatmap overlay
    
    Returns:
        Dictionary containing original image, heatmaps, and overlaid visualizations
        for each hierarchical branch
    """
    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Create intermediate model to get capsule layer outputs
    capsule_layer = model.get_layer(layer_name)
    intermediate_model = tf.keras.Model(inputs=model.input, 
                                      outputs=capsule_layer.output)
    
    # Get capsule outputs for the image
    # Make the tape persistent to allow multiple gradient calculations
    with tf.GradientTape(persistent=True) as tape:
        input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        tape.watch(input_tensor)
        capsule_outputs = intermediate_model(input_tensor)
    
    results = {}
    original_img = np.squeeze(image)
    results['original'] = original_img
    
    # Process each hierarchical branch
    for idx, capsule_output in enumerate(capsule_outputs):
        # Calculate gradients with respect to input
        grads = tape.gradient(capsule_output, input_tensor)
        
        # Calculate magnitude of capsule vectors
        capsule_magnitudes = tf.norm(capsule_output, axis=-1)
        
        # Get the index of the maximum magnitude for each sample
        pred_class = tf.argmax(capsule_magnitudes, axis=-1)
        
        # Get gradients for predicted class
        class_grads = grads[0]  # Remove batch dimension
        
        # Take absolute value and max across channels
        cam = np.abs(class_grads)
        cam = np.max(cam, axis=-1)
        
        # Normalize heatmap
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-10)
        
        # Resize heatmap to match input image size
        cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        
        # Apply colormap
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Prepare original image for overlay
        if len(original_img.shape) == 2:
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        else:
            # Ensure the image is in the correct format and range
            original_rgb = np.uint8(original_img * 255 if original_img.max() <= 1.0 else original_img)
            if original_rgb.shape[-1] == 3:  # If RGB
                original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        
        # Ensure both images are uint8 and have the same number of channels
        original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert to uint8 if not already
        original_rgb = np.uint8(original_rgb)
        heatmap = np.uint8(heatmap)
        
        # Overlay heatmap on original image
        overlaid = cv2.addWeighted(original_rgb, 1 - alpha, heatmap, alpha, 0)

        # overlaid = heatmap * 0.1 + original_img
        
        # Store results
        branch_results = {
            'heatmap': heatmap,
            'overlaid': overlaid,
            'cam_raw': cam,
            'capsule_shape': capsule_output.shape
        }
        results[f'branch_{idx}'] = branch_results
    
    # Delete the tape to free memory
    del tape
    
    return results

def plot_capsnet_visualizations(results, figsize=(10, 10)):
    """
    Plot the CAM visualizations for all branches.
    """
    # Count number of branches
    n_branches = len([k for k in results.keys() if k.startswith('branch')])
    
    # Plot original image
    # plt.title('Original Image')
    print('Original Image')
    plt.figure(figsize=figsize)
    plt.imshow(results['original'])
    plt.axis('off')
    plt.show()
    
    # Plot each branch's visualization
    for i in range(n_branches):
        print(f'Level {i+1} CAM Overlay\nCapsule Shape: {results[f"branch_{i}"]["capsule_shape"]}')
        plt.figure(figsize=figsize)
        branch_results = results[f'branch_{i}']
        plt.imshow(branch_results['overlaid'])
        # plt.title(f'Level {i+1} CAM Overlay\nCapsule Shape: {branch_results["capsule_shape"]}')
        plt.axis('off')
        plt.show()
        
    return None

def visualize_capsnet_cam_bad(model, image, layer_name='hd_caps', alpha=0.3):
    """
    Generate and visualize Class Activation Maps for HTRCapsNet model.
    
    Args:
        model: HTRCapsNet model
        image: Input image of shape (height, width, channels)
        layer_name: Name of the capsule layer to visualize
        alpha: Transparency factor for heatmap overlay
    
    Returns:
        Dictionary containing original image, heatmaps, and overlaid visualizations
        for each hierarchical branch
    """
    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Create intermediate model to get capsule layer outputs
    capsule_layer = model.get_layer(layer_name)
    intermediate_model = tf.keras.Model(inputs=model.input, 
                                      outputs=capsule_layer.output)
    
    # Get capsule outputs for the image
    # Make the tape persistent to allow multiple gradient calculations
    with tf.GradientTape(persistent=True) as tape:
        input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        tape.watch(input_tensor)
        capsule_outputs = intermediate_model(input_tensor)
    
    results = {}
    original_img = np.squeeze(image)
    results['original'] = original_img
    
    # Process each hierarchical branch
    for idx, capsule_output in enumerate(capsule_outputs):
        # Calculate gradients with respect to input
        grads = tape.gradient(capsule_output, input_tensor)
        
        # Calculate magnitude of capsule vectors
        capsule_magnitudes = tf.norm(capsule_output, axis=-1)
        
        # Get the index of the maximum magnitude for each sample
        pred_class = tf.argmax(capsule_magnitudes, axis=-1)
        
        # Get gradients for predicted class
        class_grads = grads[0]  # Remove batch dimension
        
        # Take absolute value and max across channels
        cam = np.abs(class_grads)
        cam = np.max(cam, axis=-1)

        # Normalize heatmap
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-10)
        
        # Resize heatmap to match input image size
        cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        
        # Apply colormap
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Prepare original image for overlay
        if len(original_img.shape) == 2:
            original_rgb = cv2.cvtColor(original_img, cv2.COLORMAP_HSV)
        else:
            # Ensure the image is in the correct format and range
            original_rgb = np.uint8(original_img * 255 if original_img.max() <= 1.0 else original_img)
            if original_rgb.shape[-1] == 3:  # If RGB
                original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        
        # Ensure both images are uint8 and have the same number of channels
        original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert to uint8 if not already
        original_rgb = np.uint8(original_rgb)
        heatmap = np.uint8(heatmap)
        
        # Overlay heatmap on original image
        overlaid = cv2.addWeighted(original_rgb, 1 - alpha, heatmap, alpha, 0)

        # overlaid = heatmap * 0.1 + original_img
        
        # Store results
        branch_results = {
            'heatmap': heatmap,
            'overlaid': overlaid,
            'cam_raw': cam,
            'capsule_shape': capsule_output.shape
        }
        results[f'branch_{idx}'] = branch_results
    
    # Delete the tape to free memory
    del tape
    
    return results