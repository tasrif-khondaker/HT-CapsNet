###################################################################################################################################
##   This file contains the functions for creating the hierarchical dataset. Currently, the following datasets are supported:   ## 
##                                                                                                                              ## 
##                                                           1. MNIST                                                           ## 
##                                                           2. EMNIST                                                          ## 
##                                                          3. CIFAR10                                                          ## 
##                                                          4. CIFAR100                                                         ## 
##                                                       5. Fashion-MNIST                                                       ## 
##                                                     6. CU-Birds_200_2011                                                     ## 
##                                                       7. Stanford Cars                                                       ## 
##                                                        8. Marine tree                                                        ## 
##                                                                                                                              ## 
##      For loading each dataset, you can use the corresponding function. Here are the functions for loading each dataset:      ## 
##                                                                                                                              ## 
##                                             1. MNIST(): Loads the MNIST dataset.                                             ## 
##                                            2. EMNIST(): Loads the EMNIST dataset.                                            ## 
##                                     3. Fashion_MNIST(): Loads the Fashion-MNIST dataset.                                     ## 
##                                          4. CIFAR-10(): Loads the CIFAR-10 dataset.                                          ## 
##                                         5. CIFAR_100(): Loads the CIFAR-100 dataset.                                         ## 
##                                 6. CU_Birds_200_2011(): Loads the CU-Birds_200_2011 dataset.                                 ## 
##                                     7. Stanford_Cars(): Loads the Stanford Cars dataset.                                     ## 
##                                       8. Marine_Tree(): Loads the Marine tree dataset.                                       ## 
##                                                                                                                              ## 
##                              Each dataset function supports a different number of hierarchies.                               ## 
##                                    For example, MNIST() and EMNIST() have 2 hierarchies,                                     ## 
##                             while Fashion_MNIST(), CIFAR-10(), CIFAR_100(), CU_Birds_200_2011(),                             ## 
##                           and Stanford_Cars() have 3 hierarchies. Marine_Tree() has 5 hierarchies.                           ## 
##                                                                                                                              ## 
## You can adjust the number of hierarchies by modifying the value of the "number_of_hierarchy_levels" variable in the function.## 
##                Please refer to the function descriptions for more details on how to use each dataset function.               ## 
##                                                                                                                              ## 
##                                                Author: Khondaker Tasrif Noor.                                                ## 
###################################################################################################################################

# Import libraries
import tensorflow as tf
from tensorflow import keras
from treelib import Tree
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import random
import warnings

def split_data(x_train, labels, validation_split=0.2):
    '''
    Split data into training and validation sets
    :param x_train: training data
    :param labels: labels for training data 
    '''
    num_samples = x_train.shape[0]
    num_validation_samples = int(validation_split * num_samples)

    # Randomly shuffle the data and labels
    indices = np.random.permutation(num_samples)
    x_train = x_train[indices]
    labels = [label[indices] for label in labels]

    # Split the data and labels into training and validation sets
    x_val = x_train[:num_validation_samples]
    x_train = x_train[num_validation_samples:]
    labels_val = [label[:num_validation_samples] for label in labels]
    labels_train = [label[num_validation_samples:] for label in labels]

    return x_train, x_val, labels_train, labels_val

def make_hierarchical_tree(taxonomy, labels=None):
    """
    This method draws the taxonomy using the graphviz library.
    :return: Digraph of the label tree.
    :rtype: Digraph
    :input:
             :taxonomy: List of label maps for each Parent-Child relationship. For N levels, there will be N-1 label maps, since the last level does not have any child.
             :labels: List of label names for each level. For N levels, there will be N label names. If None, then the labels will be named as L0, L1, L2, etc.
     """
    tree = Tree()
    tree.create_node("Root", "root")  # root node
    
    if labels is not None:
        # Check if the number of labels is equal to the number of taxonomies
        # If the length of the labels is 1.
        if len(labels) == 1:
            assert len(labels[0]) == len(taxonomy[0]), f'Number of labels and taxonomies do not match in level 0.\n The number of labels in level 0 is {len(labels[0])} and the number of taxonomies in level 0 is {len(taxonomy[0])}.'
        # If the length of the labels is more than 1.
        if len(labels) > 1:
            assert len(labels) == len(taxonomy)+1, "Number of labels and taxonomies do not match."
            # Check the number of labels in each level is equal to the number of labels in the taxonomy
            for i in range(len(labels)-1):
                assert len(labels[i]) == len(taxonomy[i]), f'Number of labels and taxonomies do not match in level {i}.\n The number of labels in level {i} is {len(labels[i])} and the number of taxonomies in level {i} is {len(taxonomy[i])}.'
                assert len(labels[i+1]) == len([list(row) for row in zip(*taxonomy[i])]), f'Number of labels and taxonomies do not match in level {i+1}. \n The number of labels in level {i+1} is {len(labels[i+1])} and the number of taxonomies in level {i+1} is {len([list(row) for row in zip(*taxonomy[i])])}.'

        # if len(taxonomy) > 1:
        for i in range(len(taxonomy[0])):
            tree.create_node(labels[0][i] + ' -> (L0_' + str(i) + ')', 'L0_' + str(i), parent="root")

        for l in range(len(taxonomy)):
            for i in range(len(taxonomy[l])):
                for j in range(len(taxonomy[l][i])):
                    if taxonomy[l][i][j] == 1:
                        tree.create_node(labels[l + 1][j] + ' -> (L' + str(l + 1) + '_' + str(j) + ')',
                                        'L' + str(l + 1) + '_' + str(j),
                                        parent='L' + str(l) + '_' + str(i))
        # if len(taxonomy) == 1:
        #     for i in range(len(labels[0])):
        #         tree.create_node(labels[0][i] + ' -> (L0_' + str(i) + ')', 'L0_' + str(i), parent="root")

    elif labels is None:
        # if len(taxonomy) > 1:
        for i in range(len(taxonomy[0])):
            tree.create_node('-> (L0_' + str(i) + ')', 'L0_' + str(i), parent="root")

        for l in range(len(taxonomy)):
            for i in range(len(taxonomy[l])):
                for j in range(len(taxonomy[l][i])):
                    if taxonomy[l][i][j] == 1:
                        tree.create_node('-> (L' + str(l + 1) + '_' + str(j) + ')',
                                        'L' + str(l + 1) + '_' + str(j),
                                        parent='L' + str(l) + '_' + str(i))
        # if len(taxonomy) == 1:
        #     for i in range(len(taxonomy[0])):
        #         tree.create_node('-> (L0_' + str(i) + ')', 'L0_' + str(i), parent="root")


    return tree

def Normalize_StandardScaler(data):
    '''
    This function computes the z-score normalization. Function for Standardization.
     z = (x - u) / s
        where u is the mean of the training samples. s is the standard deviation of the training samples.
    Scales each input variable separately by subtracting the mean (called centering) and dividing by the standard deviation to shift the distribution to have a mean of zero and a standard deviation of one.
    '''
    data = tf.cast(data, dtype=tf.float32) # Convert to float 32
    mean = tf.math.reduce_mean(data)
    std = tf.math.reduce_std(data)
    data = tf.subtract(data, mean)
    data = tf.divide(data, std)
    return data

def Normalize_MinMaxScaler(data):
    '''Function for Data Normalization.
    Normalization is a rescaling of the data from the original range so that all values are within the new range of 0 and 1.
    '''
    data = tf.cast(data, dtype=tf.float32) # Convert to float 32
    min = tf.math.reduce_min(data)
    max = tf.math.reduce_max(data)
    data = tf.subtract(data, min)
    data = tf.divide(data, max-min)
    return data

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

def mix_up(ds_one, ds_two, alpha=0.2):
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Ensure all labels have the same dtype (convert to float32 for consistency)
    labels_one = [tf.cast(label, tf.float32) for label in labels_one]
    labels_two = [tf.cast(label, tf.float32) for label in labels_two]
    y_l = tf.cast(y_l, tf.float32)

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = [i for i in range(len(labels_one))]
    for i in range(len(labels_one)):
        labels[i] = labels_one[i] * y_l + labels_two[i] * (1 - y_l)
        
    return (images, tuple(labels)) 

def H_MixUp(ds_one, ds_two, alpha=0.2):
    '''
    Perform Hierarchical Mixup on the datasets.
    - The labels are assumed to be a list of one-hot encoded vector for different hierarchical labels. i.e. [[level-1_labels], [level-2_labels], [level-3_labels], ..., [level-N_labels]]
    - The logic is to perform mixup only for samples where level-N labels are different and level-N-1 (previous level is the hierarchy) labels are the same.
    :Input:
        ds_one: Tuple, # Tuple of images and labels from dataset one
        ds_two: Tuple, # Tuple of images and labels from dataset two
        alpha: float, # Alpha value for mixup. Default 0.2
    :Returns:
        images: Tensor, # Images after mixup
        labels: Tuple, # Labels after mixup

    '''
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # Sanity check: Ensure that the number of levels in the labels are at least 2
    assert len(labels_one) >= 2, "The number of levels in the labels must be at least 2"
    assert len(labels_two) >= 2, "The number of levels in the labels must be at least 2"

    # Find the best matches for each sample in images_one from images_two
    levelN_1_labels_one = tf.argmax(labels_one[-2], axis=1)
    levelN_1_labels_two = tf.argmax(labels_two[-2], axis=1)
    levelN_labels_one = tf.argmax(labels_one[-1], axis=1)
    levelN_labels_two = tf.argmax(labels_two[-1], axis=1)

    # Create a mask for level N-1 matches and level N mismatches
    levelN_1_match_mask = tf.equal(tf.expand_dims(levelN_1_labels_one, axis=1), tf.expand_dims(levelN_1_labels_two, axis=0))
    levelN_mismatch_mask = tf.not_equal(tf.expand_dims(levelN_labels_one, axis=1), tf.expand_dims(levelN_labels_two, axis=0))

    # Combine the masks to find valid pairs
    valid_pairs_mask = tf.logical_and(levelN_1_match_mask, levelN_mismatch_mask)

    # Find the best matches for each sample in images_one
    best_matches = tf.argmax(tf.cast(valid_pairs_mask, tf.int32), axis=1)

    # Rearrange images_two and labels_two based on the best matches
    images_two = tf.gather(images_two, best_matches)
    labels_two = [tf.gather(label, best_matches) for label in labels_two]

    # Check if level 1 labels are the same and level N labels are different
    levelN_1_match = tf.equal(tf.argmax(labels_one[-2], axis=1), tf.argmax(labels_two[-2], axis=1))
    levelN_mismatch = tf.not_equal(tf.argmax(labels_one[-1], axis=1), tf.argmax(labels_two[-1], axis=1))

    # Perform mixup only for samples where levelN_1_match is true and levelN_mismatch is true
    perform_mixup = tf.logical_and(levelN_1_match, levelN_mismatch)

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = tf.where(tf.reshape(perform_mixup, (batch_size, 1, 1, 1)), images_one * x_l + images_two * (1 - x_l), images_one)


    # labels = [tf.where(tf.reshape(perform_mixup, (batch_size, 1)), labels_one[i] * y_l + labels_two[i] * (1 - y_l), labels_one[i]) for i in range(len(labels_one))]
    # Ensure y_l is cast to float32
    y_l = tf.cast(y_l, tf.float32)
    labels = [
        tf.where(
            tf.reshape(perform_mixup, (batch_size, 1)),
            tf.cast(labels_one[i], tf.float32) * y_l + tf.cast(labels_two[i], tf.float32) * (1 - y_l),
            tf.cast(labels_one[i], tf.float32)  # Ensure this is also casted to float32
        ) for i in range(len(labels_one))
    ]

    return (images, tuple(labels))

def get_box(lambda_value,IMG_SIZE):
    '''
    This function returns the bounding box coordinates for the cutout.
    ::input::
    lambda_value: float, hyperparameter for the cutout size.
    ::return::
    boundaryx1: int, x coordinate for the top left corner of the cutout.
    boundaryy1: int, y coordinate for the top left corner of the cutout.
    target_h: int, height of the cutout.
    target_w: int, width of the cutout.
    
    '''
    cut_rat = tf.sqrt(1.0 - lambda_value)

    cut_w = IMG_SIZE * cut_rat  # rw
    cut_w = tf.cast(cut_w, "int32")

    cut_h = IMG_SIZE * cut_rat  # rh
    cut_h = tf.cast(cut_h, "int32")

    cut_x = tf.random.uniform((1,), minval=0, maxval=IMG_SIZE)  # rx
    cut_x = tf.cast(cut_x, "int32")
    cut_y = tf.random.uniform((1,), minval=0, maxval=IMG_SIZE)  # ry
    cut_y = tf.cast(cut_y, "int32")

    boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, IMG_SIZE)
    boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, IMG_SIZE)
    bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, IMG_SIZE)
    bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, IMG_SIZE)

    target_h = bby2 - boundaryy1
    if target_h == 0:
        target_h += 1

    target_w = bbx2 - boundaryx1
    if target_w == 0:
        target_w += 1

    return boundaryx1, boundaryy1, target_h, target_w

def cutmix(train_ds_one, train_ds_two, alpha=0.2):
    '''
    This function performs the CutMix augmentation on the datasets.
    - The labels are assumed to be a list of one-hot encoded vector for different hierarchical labels. i.e. [[level-1_labels], [level-2_labels], [level-3_labels], ..., [level-N_labels]]    
    '''
    images1, labels1 = train_ds_one
    images2, labels2 = train_ds_two
    batch_size = tf.shape(images1)[0]
    IMG_SIZE = images1.shape[1]

    lambda_values = sample_beta_distribution(batch_size, alpha, alpha)

    def mix_images(i):
        boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_values[i], IMG_SIZE)

        crop2 = tf.image.crop_to_bounding_box(images2[i], boundaryy1, boundaryx1, target_h, target_w)
        image2 = tf.image.pad_to_bounding_box(crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)
        crop1 = tf.image.crop_to_bounding_box(images1[i], boundaryy1, boundaryx1, target_h, target_w)
        img1 = tf.image.pad_to_bounding_box(crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)

        image1 = images1[i] - img1
        mixed_image = image1 + image2

        lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
        lambda_value = tf.cast(lambda_value, tf.float32)

        mixed_label = [labels1[j][i] * lambda_value + labels2[j][i] * (1 - lambda_value) for j in range(len(labels1))]

        return mixed_image, mixed_label

    mixed_images, mixed_labels = tf.map_fn(
        mix_images,
        tf.range(batch_size),
        fn_output_signature=(tf.float32, [tf.float32] * len(labels1))
    )

    mixed_labels = [tf.stack(mixed_label) for mixed_label in mixed_labels]

    return mixed_images, tuple(mixed_labels)

def H_CutMix(train_ds_one, train_ds_two, alpha=0.2):
    '''
    Perform Hierarchical Cutmix on the datasets.
    - The labels are assumed to be a list of one-hot encoded vector for different hierarchical labels. i.e. [[level-1_labels], [level-2_labels], [level-3_labels], ..., [level-N_labels]]
    - The logic is to perform cutmix only for samples where level-N labels are different and level-N-1 (previous level is the hierarchy) labels are the same.
    :Input:
        ds_one: Tuple, # Tuple of images and labels from dataset one
        ds_two: Tuple, # Tuple of images and labels from dataset two
        alpha: float, # Alpha value for cutmix. Default 0.2
    :Returns:
        images: Tensor, # Images after cutmix
        labels: Tuple, # Labels after cutmix

    '''
    images1, labels1 = train_ds_one
    images2, labels2 = train_ds_two
    batch_size = tf.shape(images1)[0]
    IMG_SIZE = images1.shape[1]

    # Sanity check: Ensure that the number of levels in the labels are at least 2
    assert len(labels1) >= 2, "The number of levels in the labels must be at least 2"
    assert len(labels2) >= 2, "The number of levels in the labels must be at least 2"

    # Find the best matches for each sample in images_one from images_two
    levelN_1_labels_one = tf.argmax(labels1[-2], axis=1)
    levelN_1_labels_two = tf.argmax(labels2[-2], axis=1)
    levelN_labels_one = tf.argmax(labels1[-1], axis=1)
    levelN_labels_two = tf.argmax(labels2[-1], axis=1)

    # Create a mask for level N-1 matches and level N mismatches
    levelN_1_match_mask = tf.equal(tf.expand_dims(levelN_1_labels_one, axis=1), tf.expand_dims(levelN_1_labels_two, axis=0))
    levelN_mismatch_mask = tf.not_equal(tf.expand_dims(levelN_labels_one, axis=1), tf.expand_dims(levelN_labels_two, axis=0))

    # Combine the masks to find valid pairs
    valid_pairs_mask = tf.logical_and(levelN_1_match_mask, levelN_mismatch_mask)

    # Find the best matches for each sample in images_one
    best_matches = tf.argmax(tf.cast(valid_pairs_mask, tf.int32), axis=1)

    # Rearrange images_two and labels_two based on the best matches
    images2 = tf.gather(images2, best_matches)
    labels2 = [tf.gather(label, best_matches) for label in labels2]

    lambda_values = sample_beta_distribution(batch_size, alpha, alpha)

    # Check if level 1 labels are the same and level N labels are different
    levelN_1_match = tf.equal(tf.argmax(labels1[-2], axis=1), tf.argmax(labels2[-2], axis=1))
    levelN_mismatch = tf.not_equal(tf.argmax(labels1[-1], axis=1), tf.argmax(labels2[-1], axis=1))

    # Perform mixup only for samples where levelN_1_match is true and levelN_mismatch is true
    perform_mixup = tf.logical_and(levelN_1_match, levelN_mismatch)
    perform_mixup = tf.reshape(perform_mixup, [-1, 1])  # Reshape to match the labels shape

    def mix_images(i):
        boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_values[i], IMG_SIZE)

        crop2 = tf.image.crop_to_bounding_box(images2[i], boundaryy1, boundaryx1, target_h, target_w)
        image2 = tf.image.pad_to_bounding_box(crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)
        crop1 = tf.image.crop_to_bounding_box(images1[i], boundaryy1, boundaryx1, target_h, target_w)
        img1 = tf.image.pad_to_bounding_box(crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)

        image1 = images1[i] - img1
        # mixed_image = image1 + image2
        mixed_image = tf.where(perform_mixup[i], image1 + image2, images1[i])

        lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
        lambda_value = tf.cast(lambda_value, tf.float32)

        # mixed_label = [labels1[j][i] * lambda_value + labels2[j][i] * (1 - lambda_value) for j in range(len(labels1))]
        mixed_label = [tf.where(perform_mixup[i], labels1[j][i] * lambda_value + labels2[j][i] * (1 - lambda_value), labels1[j][i])
              for j in range(len(labels1))]


        return mixed_image, mixed_label

    mixed_images, mixed_labels = tf.map_fn(
        mix_images,
        tf.range(batch_size),
        fn_output_signature=(tf.float32, [tf.float32] * len(labels1))
    )

    mixed_labels = [tf.stack(mixed_label) for mixed_label in mixed_labels]

    return mixed_images, tuple(mixed_labels)

def mixup_or_cutmix(ds_one, ds_two, alpha=0.2):
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]
    IMG_SIZE = images_one.shape[1]

    lambda_values = sample_beta_distribution(batch_size, alpha, alpha)
    random_values = tf.random.uniform((batch_size,), minval=0, maxval=1)

    def apply_mixup(i):
        l = lambda_values[i]
        x_l = tf.reshape(l, (1, 1, 1))
        y_l = tf.reshape(l, (1,))
        mixed_image = images_one[i] * x_l + images_two[i] * (1 - x_l)
        mixed_label = [labels_one[j][i] * y_l + labels_two[j][i] * (1 - y_l) for j in range(len(labels_one))]
        return mixed_image, mixed_label

    def apply_cutmix(i):
        boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_values[i], IMG_SIZE)
        crop2 = tf.image.crop_to_bounding_box(images_two[i], boundaryy1, boundaryx1, target_h, target_w)
        image2 = tf.image.pad_to_bounding_box(crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)
        crop1 = tf.image.crop_to_bounding_box(images_one[i], boundaryy1, boundaryx1, target_h, target_w)
        img1 = tf.image.pad_to_bounding_box(crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)
        image1 = images_one[i] - img1
        mixed_image = image1 + image2
        lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
        lambda_value = tf.cast(lambda_value, tf.float32)
        mixed_label = [labels_one[j][i] * lambda_value + labels_two[j][i] * (1 - lambda_value) for j in range(len(labels_one))]
        return mixed_image, mixed_label

    def mixup_or_cutmix_single(i):
        return tf.cond(random_values[i] < 0.5, lambda: apply_mixup(i), lambda: apply_cutmix(i))

    mixed_images, mixed_labels = tf.map_fn(
        mixup_or_cutmix_single,
        tf.range(batch_size),
        fn_output_signature=(tf.float32, [tf.float32] * len(labels_one))
    )

    mixed_labels = [tf.stack(mixed_label) for mixed_label in mixed_labels]

    return mixed_images, tuple(mixed_labels)

def H_Mixup_or_Cutmix(ds_one, ds_two, alpha=0.2):
    '''
    Perform Hierarchical Mixup or Cutmix on the datasets.
    - The labels are assumed to be a list of one-hot encoded vector for different hierarchical labels. i.e. [[level-1_labels], [level-2_labels], [level-3_labels], ..., [level-N_labels]]
    - The logic is to perform cutmix only for samples where level-N labels are different and level-N-1 (previous level is the hierarchy) labels are the same.
    :Input:
        ds_one: Tuple, # Tuple of images and labels from dataset one
        ds_two: Tuple, # Tuple of images and labels from dataset two
        alpha: float, # Alpha value for cutmix. Default 0.2
    :Returns:
        images: Tensor, # Images after cutmix
        labels: Tuple, # Labels after cutmix

    '''
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]
    IMG_SIZE = images_one.shape[1]

    # Sanity check: Ensure that the number of levels in the labels are at least 2
    assert len(labels_one) >= 2, "The number of levels in the labels must be at least 2"
    assert len(labels_two) >= 2, "The number of levels in the labels must be at least 2"
    
    # Find the best matches for each sample in images_one from images_two
    levelN_1_labels_one = tf.argmax(labels_one[-2], axis=1)
    levelN_1_labels_two = tf.argmax(labels_two[-2], axis=1)
    levelN_labels_one = tf.argmax(labels_one[-1], axis=1)
    levelN_labels_two = tf.argmax(labels_two[-1], axis=1)

    # Create a mask for level N-1 matches and level N mismatches
    levelN_1_match_mask = tf.equal(tf.expand_dims(levelN_1_labels_one, axis=1), tf.expand_dims(levelN_1_labels_two, axis=0))
    levelN_mismatch_mask = tf.not_equal(tf.expand_dims(levelN_labels_one, axis=1), tf.expand_dims(levelN_labels_two, axis=0))

    # Combine the masks to find valid pairs
    valid_pairs_mask = tf.logical_and(levelN_1_match_mask, levelN_mismatch_mask)

    # Find the best matches for each sample in images_one
    best_matches = tf.argmax(tf.cast(valid_pairs_mask, tf.int32), axis=1)

    # Rearrange images_two and labels_two based on the best matches
    images_two = tf.gather(images_two, best_matches)
    labels_two = [tf.gather(label, best_matches) for label in labels_two]   

    # Check if level 1 labels are the same and level N labels are different
    levelN_1_match = tf.equal(tf.argmax(labels_one[-2], axis=1), tf.argmax(labels_two[-2], axis=1))
    levelN_mismatch = tf.not_equal(tf.argmax(labels_one[-1], axis=1), tf.argmax(labels_two[-1], axis=1))

    # Perform mixup only for samples where levelN_1_match is true and levelN_mismatch is true
    perform_mixup = tf.logical_and(levelN_1_match, levelN_mismatch)
    # perform_mixup = tf.reshape(perform_mixup, [-1, 1])  # Reshape to match the labels shape    

    lambda_values = sample_beta_distribution(batch_size, alpha, alpha)
    random_values = tf.random.uniform((batch_size,), minval=0, maxval=1)

    def apply_mixup(i):
        l = lambda_values[i]
        x_l = tf.reshape(l, (1, 1, 1))
        y_l = tf.reshape(l, (1,))
        mixed_image = images_one[i] * x_l + images_two[i] * (1 - x_l)
        mixed_label = [labels_one[j][i] * y_l + labels_two[j][i] * (1 - y_l) for j in range(len(labels_one))]
        return mixed_image, mixed_label

    def apply_cutmix(i):
        boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_values[i], IMG_SIZE)
        crop2 = tf.image.crop_to_bounding_box(images_two[i], boundaryy1, boundaryx1, target_h, target_w)
        image2 = tf.image.pad_to_bounding_box(crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)
        crop1 = tf.image.crop_to_bounding_box(images_one[i], boundaryy1, boundaryx1, target_h, target_w)
        img1 = tf.image.pad_to_bounding_box(crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)
        image1 = images_one[i] - img1
        mixed_image = image1 + image2
        lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
        lambda_value = tf.cast(lambda_value, tf.float32)
        mixed_label = [labels_one[j][i] * lambda_value + labels_two[j][i] * (1 - lambda_value) for j in range(len(labels_one))]
        return mixed_image, mixed_label

    def mixup_or_cutmix_single(i):
        return tf.cond(random_values[i] < 0.5, lambda: apply_mixup(i), lambda: apply_cutmix(i))

    mixed_images, mixed_labels = tf.map_fn(
        mixup_or_cutmix_single,
        tf.range(batch_size),
        fn_output_signature=(tf.float32, [tf.float32] * len(labels_one))
    )

    mixed_labels = [tf.stack(mixed_label) for mixed_label in mixed_labels]

    mixed_images = tf.where(tf.reshape(perform_mixup, (batch_size, 1, 1, 1)), mixed_images, images_one)
    mixed_labels = [tf.where(tf.reshape(perform_mixup, (batch_size, 1)), mixed_labels[i], labels_one[i]) for i in range(len(labels_one))]

    return mixed_images, tuple(mixed_labels)

def make_pipeline(image, label, image_size:tuple = None, normalize:str=None, batch_size:int=32, augment:str=None, alpha_val:float=0.2,image_type:str='array',dataset_path:str=None):
    '''
    This function is used to make the dataset pipeline. Provide the images and labels as a tensor/array. The function will return the dataset pipeline.
    :Input:
        image: Tensor/Array, # Provide the images as a tensor/array
        label: Tensor/Array, # Provide the labels as a tensor/array. For multi-label classification, provide the labels as a list of following format: [coarse to fine labels] eg: [coarse1, coarse2, fine]
    :Returns:
        dataset_pipeline: Dataset pipeline, # Returns the dataset pipeline
    '''
    # Check if the image is provided as a Array. If not, load image and convert it to array.
    if image_type == 'array':
        # Check if the labels are provided as a list or not. If len(label) == 1, then only one label is provided. If len(label) > 1, then multiple labels are provided.
        # Make tuple of the labels accordingly.
            
        # Check if data normalization is required or not. If yes, then normalize the data.
        if normalize == 'StandardScaler': # Standardize features by removing the mean and scaling to unit variance.
            image = Normalize_StandardScaler(image)
        elif normalize == 'MinMaxScaler': # Transform features by scaling each feature to a given range.
            image = Normalize_MinMaxScaler(image)
            
        # Check if data augmentation is required or not. If yes, then augment the data.
        if augment == 'MixUp': # If MixUp is required.

            labels = tuple(label) # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup

            # Create two datasets
            ds_one = tf.data.Dataset.from_tensor_slices((image,labels)).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            ds_two = tf.data.Dataset.from_tensor_slices((image,labels)).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            # Mixup the two datasets
            ds = tf.data.Dataset.zip((ds_one, ds_two)) # Zip the two datasets
            dataset_pipeline = ds.map(lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset
            # dataset_pipeline = tf.data.Dataset.zip((ds_one, ds_two)).map(mix_up, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
        
        elif augment == 'CutMix': # If CutMix is required.
            labels = tuple(label) # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup
            # Create two datasets
            ds_one = tf.data.Dataset.from_tensor_slices((image,labels)).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            ds_two = tf.data.Dataset.from_tensor_slices((image,labels)).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)            
            # Mixup the two datasets
            ds = tf.data.Dataset.zip((ds_one, ds_two)) # Zip the two datasets
            # Map the mixup function to the zipped dataset
            dataset_pipeline = ds.map(lambda ds_one, ds_two: cutmix(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE)
            # dataset_pipeline = tf.data.Dataset.zip((ds_one, ds_two)).map(cutmix, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)

        elif augment == 'MixupAndCutMix': # If CutMix is required.
            labels = tuple(label) # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup
            # Create two datasets
            ds_one = tf.data.Dataset.from_tensor_slices((image,labels)).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            ds_two = tf.data.Dataset.from_tensor_slices((image,labels)).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)            
            # Mixup the two datasets
            ds = tf.data.Dataset.zip((ds_one, ds_two)) # Zip the two datasets
            dataset_mixup1 = ds.map(lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset
            dataset_mixup2 = ds.map(lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset
            ds_mixup = tf.data.Dataset.zip((dataset_mixup1, dataset_mixup2)) # Zip the two datasets
            dataset_pipeline = ds_mixup.map(lambda dataset_mixup1, dataset_mixup2: cutmix(dataset_mixup1, dataset_mixup2, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset

        elif augment == 'MixupORCutMix': # If CutMix is required.
            labels = tuple(label) # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup
            # Create two datasets
            ds_one = tf.data.Dataset.from_tensor_slices((image,labels)).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            ds_two = tf.data.Dataset.from_tensor_slices((image,labels)).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)            
            # Mixup the two datasets
            ds = tf.data.Dataset.zip((ds_one, ds_two)) # Zip the two datasets
            # Map the mixup function to the zipped dataset
            dataset_pipeline = ds.map(lambda ds_one, ds_two: mixup_or_cutmix(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE)
            # dataset_pipeline = tf.data.Dataset.zip((ds_one, ds_two)).map(cutmix, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)

        elif augment == 'H_MixUp': # If HierarchicalMixUp is required.
            
            labels = tuple(label) # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup

            # Create two datasets
            ds_one = tf.data.Dataset.from_tensor_slices((image,labels)).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            ds_two = tf.data.Dataset.from_tensor_slices((image,labels)).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            # Mixup the two datasets
            ds = tf.data.Dataset.zip((ds_one, ds_two)) # Zip the two datasets
            dataset_pipeline = ds.map(lambda ds_one, ds_two: H_MixUp(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset
            # dataset_pipeline = tf.data.Dataset.zip((ds_one, ds_two)).map(mix_up, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)

        elif augment == 'H_CutMix': # If HierarchicalMixUp is required.
            
            labels = tuple(label) # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup

            # Create two datasets
            ds_one = tf.data.Dataset.from_tensor_slices((image,labels)).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            ds_two = tf.data.Dataset.from_tensor_slices((image,labels)).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            # Mixup the two datasets
            ds = tf.data.Dataset.zip((ds_one, ds_two)) # Zip the two datasets
            dataset_pipeline = ds.map(lambda ds_one, ds_two: H_CutMix(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset
            # dataset_pipeline = tf.data.Dataset.zip((ds_one, ds_two)).map(mix_up, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)

        elif augment == 'H_MixupAndCutMix': # If CutMix is required.
            labels = tuple(label) # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup
            # Create two datasets
            ds_one = tf.data.Dataset.from_tensor_slices((image,labels)).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            ds_two = tf.data.Dataset.from_tensor_slices((image,labels)).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)            
            # Mixup the two datasets
            ds = tf.data.Dataset.zip((ds_one, ds_two)) # Zip the two datasets
            dataset_mixup1 = ds.map(lambda ds_one, ds_two: H_MixUp(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset
            dataset_mixup2 = ds.map(lambda ds_one, ds_two: H_MixUp(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset
            ds_mixup = tf.data.Dataset.zip((dataset_mixup1, dataset_mixup2)) # Zip the two datasets
            dataset_pipeline = ds_mixup.map(lambda dataset_mixup1, dataset_mixup2: H_CutMix(dataset_mixup1, dataset_mixup2, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset

        elif augment == 'H_MixupORCutMix': # If CutMix is required.
            labels = tuple(label) # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup
            # Create two datasets
            ds_one = tf.data.Dataset.from_tensor_slices((image,labels)).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            ds_two = tf.data.Dataset.from_tensor_slices((image,labels)).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)            
            # Mixup the two datasets
            ds = tf.data.Dataset.zip((ds_one, ds_two)) # Zip the two datasets
            # Map the mixup function to the zipped dataset
            dataset_pipeline = ds.map(lambda ds_one, ds_two: H_Mixup_or_Cutmix(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE)
            # dataset_pipeline = tf.data.Dataset.zip((ds_one, ds_two)).map(cutmix, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)

        else: # If no augmentation is required.
            if len(label) > 1: # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup
                labels = tuple(label)
            elif len(label) == 1:
                labels = label[0]
            dataset_pipeline = tf.data.Dataset.from_tensor_slices((image,labels)).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    elif image_type == 'path': # If the image is provided as a path.
        # Check if data augmentation is required or not. If yes, then augment the data.
        if augment == 'MixUp': # If MixUp is required.

            labels = tuple(label) # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup

            # `ds_one` and `ds_two` are two different datasets with the same number of elements.
            ds_one = tf.data.Dataset.from_tensor_slices(([os.path.join(dataset_path, x) for x in image], labels)) # Create dataset one
            ds_one = ds_one.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE) # Map the encode_single_sample function to the dataset one and shuffle it. It will read the images from the path and convert it to array.
            ds_one = ds_one.map(lambda x, y: (tf.image.resize(x, (image_size[0], image_size[1])), y)) # Resize the images to a fixed input size

            ds_two = tf.data.Dataset.from_tensor_slices(([os.path.join(dataset_path, x) for x in image], labels))
            ds_two = ds_two.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            ds_two = ds_two.map(lambda x, y: (tf.image.resize(x, (image_size[0], image_size[1])), y))

            # Check if data normalization is required or not. If yes, then normalize the data.
            if normalize == 'StandardScaler': # Standardize features by removing the mean and scaling to unit variance.
                ds_one = ds_one.map(lambda x, y: (Normalize_StandardScaler(x), y))
                ds_two = ds_two.map(lambda x, y: (Normalize_StandardScaler(x), y))
            elif normalize == 'MinMaxScaler': # Transform features by scaling each feature to a given range.
                ds_one = ds_one.map(lambda x, y: (Normalize_MinMaxScaler(x), y))
                ds_two = ds_two.map(lambda x, y: (Normalize_MinMaxScaler(x), y))
            # Mixup the two datasets
            ds = tf.data.Dataset.zip((ds_one, ds_two)) # Zip the two datasets
            dataset_pipeline = ds.map(lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset

        elif augment == 'CutMix': # If CutMix is required.
            labels = tuple(label) # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup

            # `ds_one` and `ds_two` are two different datasets with the same number of elements.
            ds_one = tf.data.Dataset.from_tensor_slices(([os.path.join(dataset_path, x) for x in image], labels)) # Create dataset one
            ds_one = ds_one.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE) # Map the encode_single_sample function to the dataset one and shuffle it. It will read the images from the path and convert it to array.
            ds_one = ds_one.map(lambda x, y: (tf.image.resize(x, (image_size[0], image_size[1])), y)) # Resize the images to a fixed input size

            ds_two = tf.data.Dataset.from_tensor_slices(([os.path.join(dataset_path, x) for x in image], labels))
            ds_two = ds_two.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            ds_two = ds_two.map(lambda x, y: (tf.image.resize(x, (image_size[0], image_size[1])), y))

            # Check if data normalization is required or not. If yes, then normalize the data.
            if normalize == 'StandardScaler': # Standardize features by removing the mean and scaling to unit variance.
                ds_one = ds_one.map(lambda x, y: (Normalize_StandardScaler(x), y))
                ds_two = ds_two.map(lambda x, y: (Normalize_StandardScaler(x), y))
            elif normalize == 'MinMaxScaler': # Transform features by scaling each feature to a given range.
                ds_one = ds_one.map(lambda x, y: (Normalize_MinMaxScaler(x), y))
                ds_two = ds_two.map(lambda x, y: (Normalize_MinMaxScaler(x), y))
            # Mixup the two datasets
            ds = tf.data.Dataset.zip((ds_one, ds_two)) # Zip the two datasets
            dataset_pipeline = ds.map(lambda ds_one, ds_two: cutmix(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset

        elif augment == 'MixupAndCutMix': # If CutMix is required.
            labels = tuple(label) # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup

            # `ds_one` and `ds_two` are two different datasets with the same number of elements.
            ds_one = tf.data.Dataset.from_tensor_slices(([os.path.join(dataset_path, x) for x in image], labels)) # Create dataset one
            ds_one = ds_one.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE) # Map the encode_single_sample function to the dataset one and shuffle it. It will read the images from the path and convert it to array.
            ds_one = ds_one.map(lambda x, y: (tf.image.resize(x, (image_size[0], image_size[1])), y)) # Resize the images to a fixed input size

            ds_two = tf.data.Dataset.from_tensor_slices(([os.path.join(dataset_path, x) for x in image], labels))
            ds_two = ds_two.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            ds_two = ds_two.map(lambda x, y: (tf.image.resize(x, (image_size[0], image_size[1])), y))

            # Check if data normalization is required or not. If yes, then normalize the data.
            if normalize == 'StandardScaler': # Standardize features by removing the mean and scaling to unit variance.
                ds_one = ds_one.map(lambda x, y: (Normalize_StandardScaler(x), y))
                ds_two = ds_two.map(lambda x, y: (Normalize_StandardScaler(x), y))
            elif normalize == 'MinMaxScaler': # Transform features by scaling each feature to a given range.
                ds_one = ds_one.map(lambda x, y: (Normalize_MinMaxScaler(x), y))
                ds_two = ds_two.map(lambda x, y: (Normalize_MinMaxScaler(x), y))
            # Mixup the two datasets
            ds = tf.data.Dataset.zip((ds_one, ds_two)) # Zip the two datasets
            dataset_mixup1 = ds.map(lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset
            dataset_mixup2 = ds.map(lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset
            ds_mixup = tf.data.Dataset.zip((dataset_mixup1, dataset_mixup2)) # Zip the two datasets
            dataset_pipeline = ds_mixup.map(lambda dataset_mixup1, dataset_mixup2: cutmix(dataset_mixup1, dataset_mixup2, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset

        elif augment == 'MixupORCutMix': # If CutMix is required.
            labels = tuple(label) # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup

            # `ds_one` and `ds_two` are two different datasets with the same number of elements.
            ds_one = tf.data.Dataset.from_tensor_slices(([os.path.join(dataset_path, x) for x in image], labels)) # Create dataset one
            ds_one = ds_one.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE) # Map the encode_single_sample function to the dataset one and shuffle it. It will read the images from the path and convert it to array.
            ds_one = ds_one.map(lambda x, y: (tf.image.resize(x, (image_size[0], image_size[1])), y)) # Resize the images to a fixed input size

            ds_two = tf.data.Dataset.from_tensor_slices(([os.path.join(dataset_path, x) for x in image], labels))
            ds_two = ds_two.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            ds_two = ds_two.map(lambda x, y: (tf.image.resize(x, (image_size[0], image_size[1])), y))

            # Check if data normalization is required or not. If yes, then normalize the data.
            if normalize == 'StandardScaler': # Standardize features by removing the mean and scaling to unit variance.
                ds_one = ds_one.map(lambda x, y: (Normalize_StandardScaler(x), y))
                ds_two = ds_two.map(lambda x, y: (Normalize_StandardScaler(x), y))
            elif normalize == 'MinMaxScaler': # Transform features by scaling each feature to a given range.
                ds_one = ds_one.map(lambda x, y: (Normalize_MinMaxScaler(x), y))
                ds_two = ds_two.map(lambda x, y: (Normalize_MinMaxScaler(x), y))
            # Mixup the two datasets
            ds = tf.data.Dataset.zip((ds_one, ds_two)) # Zip the two datasets
            dataset_pipeline = ds.map(lambda ds_one, ds_two: mixup_or_cutmix(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset

        elif augment == 'H_MixUp': # If HierarchicalMixUp is required.
            labels = tuple(label) # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup

            # `ds_one` and `ds_two` are two different datasets with the same number of elements.
            ds_one = tf.data.Dataset.from_tensor_slices(([os.path.join(dataset_path, x) for x in image], labels)) # Create dataset one
            ds_one = ds_one.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE) # Map the encode_single_sample function to the dataset one and shuffle it. It will read the images from the path and convert it to array.
            ds_one = ds_one.map(lambda x, y: (tf.image.resize(x, (image_size[0], image_size[1])), y)) # Resize the images to a fixed input size

            ds_two = tf.data.Dataset.from_tensor_slices(([os.path.join(dataset_path, x) for x in image], labels))
            ds_two = ds_two.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            ds_two = ds_two.map(lambda x, y: (tf.image.resize(x, (image_size[0], image_size[1])), y))

            # Check if data normalization is required or not. If yes, then normalize the data.
            if normalize == 'StandardScaler': # Standardize features by removing the mean and scaling to unit variance.
                ds_one = ds_one.map(lambda x, y: (Normalize_StandardScaler(x), y))
                ds_two = ds_two.map(lambda x, y: (Normalize_StandardScaler(x), y))
            elif normalize == 'MinMaxScaler': # Transform features by scaling each feature to a given range.
                ds_one = ds_one.map(lambda x, y: (Normalize_MinMaxScaler(x), y))
                ds_two = ds_two.map(lambda x, y: (Normalize_MinMaxScaler(x), y))
            # Mixup the two datasets
            ds = tf.data.Dataset.zip((ds_one, ds_two)) # Zip the two datasets
            dataset_pipeline = ds.map(lambda ds_one, ds_two: H_MixUp(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset

        elif augment == 'H_CutMix': # If HierarchicalMixUp is required.
            labels = tuple(label) # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup

            # `ds_one` and `ds_two` are two different datasets with the same number of elements.
            ds_one = tf.data.Dataset.from_tensor_slices(([os.path.join(dataset_path, x) for x in image], labels)) # Create dataset one
            ds_one = ds_one.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE) # Map the encode_single_sample function to the dataset one and shuffle it. It will read the images from the path and convert it to array.
            ds_one = ds_one.map(lambda x, y: (tf.image.resize(x, (image_size[0], image_size[1])), y)) # Resize the images to a fixed input size

            ds_two = tf.data.Dataset.from_tensor_slices(([os.path.join(dataset_path, x) for x in image], labels))
            ds_two = ds_two.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            ds_two = ds_two.map(lambda x, y: (tf.image.resize(x, (image_size[0], image_size[1])), y))

            # Check if data normalization is required or not. If yes, then normalize the data.
            if normalize == 'StandardScaler': # Standardize features by removing the mean and scaling to unit variance.
                ds_one = ds_one.map(lambda x, y: (Normalize_StandardScaler(x), y))
                ds_two = ds_two.map(lambda x, y: (Normalize_StandardScaler(x), y))
            elif normalize == 'MinMaxScaler': # Transform features by scaling each feature to a given range.
                ds_one = ds_one.map(lambda x, y: (Normalize_MinMaxScaler(x), y))
                ds_two = ds_two.map(lambda x, y: (Normalize_MinMaxScaler(x), y))
            # Mixup the two datasets
            ds = tf.data.Dataset.zip((ds_one, ds_two)) # Zip the two datasets
            dataset_pipeline = ds.map(lambda ds_one, ds_two: H_CutMix(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset

        elif augment == 'H_MixupAndCutMix': # If CutMix is required.
            labels = tuple(label) # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup

            # `ds_one` and `ds_two` are two different datasets with the same number of elements.
            ds_one = tf.data.Dataset.from_tensor_slices(([os.path.join(dataset_path, x) for x in image], labels)) # Create dataset one
            ds_one = ds_one.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE) # Map the encode_single_sample function to the dataset one and shuffle it. It will read the images from the path and convert it to array.
            ds_one = ds_one.map(lambda x, y: (tf.image.resize(x, (image_size[0], image_size[1])), y)) # Resize the images to a fixed input size

            ds_two = tf.data.Dataset.from_tensor_slices(([os.path.join(dataset_path, x) for x in image], labels))
            ds_two = ds_two.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            ds_two = ds_two.map(lambda x, y: (tf.image.resize(x, (image_size[0], image_size[1])), y))

            # Check if data normalization is required or not. If yes, then normalize the data.
            if normalize == 'StandardScaler': # Standardize features by removing the mean and scaling to unit variance.
                ds_one = ds_one.map(lambda x, y: (Normalize_StandardScaler(x), y))
                ds_two = ds_two.map(lambda x, y: (Normalize_StandardScaler(x), y))
            elif normalize == 'MinMaxScaler': # Transform features by scaling each feature to a given range.
                ds_one = ds_one.map(lambda x, y: (Normalize_MinMaxScaler(x), y))
                ds_two = ds_two.map(lambda x, y: (Normalize_MinMaxScaler(x), y))
            # Mixup the two datasets
            ds = tf.data.Dataset.zip((ds_one, ds_two)) # Zip the two datasets
            dataset_mixup1 = ds.map(lambda ds_one, ds_two: H_MixUp(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset
            dataset_mixup2 = ds.map(lambda ds_one, ds_two: H_MixUp(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset
            ds_mixup = tf.data.Dataset.zip((dataset_mixup1, dataset_mixup2)) # Zip the two datasets
            dataset_pipeline = ds_mixup.map(lambda dataset_mixup1, dataset_mixup2: H_CutMix(dataset_mixup1, dataset_mixup2, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset

        elif augment == 'H_MixupORCutMix': # If H_MixupORCutMix is required.
            labels = tuple(label) # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup

            # `ds_one` and `ds_two` are two different datasets with the same number of elements.
            ds_one = tf.data.Dataset.from_tensor_slices(([os.path.join(dataset_path, x) for x in image], labels)) # Create dataset one
            ds_one = ds_one.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE) # Map the encode_single_sample function to the dataset one and shuffle it. It will read the images from the path and convert it to array.
            ds_one = ds_one.map(lambda x, y: (tf.image.resize(x, (image_size[0], image_size[1])), y)) # Resize the images to a fixed input size

            ds_two = tf.data.Dataset.from_tensor_slices(([os.path.join(dataset_path, x) for x in image], labels))
            ds_two = ds_two.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size * 100).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
            ds_two = ds_two.map(lambda x, y: (tf.image.resize(x, (image_size[0], image_size[1])), y))

            # Check if data normalization is required or not. If yes, then normalize the data.
            if normalize == 'StandardScaler': # Standardize features by removing the mean and scaling to unit variance.
                ds_one = ds_one.map(lambda x, y: (Normalize_StandardScaler(x), y))
                ds_two = ds_two.map(lambda x, y: (Normalize_StandardScaler(x), y))
            elif normalize == 'MinMaxScaler': # Transform features by scaling each feature to a given range.
                ds_one = ds_one.map(lambda x, y: (Normalize_MinMaxScaler(x), y))
                ds_two = ds_two.map(lambda x, y: (Normalize_MinMaxScaler(x), y))
            # Mixup the two datasets
            ds = tf.data.Dataset.zip((ds_one, ds_two)) # Zip the two datasets
            dataset_pipeline = ds.map(lambda ds_one, ds_two: H_Mixup_or_Cutmix(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE) # Map the mixup function to the zipped dataset

        else: # If no augmentation is required.
            if len(label) > 1: # Convert the labels to tuple # DONT TAKE IT OUTSIDE THE IF LOOP AS IT WILL GIVE ERROR for single label without mixup
                labels = tuple(label)
            elif len(label) == 1:
                labels = label[0]

            dataset_pipeline = tf.data.Dataset.from_tensor_slices(([os.path.join(dataset_path, x) for x in image], labels)) # Create dataset from the image paths and labels
            dataset_pipeline = dataset_pipeline.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE).padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE) # Map the encode_single_sample function to the dataset. This function will read the image from the path and convert it to tensor.
            # Resize the images to a fixed input size, and rescale the input channels to a range of [-1,1]
            dataset_pipeline = dataset_pipeline.map(lambda x, y: (tf.image.resize(x, (image_size[0], image_size[1])), y)) # Resize the images to a fixed input size


            # Check if data normalization is required or not. If yes, then normalize the data.
            if normalize == 'StandardScaler': # Standardize features by removing the mean and scaling to unit variance.
                dataset_pipeline = dataset_pipeline.map(lambda x, y: (Normalize_StandardScaler(x), y))
            elif normalize == 'MinMaxScaler': # Transform features by scaling each feature to a given range.
                dataset_pipeline = dataset_pipeline.map(lambda x, y: (Normalize_MinMaxScaler(x), y))
        

    return dataset_pipeline 

def encode_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_image(img, expand_animations=False)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, (512, 512))
    return img, label

class make_dataset():
    """
    This class is used to make the dataset. The dataset is made using the following steps:
    1.  Provide the training and testing images and labels. The labels should be in one-hot encoding format.
        The labels should be provided as a list. The shape of the images should be (N, H, W, C) where N is the number of images, H is the height of the image, W is the width of the image and C is the number of channels.
    :Input:
        name:str, # Name of the dataset as a string
        training_images, # Training images: Provide the training images as a numpy array.
        training_labels:list,  # List of one-hot training labels: Provide the training labels as a list. For Hierarchical Multi-Label classification, provide the labels as a list of following format: [coarse to fine labels] eg: [coarse1, coarse2, fine]
        test_images, # Testing images: Provide the testing images as a numpy array
        test_labels:list, # List of one-hot testing labels: Provide the testing labels as a list
        image_size:int=(64, 64, 3),    # Size of the image: Default (64, 64, 3)
        Batch size:int=32, # Batch size: Default 32
        data_normalizing:str='normalize',  # Data normalizing: Default 'normalize'. Normalize the data using the STD and Mean values
        class_encoding:str = 'Label_Encoder', # Class encoding: Default 'Label_Encoder'. Encode the class labels
        data_augmentation:str = 'mixup', # Default 'mixup'. Augment the data using mixup
        data_aug_alpha:float = 0.2 # Mixup alpha value: Default 0.2

    :Class Variables:
        name:str, # Name of the dataset
        
    """
    def __init__(self, name:str, # Name of the dataset
                 training_images, # Training images: Provide the training images
                 training_labels:list,  # List of one-hot training labels: Provide the training labels. Provide as a list for multi-label classification
                 test_images, 
                 test_labels:list, # In one-hot encoding format
                 labels_name: list = None, # List of labels name: Provide the labels name as a list. For multi-label classification, provide the labels name as a list of following format: [['Coarse1', 'labels', 'name'], ['Coarse2', 'labels', 'name'], ['Fine', 'labels', 'name']]
                 training_validation_split:float=0.1, # Split the training data into training and validation data. Provide the size of validation data. Default 0.1. For no validation data, provide 0.0
                 validation_split_from:str = 'training', # 'training', 'testing'. The validation set is taken from the training or testing data. Default 'training'
                 image_size:int=(64, 64, 3),    # Size of the image: Default (64, 64, 3). Change the image size matching the dataset image size
                 image_type:str = 'array', # 'array'; 'path'; Type of the image: Default 'array'. Provide the type of the image. 'array' for numpy array and 'path' for image path
                 dataset_path:str = None, # Path of the dataset: Default None. Provide the path of the dataset if the image_type is 'path'
                 batch_size:int=32, # Batch size: Default 32
                 data_normalizing:str='StandardScaler',  # Data normalizing: Default 'StandardScaler'. Normalize the data using the STD and Mean values
                 class_encoding:str = 'Label_Encoder', # Class encoding: Default 'Label_Encoder'. Encode the class labels
                 data_augmentation:str = None , # Default None. Augment the data using following methods: 'mixup' (Mixup the data)
                 data_aug_alpha:float = 0.2 # Mixup alpha value: Default 0.2. Mixup alpha value should be between 0 and 1 Required only if data_augmentation = 'mixup'
                 ):
        self.name = name
        self.image_size = image_size
        self.batch_size = batch_size
        # 1. Split the training data into training and validation data (If required)
        if training_validation_split == 0.0: # If no validation data is required
            print('No validation data is required, Validation data and Testing data will be same.')
            validation_images, validation_labels = test_images, test_labels

        elif training_validation_split == 0.5: # if validation data split is 50% of training data
            print(f'Validation data split is 50% of training data.\n The shape of the training labels and validation labels will be same. It will not work for Hierarchical Multi-Label classification.')
            pass

        elif training_validation_split > 0.0 and training_validation_split < 1.0 and training_validation_split != 0.5: # If validation data split is less than 100% of training data
            if validation_split_from == 'training':
                training_images, validation_images, *y_train = train_test_split(training_images, *training_labels,
                                                                                        test_size = training_validation_split,
                                                                                        random_state = 42,
                                                                                        stratify = training_labels[-1]
                                                                                        )
                # Split the labels into training and validation labels
                training_labels, validation_labels = [[y_train[i] for i in range(len(y_train)) if len(y_train[i])==len(training_images)],
                        [y_train[i] for i in range(len(y_train)) if len(y_train[i])==len(validation_images)]]
           
            elif validation_split_from == 'testing':
                test_images, validation_images, *y_test = train_test_split(test_images, *test_labels,
                                                                            test_size = training_validation_split,
                                                                            random_state = 42,
                                                                            stratify = test_labels[-1]
                                                                            )
                # Split the labels into testing and validation labels
                test_labels, validation_labels = [[y_test[i] for i in range(len(y_test)) if len(y_test[i])==len(test_images)],
                        [y_test[i] for i in range(len(y_test)) if len(y_test[i])==len(validation_images)]]

            
        # 2. Get the Images and Labels after splitting the data into training, validation and testing data Using above steps
        self.training_images = training_images
        self.training_labels = training_labels
        self.validation_images = validation_images
        self.validation_labels = validation_labels
        self.test_images = test_images
        self.test_labels = test_labels

        # 3. Get the labels name
        self.labels_name = labels_name
        self.data_normalizing = data_normalizing
        self.class_encoding = class_encoding
        self.data_augmentation = data_augmentation
        self.data_aug_alpha = data_aug_alpha

        # get the number of classes in each level From coarse to fine. As the input labels are in one-hot encoding format, we can get the number of classes by getting the number of unique values in the labels
        self.num_classes = [len(set(np.argmax(self.training_labels[i], axis=1))) for i in range(len(self.training_labels))] # List of number of classes in each level for training data

        # Encode the taxonomy of the dataset
        # Define the label maps for Parent-Child relationship
        # Empty list to store the label maps

        # If the number of levels are 1, i.e. len(num_classes) = 1 then follow the below steps for creating the label maps
        if len(self.num_classes) == 1:
            # level_maps = [[0 for x in range(num_classes_l0)] for y in range(num_classes_l1)]
            level_maps = [[[0] for i in range(self.num_classes[0])]] # Each Row is a parent class and each column is a child class. Each column has only one value that is 0. This is done to make the mapping with only the root level. As the root level has only one class, the mapping will be 0

        # If the number of levels are more then 1, i.e. len(num_classes) > 1 then follow the below steps for creating the label maps
        if len(self.num_classes) > 1:
            level_maps = [[[0 for x in range(self.num_classes[i+1])] for y in range(self.num_classes[i])] for i in range(len(self.num_classes)-1)] # Each Row is a parent class and each column is a child class
            for i in range(len(self.num_classes)-1): # Loop through each level. Ignore the last level as it does not have any child
                # Loop through each image labels and get the parent and child labels
                # Convert the one-hot encoding labels to integer labels using argmax function and make a list of labels,
                for (p,c) in zip(np.argmax(self.training_labels[i], axis=1).tolist(), np.argmax(self.training_labels[i+1], axis=1).tolist()):
                    level_maps[i][p][c] = 1 # Set the value to 1 for the parent (P) and child (C) from the Training Labels
        
        # The taxonomy will be the list of label maps
        self.taxonomy = level_maps  # List of label maps for each Parent-Child relationship. For N levels, there will be N-1 label maps, since the last level does not have any child
        
        # Get the hierarchical label tree Based on the taxonomy and the labels name
        self.label_tree = make_hierarchical_tree(self.taxonomy, self.labels_name)

        # Now create the dataset Pipeline.

        # 2. Make the pipeline for training, validation and test data. The pipeline function, which will return the pipelines for the datasets.
            # 2.1 Make the pipeline for training data
        self.training = make_pipeline(image = self.training_images, 
                                              label = self.training_labels,
                                              image_size= self.image_size,
                                              normalize = self.data_normalizing,
                                              batch_size = self.batch_size,
                                              augment = self.data_augmentation,
                                              alpha_val = self.data_aug_alpha,
                                              image_type = image_type,
                                              dataset_path = dataset_path)
        

            # 2.2 Make the pipeline for validation data
            # If no validation data is required then validation dataset will be same as test dataset
        self.validation = make_pipeline(image = self.validation_images,
                                                label = self.validation_labels,
                                                image_size= self.image_size,
                                                normalize = self.data_normalizing,
                                                batch_size = self.batch_size,
                                                augment = None, # No augmentation for validation data
                                                alpha_val = None,
                                                image_type = image_type,
                                                dataset_path = dataset_path)

            # 2.3 Make the pipeline for test data
        self.test = make_pipeline(image = self.test_images,
                                          label = self.test_labels,
                                          image_size= self.image_size,
                                          normalize = self.data_normalizing,
                                          batch_size = self.batch_size,
                                          augment = None, # No augmentation for test data
                                          alpha_val= None,
                                          image_type = image_type,
                                          dataset_path = dataset_path)


def MNIST(dataset_name:str = 'MNIST', 
                    training_validation_split:float = 0.0, # 0.0 for no validation set
                    image_size:tuple = (28, 28, 1), # (height, width, channels)
                    batch_size:int = 32, # batch size for pipeline
                    number_of_hierarchy_levels:list = [0,1], # Number of hierarchy levels in the dataset
                    data_normalizing:str = 'MinMaxScaler', # 'StandardScaler', 'MinMaxScaler', 'None'
                    class_encoding:str = 'one_hot', # 'one_hot', 'label_encoding', 'None'
                    data_augmentation = 'MixUp', # 'MixUp', 'None'
                    data_aug_alpha = 0.2 # Alpha value for MixUp. 0.0 for no MixUp
             ):
    MNIST = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = MNIST.load_data()
    # For Hierarchical Classification, we need to split the labels into 3 parts
      # Number of classes in each hierarchy
    num_coarse = 5 # number of coarse classes in the first hierarchy
    num_fine  = 10 # number of fine classes in the third hierarchy
    #-------------------- data loading ----------------------
    # convert to float32
    # Reshape the images to match the input shape in the model
    x_train = x_train.reshape(x_train.shape[0], image_size[0], image_size[1], image_size[2]).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], image_size[0], image_size[1], image_size[2]).astype('float32')
        # Convert labels to one-hot vectors
    y_train_fine = keras.utils.to_categorical(y_train, num_fine)
    y_test_fine = keras.utils.to_categorical(y_test, num_fine)

    # Declare the label maps for ancestor labels
    fine_coarse = {0:0, 1:2, 2:1, 3:4, 4:3, 5:4, 6:0, 7:2, 8:1, 9:3}

    # Based on the label maps, convert the fine labels to coarse labels
        # Coarse or Super labels
    y_train_coarse = np.zeros((y_train_fine.shape[0], num_coarse)).astype("float32")
    y_test_coarse = np.zeros((y_test_fine.shape[0], num_coarse)).astype("float32")
    for i in range(y_train_coarse.shape[0]):
        y_train_coarse[i][fine_coarse[np.argmax(y_train_fine[i])]] = 1.0
    for i in range(y_test_coarse.shape[0]):
        y_test_coarse[i][fine_coarse[np.argmax(y_test_fine[i])]] = 1.0
        
    coarse_labels_name = ['0', '1', '2', '3', '4']
    fine_labels_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Make a list of labels and Label names
    training_labels = [y_train_coarse, y_train_fine]
    test_labels = [y_test_coarse, y_test_fine]
    labels_name = [coarse_labels_name, fine_labels_name]

    # Sanity check: number_of_hierarchy_levels should be a list of integers between 0 and 1. If not, raise an error
    if not all(isinstance(item, int) for item in number_of_hierarchy_levels):
        raise ValueError('number_of_hierarchy_levels should be a list of integers between 0 and 1')
    if not all(item in [0,1] for item in number_of_hierarchy_levels):
        raise ValueError('number_of_hierarchy_levels should be a list of integers between 0 and 1')
    
    # Match the required labels with number_of_hierarchy_levels. For example, if number_of_hierarchy_levels = [0,1] then the labels will be [Level_0_labels_name, Level_1_labels_name]; if number_of_hierarchy_levels = [2] then the labels will be [Level_2_labels_name]
    training_labels = [training_labels[i] for i in number_of_hierarchy_levels]
    test_labels = [test_labels[i] for i in number_of_hierarchy_levels]
    labels_name = [labels_name[i] for i in number_of_hierarchy_levels]
    
    dataset = make_dataset(name = dataset_name,
                            training_images = x_train,
                            training_labels = training_labels,
                            test_images = x_test,
                            test_labels = test_labels,
                            labels_name = labels_name, 
                            training_validation_split = training_validation_split,
                            image_size = image_size,
                            batch_size = batch_size,
                            data_normalizing = data_normalizing,
                            class_encoding = class_encoding,
                            data_augmentation = data_augmentation,
                            data_aug_alpha = data_aug_alpha)
    return dataset

def EMNIST(dataset_name:str = 'EMNIST', 
                    training_validation_split:float = 0.0, # 0.0 for no validation set
                    image_size:tuple = (28, 28, 1), # (height, width, channels)
                    batch_size:int = 32, # batch size for pipeline
                    number_of_hierarchy_levels:list = [0,1], # Number of hierarchy levels in the dataset
                    data_normalizing:str = 'MinMaxScaler', # 'StandardScaler', 'MinMaxScaler', 'None'
                    class_encoding:str = 'one_hot', # 'one_hot', 'label_encoding', 'None'
                    data_augmentation = 'MixUp', # 'MixUp', 'None'
                    data_aug_alpha = 0.2 # Alpha value for MixUp. 0.0 for no MixUp
             ):
    from emnist import extract_training_samples, extract_test_samples
    x_train, y_train = extract_training_samples('balanced')
    x_test, y_test = extract_test_samples('balanced')
    # For Hierarchical Classification, we need to split the labels into 3 parts
      # Number of classes in each hierarchy
    num_coarse = 2 # number of coarse classes in the first hierarchy
    num_fine  = 47 # number of fine classes in the third hierarchy
    #-------------------- data loading ----------------------
    # convert to float32
    # Reshape the images to match the input shape in the model
    x_train = x_train.reshape(x_train.shape[0], image_size[0], image_size[1], image_size[2]).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], image_size[0], image_size[1], image_size[2]).astype('float32')
        # Convert labels to one-hot vectors
    y_train_fine = keras.utils.to_categorical(y_train, num_fine)
    y_test_fine = keras.utils.to_categorical(y_test, num_fine)

    # Declare the label maps for ancestor labels
    fine_coarse = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:1,11:1,12:1,13:1,14:1,15:1,16:1,17:1,18:1,19:1,20:1,
                   21:1,22:1,23:1,24:1,25:1,26:1,27:1,28:1,29:1,30:1,31:1,32:1,33:1,34:1,35:1,36:1,37:1,38:1,39:1,
                   40:1,41:1,42:1,43:1,44:1,45:1,46:1}

    # Based on the label maps, convert the fine labels to coarse labels
        # Coarse or Super labels
    y_train_coarse = np.zeros((y_train_fine.shape[0], num_coarse)).astype("float32")
    y_test_coarse = np.zeros((y_test_fine.shape[0], num_coarse)).astype("float32")
    for i in range(y_train_coarse.shape[0]):
        y_train_coarse[i][fine_coarse[np.argmax(y_train_fine[i])]] = 1.0
    for i in range(y_test_coarse.shape[0]):
        y_test_coarse[i][fine_coarse[np.argmax(y_test_fine[i])]] = 1.0
        
    coarse_labels_name = ['Number', 'Letter']
    fine_labels_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

    # Make a list of labels and Label names
    training_labels = [y_train_coarse, y_train_fine]
    test_labels = [y_test_coarse, y_test_fine]
    labels_name = [coarse_labels_name, fine_labels_name]

    # Sanity check: number_of_hierarchy_levels should be a list of integers between 0 and 1. If not, raise an error
    if not all(isinstance(item, int) for item in number_of_hierarchy_levels):
        raise ValueError('number_of_hierarchy_levels should be a list of integers between 0 and 1')
    if not all(item in [0,1] for item in number_of_hierarchy_levels):
        raise ValueError('number_of_hierarchy_levels should be a list of integers between 0 and 1')
    
    # Match the required labels with number_of_hierarchy_levels. For example, if number_of_hierarchy_levels = [0,1] then the labels will be [Level_0_labels_name, Level_1_labels_name]; if number_of_hierarchy_levels = [2] then the labels will be [Level_2_labels_name]
    training_labels = [training_labels[i] for i in number_of_hierarchy_levels]
    test_labels = [test_labels[i] for i in number_of_hierarchy_levels]
    labels_name = [labels_name[i] for i in number_of_hierarchy_levels]


    dataset = make_dataset(name = dataset_name,
                            training_images = x_train,
                            training_labels = training_labels,
                            test_images = x_test,
                            test_labels = test_labels,
                            labels_name = labels_name, 
                            training_validation_split = training_validation_split,
                            image_size = image_size,
                            batch_size = batch_size,
                            data_normalizing = data_normalizing,
                            class_encoding = class_encoding,
                            data_augmentation = data_augmentation,
                            data_aug_alpha = data_aug_alpha)
    return dataset

def Fashion_MNIST(dataset_name:str = 'Fashion_MNIST', 
                    training_validation_split:float = 0.0, # 0.0 for no validation set
                    image_size:tuple = (28, 28, 1), # (height, width, channels)
                    batch_size:int = 32, # batch size for pipeline
                    number_of_hierarchy_levels:list = [0,1,2], # Number of hierarchy levels in the dataset
                    data_normalizing:str = 'MinMaxScaler', # 'StandardScaler', 'MinMaxScaler', 'None'
                    class_encoding:str = 'one_hot', # 'one_hot', 'label_encoding', 'None'
                    data_augmentation = 'MixUp', # 'MixUp', 'None'
                    data_aug_alpha = 0.2 # Alpha value for MixUp. 0.0 for no MixUp
             ):
    F_MNIST = keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = F_MNIST.load_data()
    # For Hierarchical Classification, we need to split the labels into 3 parts
      # Number of classes in each hierarchy
    num_coarse_1 = 2 # number of coarse classes in the first hierarchy
    num_coarse_2 = 6 # number of coarse classes in the second hierarchy
    num_fine  = 10 # number of fine classes in the third hierarchy
    #-------------------- data loading ----------------------
    # convert to float32
    # Reshape the images to match the input shape in the model
    x_train = x_train.reshape(x_train.shape[0], image_size[0], image_size[1], image_size[2]).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], image_size[0], image_size[1], image_size[2]).astype('float32')
        # Convert labels to one-hot vectors
    y_train_fine = keras.utils.to_categorical(y_train, num_fine)
    y_test_fine = keras.utils.to_categorical(y_test, num_fine)

    # Declare the label maps for ancestor labels
    fine_coarse2 = {0:0, 1:1, 2:0, 3:2, 4:3, 5:5, 6:0, 7:5, 8:4, 9:5}
    coarse2_coarse1 = {0:0, 1:0, 2:0, 3:0, 4:1, 5:1}

    # Based on the label maps, convert the fine labels to coarse labels
        # Coarse 2 or Medium labels
    y_train_coarse2 = np.zeros((y_train_fine.shape[0], num_coarse_2)).astype("float32") # Empty array for coarse labels
    y_test_coarse2 = np.zeros((y_test_fine.shape[0], num_coarse_2)).astype("float32") # Empty array for coarse labels
    for i in range(y_train_coarse2.shape[0]):
        y_train_coarse2[i][fine_coarse2[np.argmax(y_train_fine[i])]] = 1.0
    for i in range(y_test_coarse2.shape[0]):
        y_test_coarse2[i][fine_coarse2[np.argmax(y_test_fine[i])]] = 1.0
        
        # Coarse 1 or Super labels
    y_train_coarse1 = np.zeros((y_train_coarse2.shape[0], num_coarse_1)).astype("float32")
    y_test_coarse1 = np.zeros((y_test_coarse2.shape[0], num_coarse_1)).astype("float32")
    for i in range(y_train_coarse1.shape[0]):
        y_train_coarse1[i][coarse2_coarse1[np.argmax(y_train_coarse2[i])]] = 1.0
    for i in range(y_test_coarse1.shape[0]):
        y_test_coarse1[i][coarse2_coarse1[np.argmax(y_test_coarse2[i])]] = 1.0

    # Names of the classes in each hierarchy
    coarse1_labels_name = ['Clothing', 'Goods']
    coarse2_labels_name = ['Tops', 'Bottoms', 'Dresses', 'Outerwear', 'Accessories', 'Footwear']
    fine_labels_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt','Sneaker', 'Bag', 'Ankle boot']

    # Make a list of labels and Label names
    training_labels = [y_train_coarse1, y_train_coarse2, y_train_fine]
    test_labels = [y_test_coarse1, y_test_coarse2, y_test_fine]
    labels_name = [coarse1_labels_name, coarse2_labels_name, fine_labels_name]

    # Sanity check: number_of_hierarchy_levels should be a list of integers between 0 and 2. If not, raise an error
    if not all(isinstance(item, int) for item in number_of_hierarchy_levels):
        raise ValueError('number_of_hierarchy_levels should be a list of integers between 0 and 2')
    if not all(item in [0,1,2] for item in number_of_hierarchy_levels):
        raise ValueError('number_of_hierarchy_levels should be a list of integers between 0 and 2')
    
    # Match the required labels with number_of_hierarchy_levels. For example, if number_of_hierarchy_levels = [0,1,2] then the labels will be [Level_0_labels_name, Level_1_labels_name, Level_2_labels_name]; if number_of_hierarchy_levels = [0,2] then the labels will be [Level_0_labels_name, Level_2_labels_name]
    training_labels = [training_labels[i] for i in number_of_hierarchy_levels]
    test_labels = [test_labels[i] for i in number_of_hierarchy_levels]
    labels_name = [labels_name[i] for i in number_of_hierarchy_levels]


    dataset = make_dataset(name = dataset_name,
                            training_images = x_train,
                            training_labels = training_labels,
                            test_images = x_test,
                            test_labels = test_labels,
                            labels_name = labels_name,
                            training_validation_split = training_validation_split,
                            image_size = image_size,
                            batch_size = batch_size,
                            data_normalizing = data_normalizing,
                            class_encoding = class_encoding,
                            data_augmentation = data_augmentation,
                            data_aug_alpha = data_aug_alpha)
    return dataset

def CIFAR_10(dataset_name:str = 'CIFAR_10', 
             training_validation_split:float = 0.0,
             image_size:tuple = (32, 32, 3),
             batch_size:int = 32,
             number_of_hierarchy_levels:list = [0,1,2], # Number of hierarchy levels in the dataset
             data_normalizing:str = 'StandardScaler', # 'StandardScaler', 'MinMaxScaler', 'None'
             class_encoding:str = 'one_hot', # 'one_hot', 'label_encoding', 'None'
             data_augmentation = 'MixUp', # 'MixUp', 'None'
             data_aug_alpha = 0.2 # Alpha value for MixUp. 0.0 for no MixUp
             ):
    CIFAR10 = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = CIFAR10.load_data()
    # For Hierarchical Classification, we need to split the labels into 3 parts
      # Number of classes in each hierarchy
    num_coarse_1 = 2 # number of coarse classes in the first hierarchy
    num_coarse_2 = 7 # number of coarse classes in the second hierarchy
    num_fine  = 10 # number of fine classes in the third hierarchy
    #-------------------- data loading ----------------------
    x_train = x_train.astype('float32') # convert to float32
    x_test = x_test.astype('float32') # convert to float32
        # Convert labels to one-hot vectors
    y_train_fine = keras.utils.to_categorical(y_train, num_fine)
    y_test_fine = keras.utils.to_categorical(y_test, num_fine)

    # Declare the label maps for ancestor labels
    fine_coarse2 = {0: 0, 1: 2, 2: 3, 3: 5, 4: 6, 5: 5, 6: 4, 7: 6, 8: 1, 9: 2}
    coarse2_coarse1 = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}

    # Based on the label maps, convert the fine labels to coarse labels
        # Coarse 2 or Medium labels
    y_train_coarse2 = np.zeros((y_train_fine.shape[0], num_coarse_2)).astype("float32") # Empty array for coarse labels
    y_test_coarse2 = np.zeros((y_test_fine.shape[0], num_coarse_2)).astype("float32") # Empty array for coarse labels
    for i in range(y_train_coarse2.shape[0]):
        y_train_coarse2[i][fine_coarse2[np.argmax(y_train_fine[i])]] = 1.0
    for i in range(y_test_coarse2.shape[0]):
        y_test_coarse2[i][fine_coarse2[np.argmax(y_test_fine[i])]] = 1.0
        
        # Coarse 1 or Super labels
    y_train_coarse1 = np.zeros((y_train_coarse2.shape[0], num_coarse_1)).astype("float32")
    y_test_coarse1 = np.zeros((y_test_coarse2.shape[0], num_coarse_1)).astype("float32")
    for i in range(y_train_coarse1.shape[0]):
        y_train_coarse1[i][coarse2_coarse1[np.argmax(y_train_coarse2[i])]] = 1.0
    for i in range(y_test_coarse1.shape[0]):
        y_test_coarse1[i][coarse2_coarse1[np.argmax(y_test_coarse2[i])]] = 1.0

    # Names of the classes in each hierarchy
    coarse1_labels_name = ['transport', 'animal']
    coarse2_labels_name = ['sky', 'water', 'road', 'bird', 'reptile', 'pet', 'medium']
    fine_labels_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Make a list of labels and Label names
    training_labels = [y_train_coarse1, y_train_coarse2, y_train_fine]
    test_labels = [y_test_coarse1, y_test_coarse2, y_test_fine]
    labels_name = [coarse1_labels_name, coarse2_labels_name, fine_labels_name]

    # Sanity check: number_of_hierarchy_levels should be a list of integers between 0 and 2. If not, raise an error
    if not all(isinstance(item, int) for item in number_of_hierarchy_levels):
        raise ValueError('number_of_hierarchy_levels should be a list of integers between 0 and 2')
    if not all(item in [0,1,2] for item in number_of_hierarchy_levels):
        raise ValueError('number_of_hierarchy_levels should be a list of integers between 0 and 2')
    
    # Match the required labels with number_of_hierarchy_levels. For example, if number_of_hierarchy_levels = [0,1,2] then the labels will be [Level_0_labels_name, Level_1_labels_name, Level_2_labels_name]; if number_of_hierarchy_levels = [0,2] then the labels will be [Level_0_labels_name, Level_2_labels_name]
    training_labels = [training_labels[i] for i in number_of_hierarchy_levels]
    test_labels = [test_labels[i] for i in number_of_hierarchy_levels]
    labels_name = [labels_name[i] for i in number_of_hierarchy_levels]


    dataset = make_dataset(name = dataset_name,
                            training_images = x_train,
                            training_labels = training_labels,
                            test_images = x_test,
                            test_labels = test_labels,
                            labels_name = labels_name,
                            training_validation_split = training_validation_split,
                            image_size = image_size,
                            batch_size = batch_size,
                            data_normalizing = data_normalizing,
                            class_encoding = class_encoding,
                            data_augmentation = data_augmentation,
                            data_aug_alpha = data_aug_alpha)
    return dataset

def CIFAR_100(dataset_name:str = 'CIFAR_100', 
             training_validation_split:float = 0.0,
             image_size:tuple = (32, 32, 3),
             batch_size:int = 32,
             number_of_hierarchy_levels:list = [0,1,2], # Number of hierarchy levels in the dataset
             data_normalizing:str = 'StandardScaler', # 'StandardScaler', 'MinMaxScaler', 'None'
             class_encoding:str = 'one_hot', # 'one_hot', 'label_encoding', 'None'
             data_augmentation = 'MixUp', # 'MixUp', 'None'
             data_aug_alpha = 0.2 # Alpha value for MixUp. 0.0 for no MixUp
             ):
    CIFAR100 = keras.datasets.cifar100
    (x_train, y_train), (x_test, y_test) = CIFAR100.load_data(label_mode='fine')
    # For Hierarchical Classification, we need to split the labels into 3 parts
      # Number of classes in each hierarchy
    num_coarse_1 = 8 # number of coarse classes in the first hierarchy
    num_coarse_2 = 20 # number of coarse classes in the second hierarchy
    num_fine  = 100 # number of fine classes in the third hierarchy
    #-------------------- data loading ----------------------
    x_train = x_train.astype('float32') # convert to float32
    x_test = x_test.astype('float32') # convert to float32
        # Convert labels to one-hot vectors
    y_train_fine = keras.utils.to_categorical(y_train, num_fine)
    y_test_fine = keras.utils.to_categorical(y_test, num_fine)

    # Declare the label maps for ancestor labels
    fine_coarse2 = {
    0:4,1:1,2:14,3:8,4:0,5:6,6:7,7:7,8:18,9:3,
    10:3,11:14,12:9,13:18,14:7,15:11,16:3,17:9,18:7,19:11,
    20:6,21:11,22:5,23:10,24:7,25:6,26:13,27:15,28:3,29:15,
    30:0,31:11,32:1,33:10,34:12,35:14,36:16,37:9,38:11,39:5,
    40:5,41:19,42:8,43:8,44:15,45:13,46:14,47:17,48:18,49:10,
    50:16,51:4,52:17,53:4,54:2,55:0,56:17,57:4,58:18,59:17,
    60:10,61:3,62:2,63:12,64:12,65:16,66:12,67:1,68:9,69:19,
    70:2,71:10,72:0,73:1,74:16,75:12,76:9,77:13,78:15,79:13,
    80:16,81:19,82:2,83:4,84:6,85:19,86:5,87:5,88:8,89:19,
    90:18,91:1,92:2,93:15,94:6,95:0,96:17,97:8,98:14,99:13
    }
    coarse2_coarse1 = {
      0:0, 1:0, 2:1, 3:2, 
      4:1, 5:2, 6:2, 7:3, 
      8:4, 9:5, 10:5, 11:4, 
      12:4, 13:3, 14:6, 15:4, 
      16:4, 17:1, 18:7, 19:7
    }

    # Based on the label maps, convert the fine labels to coarse labels
        # Coarse 2 or Medium labels
    y_train_coarse2 = np.zeros((y_train_fine.shape[0], num_coarse_2)).astype("float32") # Empty array for coarse labels
    y_test_coarse2 = np.zeros((y_test_fine.shape[0], num_coarse_2)).astype("float32") # Empty array for coarse labels
    for i in range(y_train_coarse2.shape[0]):
        y_train_coarse2[i][fine_coarse2[np.argmax(y_train_fine[i])]] = 1.0
    for i in range(y_test_coarse2.shape[0]):
        y_test_coarse2[i][fine_coarse2[np.argmax(y_test_fine[i])]] = 1.0
        
        # Coarse 1 or Super labels
    y_train_coarse1 = np.zeros((y_train_coarse2.shape[0], num_coarse_1)).astype("float32")
    y_test_coarse1 = np.zeros((y_test_coarse2.shape[0], num_coarse_1)).astype("float32")
    for i in range(y_train_coarse1.shape[0]):
        y_train_coarse1[i][coarse2_coarse1[np.argmax(y_train_coarse2[i])]] = 1.0
    for i in range(y_test_coarse1.shape[0]):
        y_test_coarse1[i][coarse2_coarse1[np.argmax(y_test_coarse2[i])]] = 1.0

    # Names of the classes in each hierarchy
    coarse1_labels_name = ['Aquatic Animal', 'Plants', 'Artificial', 'Invertebrates', 'Terrestrial and Amphibians Animal', 'Outdoor', 'Human', 'Transportation']

    coarse2_labels_name = ['aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
    'medium_mammals', 'non-insect_inverterates', 'people', 'reptiles', 'small_mammals', 'trees',
    'vehicles_1', 'vehicles_2']

    fine_labels_name = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
    'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
    'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur', 'dolphin',
    'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp',
    'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',
    'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
    'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television',
    'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf',
    'woman', 'worm']

    training_labels = [y_train_coarse1, y_train_coarse2, y_train_fine]
    test_labels = [y_test_coarse1, y_test_coarse2, y_test_fine]
    labels_name = [coarse1_labels_name, coarse2_labels_name, fine_labels_name]

    # Sanity check: number_of_hierarchy_levels should be a list of integers between 0 and 2. If not, raise an error
    if not all(isinstance(item, int) for item in number_of_hierarchy_levels):
        raise ValueError('number_of_hierarchy_levels should be a list of integers between 0 and 2')
    if not all(item in [0,1,2] for item in number_of_hierarchy_levels):
        raise ValueError('number_of_hierarchy_levels should be a list of integers between 0 and 2')
    
    # Match the required labels with number_of_hierarchy_levels. For example, if number_of_hierarchy_levels = [0,1,2] then the labels will be [Level_0_labels_name, Level_1_labels_name, Level_2_labels_name]; if number_of_hierarchy_levels = [0,2] then the labels will be [Level_0_labels_name, Level_2_labels_name]
    training_labels = [training_labels[i] for i in number_of_hierarchy_levels]
    test_labels = [test_labels[i] for i in number_of_hierarchy_levels]
    labels_name = [labels_name[i] for i in number_of_hierarchy_levels]


    dataset = make_dataset(name = dataset_name,
                            training_images = x_train,
                            training_labels = training_labels,
                            test_images = x_test,
                            test_labels = test_labels,
                            labels_name = labels_name,
                            training_validation_split = training_validation_split,
                            image_size = image_size,
                            batch_size = batch_size,
                            data_normalizing = data_normalizing,
                            class_encoding = class_encoding,
                            data_augmentation = data_augmentation,
                            data_aug_alpha = data_aug_alpha)
    return dataset

def Stanford_Cars(dataset_name:str = 'Stanford_Cars', 
                    training_validation_split:float = 0.499, # Percentage of test data to be used for validation
                    validation_split_from:str = 'testing', # 'training', 'testing'
                    image_size:tuple = (64, 64, 3),
                    batch_size:int = 32,
                    number_of_hierarchy_levels:list = [0,1,2], # Number of hierarchy levels in the dataset
                    data_normalizing:str = 'StandardScaler', # 'StandardScaler', 'MinMaxScaler', 'None'
                    class_encoding:str = 'one_hot', # 'one_hot', 'label_encoding', 'None'
                    data_augmentation = 'MixUp', # 'MixUp', 'None'
                    data_aug_alpha = 0.2 # Alpha value for MixUp. 0.0 for no MixUp
                    ):
    '''
    Hierarchical dataset for Stanford Cars. The dataset is divided into 3 hierarchical levels.
    This function splits the test data into validation and test sets.
    '''
    # Get training set.
    base_dir = os.path.expanduser('~/.keras/datasets')
    dataset_path = os.path.join(base_dir, 'car_ims')
    if not os.path.exists(dataset_path):
        train_data_url = 'http://ai.stanford.edu/~jkrause/car196/car_ims.tgz' # URL for training set
        dataset_path = tf.keras.utils.get_file('car_ims', train_data_url, untar=True) # Download and extract training set
        #>>>>>>>>>>> CHANGE THE PATH TO THE DATASET HERE <<<<<<<<<<<<<<<

    # GET LABELS
        # Get labels for training set
    train_labels_url = 'https://rbouadjenek.github.io/datasets/stanford_cars_train_labels.csv' # URL for training set labels
    train_labels_path = tf.keras.utils.get_file("stanford_cars_train_labels.csv", train_labels_url) # Download training set labels
        # Get labels for test set
    test_labels_url = 'https://rbouadjenek.github.io/datasets/stanford_cars_test_labels.csv' # URL for test set labels
    test_labels_path = tf.keras.utils.get_file("stanford_cars_test_labels.csv", test_labels_url) # Download test set labels
        # Read labels for training set using pandas
    train_labels_df = pd.read_csv(train_labels_path, sep=",", header=0) # Read training set labels
    train_labels_df = train_labels_df.sample(frac=1).reset_index(drop=True) # Shuffle training set labels. .sample(frac=1) shuffles the dataframe. .reset_index(drop=True) resets the index of the shuffled dataframe
        # Read labels for test set using pandas
    test_labels_df = pd.read_csv(test_labels_path, sep=",", header=0) # Read test set labels
    test_labels_df = test_labels_df.sample(frac=1, random_state=1).reset_index(drop=True) # Shuffle test set labels. .sample(frac=1) shuffles the dataframe. .reset_index(drop=True) resets the index of the shuffled dataframe

    # Get the training Labels
    level_0_train_labels = keras.utils.to_categorical(list(train_labels_df['class_level_0'])) # Convert training labels to one-hot encoding
    level_1_train_labels = keras.utils.to_categorical(list(train_labels_df['class_level_1'])) # Convert training labels to one-hot encoding
    level_2_train_labels = keras.utils.to_categorical(list(train_labels_df['class_level_2'])) # Convert training labels to one-hot encoding

    training_labels = [level_0_train_labels, level_1_train_labels, level_2_train_labels] # Combine training labels into a list

    # Get the test Labels
    level_0_test_labels = keras.utils.to_categorical(list(test_labels_df['class_level_0'])) # Convert test labels to one-hot encoding
    level_1_test_labels = keras.utils.to_categorical(list(test_labels_df['class_level_1'])) # Convert test labels to one-hot encoding
    level_2_test_labels = keras.utils.to_categorical(list(test_labels_df['class_level_2'])) # Convert test labels to one-hot encoding

    test_labels = [level_0_test_labels, level_1_test_labels, level_2_test_labels] # Combine test labels into a list

    # Get the training images
    train_images = train_labels_df['fname']
    # Get the test images
    test_images = test_labels_df['fname']

    # Name of the labels
    Level_0_labels_name = list(train_labels_df.groupby('class_level_0')['label_level_0'].unique().apply(', '.join)) # Get the names of the labels for level 0
    Level_1_labels_name = list(train_labels_df.groupby('class_level_1')['label_level_1'].unique().apply(', '.join)) # Get the names of the labels for level 1
    Level_2_labels_name = list(train_labels_df.groupby('class_level_2')['label_level_2'].unique().apply(', '.join)) # Get the names of the labels for level 2
    labels_name = [Level_0_labels_name, Level_1_labels_name, Level_2_labels_name] # Combine the names of the labels into a list

    # Sanity check: number_of_hierarchy_levels should be a list of integers between 0 and 2. If not, raise an error
    if not all(isinstance(item, int) for item in number_of_hierarchy_levels):
        raise ValueError('number_of_hierarchy_levels should be a list of integers between 0 and 2')
    if not all(item in [0,1,2] for item in number_of_hierarchy_levels):
        raise ValueError('number_of_hierarchy_levels should be a list of integers between 0 and 2')
    
    # Match the required labels with number_of_hierarchy_levels. For example, if number_of_hierarchy_levels = [0,1,2] then the labels will be [Level_0_labels_name, Level_1_labels_name, Level_2_labels_name]; if number_of_hierarchy_levels = [0,2] then the labels will be [Level_0_labels_name, Level_2_labels_name]
    training_labels = [training_labels[i] for i in number_of_hierarchy_levels]
    test_labels = [test_labels[i] for i in number_of_hierarchy_levels]
    labels_name = [labels_name[i] for i in number_of_hierarchy_levels]

    dataset = make_dataset(name = dataset_name,
                            training_images = train_images,
                            training_labels = training_labels,
                            test_images = test_images,
                            test_labels = test_labels,
                            labels_name = labels_name,
                            training_validation_split = training_validation_split,
                            validation_split_from = validation_split_from,
                            image_size = image_size,
                            image_type = 'path',
                            dataset_path = dataset_path,
                            batch_size = batch_size,
                            data_normalizing = data_normalizing,
                            class_encoding = class_encoding,
                            data_augmentation = data_augmentation,
                            data_aug_alpha = data_aug_alpha)
    return dataset

def CU_Birds_200_2011(dataset_name:str = 'CU_Birds_200_2011', 
                    training_validation_split:float = 0.499, # Percentage of test data to be used for validation
                    validation_split_from:str = 'testing', # 'training', 'testing'
                    image_size:tuple = (64, 64, 3),
                    batch_size:int = 32,
                    number_of_hierarchy_levels:list = [0,1,2], # Number of hierarchy levels in the dataset
                    data_normalizing:str = 'StandardScaler', # 'StandardScaler', 'MinMaxScaler', 'None'
                    class_encoding:str = 'one_hot', # 'one_hot', 'label_encoding', 'None'
                    data_augmentation = 'MixUp', # 'MixUp', 'None'
                    data_aug_alpha = 0.2 # Alpha value for MixUp. 0.0 for no MixUp
                    ):
    '''
    Hierarchical dataset for CU_Birds_200_2011. The dataset is divided into 3 hierarchical levels.
    This function splits the test data into validation and test sets.
    '''
    # Get training set. 
    base_dir = os.path.expanduser('~/.keras/datasets')
    dataset_path = os.path.join(base_dir, 'CUB_200_2011_v0.2')
    if not os.path.exists(dataset_path):
        train_data_url = 'http://206.12.93.90:8080/CUB_200_2011/CUB_200_2011_v0.2.tar.gz' # URL for training set
        dataset_path = tf.keras.utils.get_file('CUB_200_2011_v0.2', train_data_url, untar=True) # Download and extract training set
    #>>>>>>>>>>> CHANGE THE PATH TO THE DATASET HERE <<<<<<<<<<<<<<<

    # GET LABELS
        # Get labels for training set
    train_labels_url = 'https://rbouadjenek.github.io/datasets/cu_birds_train_labels.csv' # URL for training set labels
    train_labels_path = tf.keras.utils.get_file("cu_birds_train_labels.csv", train_labels_url) # Download training set labels
        # Get labels for test set
    test_labels_url = 'https://rbouadjenek.github.io/datasets/cu_birds_test_labels.csv' # URL for test set labels
    test_labels_path = tf.keras.utils.get_file("cu_birds_test_labels.csv", test_labels_url) # Download test set labels
        # Read labels for training set using pandas
    train_labels_df = pd.read_csv(train_labels_path, sep=",", header=0) # Read training set labels
    train_labels_df = train_labels_df.sample(frac=1).reset_index(drop=True) # Shuffle training set labels. .sample(frac=1) shuffles the dataframe.reset_index(drop=True) resets the index of the shuffled dataframe
        # Read labels for test set using pandas
    test_labels_df = pd.read_csv(test_labels_path, sep=",", header=0) # Read test set labels
    test_labels_df = test_labels_df.sample(frac=1, random_state=1).reset_index(drop=True) # Shuffle test set labels. .sample(frac=1) shuffles the dataframe. .reset_index(drop=True) resets the index of the shuffled dataframe

    # Get the training Labels
    level_0_train_labels = keras.utils.to_categorical(list(train_labels_df['class_level_0'])) # Convert training labels to one-hot encoding
    level_1_train_labels = keras.utils.to_categorical(list(train_labels_df['class_level_1'])) # Convert training labels to one-hot encoding
    level_2_train_labels = keras.utils.to_categorical(list(train_labels_df['class_level_2'])) # Convert training labels to one-hot encoding

    training_labels = [level_0_train_labels, level_1_train_labels, level_2_train_labels] # Combine training labels into a list

    # Get the test Labels
    level_0_test_labels = keras.utils.to_categorical(list(test_labels_df['class_level_0'])) # Convert test labels to one-hot encoding
    level_1_test_labels = keras.utils.to_categorical(list(test_labels_df['class_level_1'])) # Convert test labels to one-hot encoding
    level_2_test_labels = keras.utils.to_categorical(list(test_labels_df['class_level_2'])) # Convert test labels to one-hot encoding

    test_labels = [level_0_test_labels, level_1_test_labels, level_2_test_labels] # Combine test labels into a list

    # Get the training images
    train_images = train_labels_df['fname']
    # Get the test images
    test_images = test_labels_df['fname']

    # Name of the labels
    Level_0_labels_name = list(train_labels_df.groupby('class_level_0')['label_level_0'].unique().apply(', '.join)) # Get the names of the labels for level 0
    Level_1_labels_name = list(train_labels_df.groupby('class_level_1')['label_level_1'].unique().apply(', '.join)) # Get the names of the labels for level 1
    Level_2_labels_name = list(train_labels_df.groupby('class_level_2')['label_level_2'].unique().apply(', '.join)) # Get the names of the labels for level 2
    labels_name = [Level_0_labels_name, Level_1_labels_name, Level_2_labels_name] # Combine the names of the labels into a list

    # Sanity check: number_of_hierarchy_levels should be a list of integers between 0 and 2. If not, raise an error
    if not all(isinstance(item, int) for item in number_of_hierarchy_levels):
        raise ValueError('number_of_hierarchy_levels should be a list of integers between 0 and 2')
    if not all(item in [0,1,2] for item in number_of_hierarchy_levels):
        raise ValueError('number_of_hierarchy_levels should be a list of integers between 0 and 2')
    
    # Match the required labels with number_of_hierarchy_levels. For example, if number_of_hierarchy_levels = [0,1,2] then the labels will be [Level_0_labels_name, Level_1_labels_name, Level_2_labels_name]; if number_of_hierarchy_levels = [0,2] then the labels will be [Level_0_labels_name, Level_2_labels_name]
    training_labels = [training_labels[i] for i in number_of_hierarchy_levels]
    test_labels = [test_labels[i] for i in number_of_hierarchy_levels]
    labels_name = [labels_name[i] for i in number_of_hierarchy_levels]



    dataset = make_dataset(name = dataset_name,
                            training_images = train_images,
                            training_labels = training_labels,
                            test_images = test_images,
                            test_labels = test_labels,
                            labels_name = labels_name,
                            training_validation_split = training_validation_split,
                            validation_split_from = validation_split_from,
                            image_size = image_size,
                            image_type = 'path',
                            dataset_path = dataset_path,
                            batch_size = batch_size,
                            data_normalizing = data_normalizing,
                            class_encoding = class_encoding,
                            data_augmentation = data_augmentation,
                            data_aug_alpha = data_aug_alpha)
    return dataset

def Marine_Tree(dataset_name:str = 'Marine_Tree',
                    dataset_path:str = None, # Path to dataset
                    subtype='Combined', # 'Tropical', 'Temperate', 'Combined'
                    number_of_hierarchy_levels:list = [0,1,2,3,4], # Number of hierarchy levels in the dataset
                    training_validation_split:float = 0.12, # Percentage of test data to be used for validation
                    validation_split_from:str = 'training', # 'training', 'testing'
                    image_size:tuple = (64, 64, 3),
                    batch_size:int = 32,
                    data_normalizing:str = 'StandardScaler', # 'StandardScaler', 'MinMaxScaler', 'None'
                    class_encoding:str = 'one_hot', # 'one_hot', 'label_encoding', 'None'
                    data_augmentation = 'MixUp', # 'MixUp', 'None'
                    data_aug_alpha = 0.2 # Alpha value for MixUp. 0.0 for no MixUp
                    ):
    '''
    Hierarchical dataset for Marine Tree images. The dataset is divided into 5 hierarchical levels.
    This function splits the test data into validation and test sets.
    '''
    # Sanity checks. Check if the dataset_path exists. If not, raise an error.
    if not os.path.exists(dataset_path):
        raise ValueError('dataset_path does not exist.')
    # Get training set. 
    # Check which subtype of the dataset is required. Get the path to the labels for the training and test sets based on the subtype.
    if subtype == 'Tropical':
        train_labels_path = os.path.join(dataset_path, "train_labels_trop.csv")
        test_labels_path = os.path.join(dataset_path, "test_labels_trop.csv")

    elif subtype == 'Temperate':
        train_labels_path = os.path.join(dataset_path, "train_labels_temp.csv")
        test_labels_path = os.path.join(dataset_path, "test_labels_temp.csv")

    elif subtype == 'Combined':
        train_labels_path = os.path.join(dataset_path, "train_labels_comb.csv")
        test_labels_path = os.path.join(dataset_path, "test_labels_comb.csv")

    else:
        raise ValueError('subtype must be one of "Tropical", "Temperate" or "Combined"')


    #>>>>>>>>>>> CHANGE THE PATH TO THE DATASET HERE <<<<<<<<<<<<<<<
    dataset_image_path = os.path.join(dataset_path, "marine_images") # path for dataset image. Add 'marine_images' folder to the path. All the images are stored in this folder

    # Read labels for training set using pandas
    train_labels_df = pd.read_csv(train_labels_path, sep=",", header=0) # Read training set labels
    train_labels_df = train_labels_df.sample(frac=1).reset_index(drop=True) # Shuffle training set labels. .sample(frac=1) shuffles the dataframe. .reset_index(drop=True) resets the index of the shuffled dataframe
    # Read labels for test set using pandas
    test_labels_df = pd.read_csv(test_labels_path, sep=",", header=0) # Read test set labels
    test_labels_df = test_labels_df.sample(frac=1, random_state=1).reset_index(drop=True) # Shuffle test set labels. .sample(frac=1) shuffles the dataframe. .reset_index(drop=True) resets the index of the shuffled dataframe

    # Get the training Labels
    level_0_train_labels = keras.utils.to_categorical(list(train_labels_df['class_level_0'])) # Convert training labels to one-hot encoding
    level_1_train_labels = keras.utils.to_categorical(list(train_labels_df['class_level_1'])) # Convert training labels to one-hot encoding
    level_2_train_labels = keras.utils.to_categorical(list(train_labels_df['class_level_2'])) # Convert training labels to one-hot encoding
    level_3_train_labels = keras.utils.to_categorical(list(train_labels_df['class_level_3'])) # Convert training labels to one-hot encoding
    level_4_train_labels = keras.utils.to_categorical(list(train_labels_df['class_level_4'])) # Convert training labels to one-hot encoding

    training_labels = [level_0_train_labels, level_1_train_labels, level_2_train_labels,level_3_train_labels, level_4_train_labels] # Combine training labels into a list

    # Get the test Labels
    level_0_test_labels = keras.utils.to_categorical(list(test_labels_df['class_level_0'])) # Convert test labels to one-hot encoding
    level_1_test_labels = keras.utils.to_categorical(list(test_labels_df['class_level_1'])) # Convert test labels to one-hot encoding
    level_2_test_labels = keras.utils.to_categorical(list(test_labels_df['class_level_2'])) # Convert test labels to one-hot encoding
    level_3_test_labels = keras.utils.to_categorical(list(test_labels_df['class_level_3'])) # Convert test labels to one-hot encoding
    level_4_test_labels = keras.utils.to_categorical(list(test_labels_df['class_level_4'])) # Convert test labels to one-hot encoding

    test_labels = [level_0_test_labels, level_1_test_labels, level_2_test_labels,level_3_test_labels, level_4_test_labels] # Combine test labels into a list

    # Get the training images
    train_images = train_labels_df['fname']
    # Get the test images
    test_images = test_labels_df['fname']

    # Name of the labels
    Level_0_labels_name = list(train_labels_df.groupby('class_level_0')['label_level_0'].unique().apply(', '.join)) # Get the names of the labels for level 0
    Level_1_labels_name = list(train_labels_df.groupby('class_level_1')['label_level_1'].unique().apply(', '.join)) # Get the names of the labels for level 1
    Level_2_labels_name = list(train_labels_df.groupby('class_level_2')['label_level_2'].unique().apply(', '.join)) # Get the names of the labels for level 2
    Level_3_labels_name = list(train_labels_df.groupby('class_level_3')['label_level_3'].unique().apply(', '.join)) # Get the names of the labels for level 3
    Level_4_labels_name = list(train_labels_df.groupby('class_level_4')['label_level_4'].unique().apply(', '.join)) # Get the names of the labels for level 4

    labels_name = [Level_0_labels_name, Level_1_labels_name, Level_2_labels_name, Level_3_labels_name, Level_4_labels_name] # Combine the names of the labels into a list

    # Sanity check: number_of_hierarchy_levels should be a list of integers between 0 and 4. If not, raise an error
    if not all(isinstance(item, int) for item in number_of_hierarchy_levels):
        raise ValueError('number_of_hierarchy_levels should be a list of integers between 0 and 4')
    if not all(item in [0,1,2,3,4] for item in number_of_hierarchy_levels):
        raise ValueError('number_of_hierarchy_levels should be a list of integers between 0 and 4')
    
    
    # Match the required labels with number_of_hierarchy_levels. For example, if number_of_hierarchy_levels = [0,1,2,3,4] then the labels will be [Level_0_labels_name, Level_1_labels_name, Level_2_labels_name, Level_3_labels_name, Level_4_labels_name]; if number_of_hierarchy_levels = [0,1,4] then the labels will be [Level_0_labels_name, Level_1_labels_name, Level_4_labels_name]

    training_labels = [training_labels[i] for i in number_of_hierarchy_levels]
    test_labels = [test_labels[i] for i in number_of_hierarchy_levels]
    labels_name = [labels_name[i] for i in number_of_hierarchy_levels]

    dataset = make_dataset(name = f'{dataset_name}_{subtype}',
                            training_images = train_images,
                            training_labels = training_labels,
                            test_images = test_images,
                            test_labels = test_labels,
                            labels_name = labels_name,
                            training_validation_split = training_validation_split,
                            validation_split_from = validation_split_from,
                            image_size = image_size,
                            image_type = 'path',
                            dataset_path = dataset_image_path,
                            batch_size = batch_size,
                            data_normalizing = data_normalizing,
                            class_encoding = class_encoding,
                            data_augmentation = data_augmentation,
                            data_aug_alpha = data_aug_alpha)
    return dataset
    
def print_hierarchical_ds_sample(dataset, print_batch_size:int = 4, show_images:bool = True):
    '''
    The function prints sample images from the hierarchical dataset.
    ::dataset:: The dataset object. It should be a tf.data.Dataset object.
                the dataset object should have the following attributes:
                (x,(y1,y2,...,yn)), where x is the image and y1,y2,...,yn are the labels at different hierarchy levels
    '''
    # unbatch the dataset and rebatch the dataset with a smaller batch size
    for x,y in dataset.unbatch().batch(print_batch_size).take(1):
        for i in range(len(x)):
            print('Example = ', i)
            for j in range(len(y)):
                print(f'Level_{j} =', {k:v for k,v in enumerate(y[j][i].numpy()) if v != 0})
            if show_images:
                plt.imshow(x[i])
                plt.show()

def _load_(dataset_name:str, args, **kwargs):
    '''
    Load the dataset based on the dataset_name
    '''
    valid_datasets = [
                        'MNIST',
                        'E_MNIST',
                        'F_MNIST',
                        'CIFAR10',
                        'CIFAR100',
                        'S_Cars',
                        'CU_Birds',
                        'M_Tree',
                        'M_Tree_L4',
                        'M_Tree_L3',
                        'M_Tree_L2',
                        'M_Tree_L1',
                        ]
    # print(kwargs)
    if dataset_name == 'MNIST':
        if args.input_size != (28, 28, 1):
            warnings.warn(f"Image size for MNIST should be (28, 28, 1). Changing image size to (28, 28, 1)!\nargs.input_size value will be updated to (28, 28, 1)", UserWarning)
            args.input_size = kwargs['image_size'] = (28, 28, 1)
        return MNIST(**kwargs)
    elif dataset_name == 'E_MNIST':
        if args.input_size != (28, 28, 1):
            warnings.warn(f"Image size for E_MNIST should be (28, 28, 1). Changing image size to (28, 28, 1)!\nargs.input_size value will be updated to (28, 28, 1)", UserWarning)
            args.input_size = kwargs['image_size'] = (28, 28, 1)
        return EMNIST(**kwargs)
    elif dataset_name == 'F_MNIST':
        if args.input_size != (28, 28, 1):
            warnings.warn(f"Image size for F_MNIST should be (28, 28, 1). Changing image size to (28, 28, 1)!\nargs.input_size value will be updated to (28, 28, 1)", UserWarning)
            args.input_size = kwargs['image_size'] = (28, 28, 1)
        return Fashion_MNIST(**kwargs)
    elif dataset_name == 'CIFAR10':
        if args.input_size != (32, 32, 3):
            warnings.warn(f"Image size for CIFAR10 should be (32, 32, 3). Changing image size to (32, 32, 3)!\nargs.input_size value will be updated to (32, 32, 3)", UserWarning)
            args.input_size = kwargs['image_size'] = (32, 32, 3)
        return CIFAR_10(**kwargs)
    elif dataset_name == 'CIFAR100':
        if args.input_size != (32, 32, 3):
            warnings.warn(f"Image size for CIFAR100 should be (32, 32, 3). Changing image size to (32, 32, 3)!\nargs.input_size value will be updated to (32, 32, 3)", UserWarning)
            args.input_size = kwargs['image_size'] = (32, 32, 3)
        return CIFAR_100(**kwargs)
    elif dataset_name == 'S_Cars':
        return Stanford_Cars(**kwargs)
    elif dataset_name == 'CU_Birds':
        return CU_Birds_200_2011(**kwargs)
    elif dataset_name == 'M_Tree':
        if not os.path.exists(args.data_path):
            raise ValueError('dataset_path does not exist. Please download the dataset and provide the path to the dataset')
        return Marine_Tree(dataset_path=args.data_path, **kwargs)    
    elif dataset_name == 'M_Tree_L4':
        if not os.path.exists(args.data_path):
            raise ValueError('dataset_path does not exist. Please download the dataset and provide the path to the dataset')
        return Marine_Tree(dataset_path=args.data_path, number_of_hierarchy_levels = [0,1,2,3], **kwargs)
    elif dataset_name == 'M_Tree_L3':
        if not os.path.exists(args.data_path):
            raise ValueError('dataset_path does not exist. Please download the dataset and provide the path to the dataset')
        return Marine_Tree(dataset_path=args.data_path, number_of_hierarchy_levels = [0,1,2], **kwargs)
    elif dataset_name == 'M_Tree_L2':
        if not os.path.exists(args.data_path):
            raise ValueError('dataset_path does not exist. Please download the dataset and provide the path to the dataset')
        return Marine_Tree(dataset_path=args.data_path, number_of_hierarchy_levels = [0,1], **kwargs)
    elif dataset_name == 'M_Tree_L1':
        if not os.path.exists(args.data_path):
            raise ValueError('dataset_path does not exist. Please download the dataset and provide the path to the dataset')
        return Marine_Tree(dataset_path=args.data_path, number_of_hierarchy_levels = [0], **kwargs)
    else:
        raise ValueError(f"Invalid dataset_name '{dataset_name}'. dataset_name should be one of {', '.join(valid_datasets)}")


