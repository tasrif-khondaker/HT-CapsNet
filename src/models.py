import tensorflow as tf
from tensorflow import keras
import platform
from .model_arch import capsnet
from .model_arch import HTRCapsNet
import os
import csv
import numpy as np

def initial_lw(class_labels: list):
    """
    This function takes a list as input,
        - where each element in the list represents the number of classes at a specific hierarchical level
        - the order of the elements in the list corresponds to the hierarchical level.
    
    The function returns a list of initial loss weight values, calculated based on the number of classes in each level relative to the total across all levels.
        - The loss weights are calculated as the relative proportion of classes at each level.
            - The higher the number of classes at a level, the lower the loss weight.
            - The lower the number of classes at a level, the higher the loss weight.
        - The loss weights are normalized to sum to 1.
    
    Example: 
        For a hierarchy with 3 levels and respective classes [2, 7, 10], the function will return a [0.4473684210526316, 0.3157894736842105, 0.2368421052631579].
        - if [10, 9, 7] is provided, return [0.3076923076923077, 0.3269230769230769, 0.3653846153846154]
    
    :param class_labels: List of integers representing number of classes per level
    :return: List of initial loss weight values corresponding to each level
    """
    
    # Step 1: Calculate the relative proportion of classes for each level
    q = [(1 - (v / sum(class_labels))) for v in class_labels]
    
    # Step 2: Normalize the proportions to get the loss weights
    total_q_sum = sum(q)
    c = [x / total_q_sum for x in q]
    
    return c

class dynamic_LW_Modifier(keras.callbacks.Callback):

    def __init__(self, num_classes : list, weight:float=0.0, directory : str = None):

        super(dynamic_LW_Modifier, self).__init__()
        self.initial_lw = [tf.Variable(v,name=f'lw_{i}') for i, v in enumerate(initial_lw(num_classes))] # for keeping initial loss weights; Used in args.LossWeight == 'static' mode
        self.values = [tf.Variable(v,name=f'lw_{i}') for i, v in enumerate(initial_lw(num_classes))] # for keeping updated loss weights; Used in args.LossWeight == 'dynamic' mode
        self.weight = weight # Overall weight for the loss weights (1-weight) will be the accuracy weight

        self.directory = directory
        self.csv_file_epoch_end = os.path.join(self.directory,"LW_Values_epoch_end.csv") if self.directory else None 
        self.csv_file_batch_end = os.path.join(self.directory,"LW_Values_batch_end.csv") if self.directory else None
        
        self.epoch_count = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_count += 1

    def on_epoch_end(self, epoch, logs=None):
        acc = [logs[f'{name}_accuracy'] for name in self.model.output_names]
        val_acc = [logs[f'val_{name}_accuracy'] for name in self.model.output_names]

        # taus = [1.0-acc * self.initial_lw[i] for i, acc in enumerate(acc)]
        # tau_sum = tf.reduce_sum(taus)
        # updated_lw = [(1.0 - self.weight) * (tau/tau_sum) for tau in taus]

        # for i, v in enumerate(updated_lw):
        #     self.values[i].assign(v)

        # Append the results to the CSV file
        if self.csv_file_epoch_end:
            file_exists = os.path.isfile(self.csv_file_epoch_end)
            with open(self.csv_file_epoch_end, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                
                # Write the header only if the file doesn't already exist
                if not file_exists:
                    header = ['Epoch'] + [f'{name}_Acc' for name in self.model.output_names] + [f'{name}_ValAcc' for name in self.model.output_names] + [f'{name}_LW' for name in self.model.output_names]
                    writer.writerow(header)
                
                # Write the epoch data (accuracy and updated loss weights)
                row = [epoch] + acc + val_acc+ [v.numpy() for v in self.values]
                writer.writerow(row)
 

    def on_train_batch_end(self, batch, logs=None):
        acc = [logs[f'{name}_accuracy'] for name in self.model.output_names]
        taus = [1.0-acc * self.initial_lw[i] for i, acc in enumerate(acc)]
        tau_sum = tf.reduce_sum(taus)
        updated_lw = [(1.0 - self.weight) * (tau/tau_sum) for tau in taus]

        for i, v in enumerate(updated_lw):
            self.values[i].assign(v)

        # Append the results to the CSV file
        if self.csv_file_batch_end:
            file_exists = os.path.isfile(self.csv_file_batch_end)
            with open(self.csv_file_batch_end, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                
                # Write the header only if the file doesn't already exist
                if not file_exists:
                    header = ['Epoch'] + ['Batch'] + [f'{name}_Acc' for name in self.model.output_names] + [f'{name}_LW' for name in self.model.output_names]
                    writer.writerow(header)
                
                # Write the batch data (accuracy and updated loss weights)
                row = [self.epoch_count] + [batch] + acc + [v.numpy() for v in self.values]
                writer.writerow(row)

    def on_train_end(self, logs=None):
        self.epoch_count = 0
             
class LR_ExponentialDecay:
    def __init__(self, initial_LR, start_epoch:int = 0, decay_factor:float = 0.9):
        """
        Initialize the scheduler with the initial learning rate and decay parameters.
        Args:
            initial_LR (float): The initial learning rate.
            decay_start_epoch (int): The epoch after which learning rate decay should start.
            decay_factor (float): The factor by which the learning rate decays after each epoch.
        """
        self.initial_LR = initial_LR
        self.decay_start_epoch = start_epoch
        self.decay_factor = decay_factor

    def scheduler(self, epoch, lr):
        """
        Adjusts the learning rate based on the current epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Updated learning rate.
        """
        if epoch < self.decay_start_epoch:
            learning_rate =  self.initial_LR
        else: # Apply decay if the epoch exceeds the decay start threshold
            learning_rate = self.initial_LR * (self.decay_factor ** (epoch - self.decay_start_epoch))
        
        # Log the learning rate using TensorBoard
        tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
        
        return learning_rate

    def get_scheduler_callback(self):
        """
        Returns a Keras LearningRateScheduler callback for training.

        Returns:
            tf.keras.callbacks.LearningRateScheduler: Keras callback to be used during model training.
        """
        return tf.keras.callbacks.LearningRateScheduler(self.scheduler)


def predict_from_pipeline(model, dataset, return_images: bool = False):
    y_true = []
    y_pred = []
    x_data = []
    

    # Initialize empty lists for true and predicted values for each output
    for _ in range(len(model.output)):
        y_true.append([])
        y_pred.append([])

    # Iterate through the dataset
    for x, y in dataset:
        batch_pred = model.predict(x)
        
        # Store the input data (images)
        if return_images:
            x_data.extend(x.numpy().tolist())
        
        # Extend true and predicted values for each output
        for i in range(len(model.output)):
            y_true[i].extend(y[i].numpy().tolist())
            y_pred[i].extend(batch_pred[i].tolist())
    
    # Convert all lists to numpy arrays before returning
    return np.array(x_data), [np.array(y_t) for y_t in y_true], [np.array(y_p) for y_p in y_pred]

def get_model(model_name: str, 
              args: dict,
              **kwargs):
    """
    This function returns a model instance based on the model name provided.

    :param model_name: Name of the model to be loaded
    :param args: Dictionary containing model configuration parameters
    :param kwargs: Additional keyword arguments
    :return: Model instance
    """
    num_classes = kwargs['num_classes']

    SCap_dims = [int(args.SCaps_dim * (2**0)) if args.SCaps_dim_mode == 'same' else int(args.SCaps_dim * (2**i)) if args.SCaps_dim_mode == 'increase' else int(args.SCaps_dim * (2**-i)) if args.SCaps_dim_mode == 'decrease' else ValueError for i, j in enumerate(num_classes)]

    if model_name == 'BUH_CapsNet':
        

        model = capsnet.BUH_CapsNet(input_shape = args.input_size, 
                                    num_classes = num_classes, 
                                    PCaps_dim = args.PCaps_dim,
                                    SCaps_dim = SCap_dims,
                                    backbone_name = args.backbone_net, 
                                    backbone_net_weights = args.backbone_net_weights,
                                    num_blocks = 4,  # for custom backbone
                                    initial_filters = 64,  # for custom backbone
                                    filter_increment = 2, # for custom backbone
                                    routing_iterations = args.Routing_N,
                                    )
            
    elif model_name == 'HD_CapsNet':

        model = capsnet.HD_CapsNet(
                            input_shape = args.input_size, 
                            num_classes = num_classes, 
                            PCaps_dim = args.PCaps_dim,
                            SCaps_dim = SCap_dims,
                            backbone_name = args.backbone_net, 
                            backbone_net_weights = args.backbone_net_weights,
                            num_blocks = 4,  # for custom backbone
                            initial_filters = 64,  # for custom backbone
                            filter_increment = 2, # for custom backbone
                            )
    
    elif model_name == 'HD_CapsNet_EM':
        model = capsnet.HD_CapsNet_EM(
                            input_shape = args.input_size, 
                            num_classes = num_classes, 
                            PCaps_dim = args.PCaps_dim,
                            SCaps_dim = SCap_dims,
                            backbone_name = args.backbone_net, 
                            backbone_net_weights = args.backbone_net_weights,
                            num_blocks = 4,  # for custom backbone
                            initial_filters = 64,  # for custom backbone
                            filter_increment = 2, # for custom backbone
                            )
        
    elif model_name == 'ML_CapsNet':
        model = capsnet.ML_CapsNet(
                            input_shape = args.input_size, 
                            num_classes = num_classes, 
                            PCaps_dim = args.PCaps_dim,
                            SCaps_dim = SCap_dims,
                            backbone_name = args.backbone_net, 
                            backbone_net_weights = args.backbone_net_weights,
                            num_blocks = 4,  # for custom backbone
                            initial_filters = 64,  # for custom backbone
                            filter_increment = 2, # for custom backbone
                            )
        
    elif model_name == 'HDR_CapsNet':
        model = capsnet.HDR_CapsNet(
                            input_shape = args.input_size, 
                            num_classes = num_classes, 
                            PCaps_dim = args.PCaps_dim,
                            SCaps_dim = SCap_dims,
                            backbone_name = args.backbone_net, 
                            backbone_net_weights = args.backbone_net_weights,
                            num_blocks = 4,  # for custom backbone
                            initial_filters = 64,  # for custom backbone
                            filter_increment = 2, # for custom backbone
                            )
                
    elif model_name == 'HD_CapsNet_Eff':
        model = capsnet.HD_CapsNet_Eff(
                            input_shape = args.input_size, 
                            num_classes = num_classes, 
                            PCaps_dim = args.PCaps_dim,
                            SCaps_dim = SCap_dims,
                            backbone_name = args.backbone_net, 
                            backbone_net_weights = args.backbone_net_weights,
                            num_blocks = 4,  # for custom backbone
                            initial_filters = 64,  # for custom backbone
                            filter_increment = 2, # for custom backbone
                            routing_iterations = args.Routing_N,
                            )
        
    elif model_name == 'HTR_CapsNet':
        taxonomy = kwargs.get('taxonomy') or (lambda: (_ for _ in ()).throw(ValueError("Taxonomy not provided for HTR_CapsNet model. Please provide a taxonomy.")))()
        model = HTRCapsNet.HTRCapsNet_model(
                            input_shape = args.input_size, 
                            num_classes = num_classes, 
                            PCaps_dim = args.PCaps_dim,
                            SCaps_dim = SCap_dims,
                            taxonomy= taxonomy,
                            routing_iterations=args.Routing_N,
                            backbone_name = args.backbone_net, 
                            backbone_net_weights = args.backbone_net_weights,
                            num_blocks = 4,  # for custom backbone
                            initial_filters = 64,  # for custom backbone
                            filter_increment = 2, # for custom backbone
                            taxonomy_temperature = args.HTR_taxonomy_temperature,
                            mask_threshold_high = args.HTR_mask_threshold_high,
                            mask_threshold_low = args.HTR_mask_threshold_low,
                            mask_temperature = args.HTR_mask_temperature,
                            mask_center = args.HTR_mask_center,
                            num_heads = args.HTR_att_num_heads,
                            key_dim = args.HTR_att_key_dim,
                            )

    else:
        raise ValueError(f"Model name '{model_name}' not recognized or implemented.")
            

    return model


    