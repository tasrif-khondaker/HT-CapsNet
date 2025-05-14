import os
import sys
import tensorflow as tf


class systeminfo():
    '''
    For printing system information
    '''
    def __init__(self):
    
        self.directory = os.getcwd()
        self.python_ver = sys.version
        self.tf_ver = tf.__version__
        self.keras_ver = tf.keras.__version__
        try:
            self.pcname = os.environ['COMPUTERNAME']
            self.envname = 'Anaconda Environment Name : '+os.environ['CONDA_DEFAULT_ENV']
        except:
            self.pcname = os.name
            self.envname = "Not Using Anaconda"
            

    def __str__(self):
        return ("\033[91m\033[1m\n\u2022 Computer Name = \033[0m{}"
        "\033[91m\033[1m\n\u2022 Working Directory = \033[0m{}"
        "\033[91m\033[1m\n\u2022 Python Version = \033[0m{}"
        "\033[91m\033[1m\n\u2022 TensorFlow Version = \033[0m{}"
        "\033[91m\033[1m\n\u2022 Keras Version = \033[0m{}"
        "\033[91m\033[1m\n\u2022 Current Environment = \033[0m{}"
        .format(self.pcname,
        self.directory,
        self.python_ver,
        self.tf_ver,
        self.keras_ver,
        self.envname))
       
class gpugrowth:
    """
    Limiting GPU growth.
    By default it select only one GPU (GPU:0)
    :INPUT:
    gpus: number of GPUS as string. For 4 GPUs input example: gpus = "0,1,2,3"
    """
    def __init__(self, gpus : str = "0"):
        self.gpu_select = gpus
        try:
            os.environ["CUDA_VISIBLE_DEVICES"]= self.gpu_select ## Include GPU Numbers
            print('Following GPUS are selected = ', self.gpu_select)
        except:
            print('os.environ: GPU selection failed')
    
        self.gpus = tf.config.experimental.list_physical_devices('GPU')
            
    def memory_growth(self):
        if self.gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in self.gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print("Done: GPU "+str(gpu))
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(self.gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    
def log_dir(base_path: list = ['test'], create: bool = True):
    """
    Create a log directory and return the directory as String.
    Structure: 'logs/base_path[0]/base_path[1]/.../base_path[n]'
    Tree:
            logs <<<< THIS IS THE ROOT DIRECTORY | IT WILL BE CREATED AUTOMATICALLY IF NOT EXIST >>>>
            ├── base_path[0]
            │   └── base_path[1]
            │       └── ...
            │            └── base_path[n]
    
    - If the directory already exist, it will print a warning.

    :INPUT:
        base_path: list of strings. Example: ['test', 'Dev']
    :RETURN:
        String: Path of the directory. Example: 'logs/test/Dev'
    """
    path = os.path.join('logs',*[str(x) for x in base_path])
    if create:
        if not os.path.exists(path):
            os.makedirs(path)
            print("FOLDER CREATED = ", path)
        else:
            print("Warning: Folder already exist.")
    else:
        if not os.path.exists(path):
            print(f'Warning: Folder does not exist. Create = {create}. Path = {path}')
        else:
            print(f'Folder exist. Create = {create}. Path = {path}')
    return path

def multi_gpu_select(os_system : str = 'windows'):
    if os_system == 'windows':
        #Multiple GPU ------ Windows
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    elif os_system == 'linux':
        #Multiple GPU ------ LINUX
        strategy = tf.distribute.MirroredStrategy()
    return strategy
