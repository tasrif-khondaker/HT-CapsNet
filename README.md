# Taxonomy-Guided Routing in Capsule Network for Hierarchical Image Classification

This is the official implementation of the paper titled "Taxonomy-Guided Routing in Capsule Network for Hierarchical Image Classification" Khondaker Tasrif Noor, Wei Luo, Antonio Robles-Kelly, Leo Yu Zhang, and Mohamed Reda Bouadjenek. The paper is currently under review on Knlowledge-Based Systems. Preprint available at [SSRN](https://ssrn.com/abstract=5127434).

## Abstract
Hierarchical multi-label classification in computer vision presents significant challenges in maintaining consistency across different levels of class granularity while capturing fine-grained visual details. This paper presents Taxonomy-aware Capsule Network (HT-CapsNet), a novel capsule network architecture that explicitly incorporates taxonomic relationships into its routing mechanism to address these challenges. Our key innovation lies in a taxonomy-aware routing algorithm that dynamically adjusts capsule connections based on known hierarchical relationships, enabling more effective learning of hierarchical features while enforcing taxonomic consistency. Extensive experiments on six benchmark datasets, including Fashion-MNIST, Marine-Tree, CIFAR-10, CIFAR-100, CUB-200-2011, and Stanford Cars, demonstrate that HT-CapsNet significantly outperforms existing methods across various hierarchical classification metrics. Notably, on CUB-200-2011, HT-CapsNet achieves absolute improvements of $10.32\%$, $10.2\%$, $10.3\%$, and $8.55\%$ in hierarchical accuracy, F1-score, consistency, and exact match, respectively, compared to the best-performing baseline. On the Stanford Cars dataset, the model improves upon the best baseline by $21.69\%$, $18.29\%$, $37.34\%$, and $19.95\%$ in the same metrics, demonstrating the robustness and effectiveness of our approach for complex hierarchical classification tasks.

## Architecture of the proposed Hierarchical Taxonomy-aware Capsule Network (HT-CapsNet)
The network consists of a feature extraction backbone, and for each hierarchical level $l$, one primary capsule layer $(P_{l})$ and one taxonomy-aware secondary capsule layer $(S_{l})$. The primary capsules are reshaped from the feature maps extracted by the backbone network. Each secondary capsule layer is formed using the corresponding level's primary capsules and the output from the previous secondary layer. Connections between secondary capsules represent hierarchical relationships defined by the taxonomy. Primary capsules are not connected across layers, as each layer's $P_l$ is independently derived from the shared feature maps and used solely as input to its corresponding secondary capsule layer.
![Fig: Architecture of the proposed Hierarchical Taxonomy-aware Capsule Network (HT-CapsNet)](src/model_arch/FIG-Architecture.png?raw=true "Architecture of the proposed Hierarchical Taxonomy-aware Capsule Network (HT-CapsNet)")
The routing process between capsules ($P_{l}$ to $S_{l}$ and $S_{l}$ to $S_{l+1}$) is guided by the taxonomy-aware routing mechanism (Algorithm~\ref{alg:htrcapsule}) to enforce hierarchical consistency. Final predictions are obtained from the normalised lengths of secondary capsule vectors. The network is trained end-to-end with a multi-level loss incorporating classification and hierarchical consistency constraints.

## Requirements:
This code is developed and tested with `Python 3.8.10` and `TensorFlow 2.8.0`. You can create a virtual environment and install the required packages using the following command:

```bash
conda env create --file conda_env.yml
```

## Datasets
The datasets used in this project are publicly available. The `src\hierarchical_dataset.py` file contains the code to download and preprocess the datasets. The datasets are as follows:
- **Fashion-MNIST**
- **Marine-Tree**
- **CIFAR-10**
- **CIFAR-100**
- **CUB-200-2011**
- **Stanford Cars**

Note: Additional datasets can be added by modifying the `src\hierarchical_dataset.py` file. The datasets should be in the same format as the existing datasets.

## Training

To train the model, run the `main.ipynb` notebook. The notebook contains the code to train the model on the specified dataset. You can modify the hyperparameters and other settings in the notebook.

The `args_dict{}` dictionary contains the hyperparameters and other settings for the training process. You can modify the following parameters in the `args_dict{}` dictionary:
- `dataset`: The dataset to be used for training. Options are `Fashion-MNIST`, `Marine-Tree`, `CIFAR-10`, `CIFAR-100`, `CUB-200-2011`, and `Stanford Cars`.
- `batch_size`: The batch size for training.
- `epochs`: The number of epochs for training.
- `Routing_N`: The number of routing iterations.
Check the `args_dict{}` dictionary in the `main.ipynb` notebook for more hyperparameters and settings.

To execute the notebook, you can use Jupyter Notebook or Jupyter Lab. If you don't have Jupyter installed, you can install it using the following command:

```bash
pip install jupyter
```
You can then run the notebook by opening it in Jupyter Notebook or Jupyter Lab and executing the cells one by one.
Alternatively, you can run the notebook from the command line using `papermill`. If you don't have `papermill` installed, you can install it using the following command:
```bash
pip install papermill
```
Then, you can run the notebook using the following command:
```bash
papermill main.ipynb main_output.ipynb -p args_dict '{
                                                    "dataset": "CIFAR-10",
                                                    "batch_size": 32,
                                                    "epochs": 100,
                                                    "Routing_N": 3,
                                                    ...
                                                }'
```

or you can use the `run-HT-CapsNet.sh` script to run the notebook with the specified parameters. The script will create a new notebook with the output of the training process.

```bash
bash run-HT-CapsNet.sh
```

The training process will save the model checkpoints and logs in the `logs` directory. You can monitor the training process using TensorBoard by running the following command:

```bash
tensorboard --logdir logs
```

## Evaluation
To evaluate the model, run the `main.ipynb` notebook after training: providing `'Test_only':'BOOL_FLAG'` in the `args_dict{}` dictionary. The notebook contains the code to evaluate the model on the test set. The evaluation process will save the evaluation metrics in the `logs` directory. You can visualize the evaluation metrics using TensorBoard by running the following command:

```bash
tensorboard --logdir logs
```
## Citation
If you find this code useful, please consider citing our paper:

```bibtex
@article{noor2023taxonomy,
  title={Taxonomy-Guided Routing in Capsule Network for Hierarchical Image Classification},
  author={Noor, Khondaker Tasrif and Luo, Wei and Robles-Kelly, Antonio and Zhang, Leo Yu and Bouadjenek, Mohamed Reda},
  journal={Available at SSRN 5127434},
  year={2025},
  publisher={SSRN},
  doi={https://dx.doi.org/10.2139/ssrn.5127434},
}
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.