import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix, multilabel_confusion_matrix, classification_report, ConfusionMatrixDisplay, average_precision_score
from scipy.stats import hmean
from treelib import Tree
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import os


def get_top_k_accuracy_score(y_true: list, y_pred: list, k=1):
    if len(list(y_pred[0])) == 2:
        if k == 1:
            return accuracy_score(y_true, np.argmax(y_pred, axis=1))
        else:
            return 1
    else:
        return top_k_accuracy_score(y_true, y_pred, k=k)


def get_top_k_taxonomical_accuracy(y_true: list, y_pred: list, k=1):
    """
    This method computes the top k accuracy for each level in the taxonomy.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :return: accuracy for each level of the taxonomy.
    :rtype: list
    """
    if len(y_true) != len(y_pred):
        raise Exception('Size of the inputs should be the same.')
    accuracy = [get_top_k_accuracy_score(y_, y_pred_, k) for y_, y_pred_ in zip(y_true, y_pred)]
    return accuracy


def get_h_accuracy(y_true: list, y_pred: list, k=1):
    """
    This method computes the harmonic mean of accuracies of all level in the taxonomy.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :return: accuracy for each level of the taxonomy.
    :rtype: list
    """
    return hmean(get_top_k_taxonomical_accuracy(y_true, y_pred, k))


def get_m_accuracy(y_true: list, y_pred: list, k=1):
    """
    This method computes the arithmetic mean of accuracies of all level in the taxonomy.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :return: accuracy for each level of the taxonomy.
    :rtype: list
    """
    return np.mean(get_top_k_taxonomical_accuracy(y_true, y_pred, k))


def get_exact_match(y_true: list, y_pred: list, argmax_true= False, only_array=False):
    """
    This method compute the exact match score. Exact match is defined as the #of examples for
    which the predictions for all level in the taxonomy is correct by the total #of examples.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :return: the exact match value
    :rtype: float
    """
    # MAKE TRUE AND PREDICTION INTO LIST OF LIST
    if argmax_true is True:
        y_true = [np.argmax(x, axis=1) for x in y_true]
        # y_true_argmax[x]=np.argmax(y_true[x], axis=1).tolist()

    y_pred = [np.argmax(x, axis=1) for x in y_pred]
    if len(y_true) != len(y_pred):
        raise Exception('Shape of the inputs should be the same')
    exact_match = []
    for j in range(len(y_true[0])):
        v = 1
        for i in range(len(y_true)):
            if y_true[i][j] != y_pred[i][j]:
                v = 0
                break
        exact_match.append(v)
    if only_array is True:
        return exact_match
    return np.mean(exact_match)


def get_consistency(y_pred: list, tree: Tree,only_array=False):
    """
    This methods estimates the consistency.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: np.array
    :param tree: A tree of the taxonomy.
    :type tree: Tree
    :return: value of consistency.
    :rtype: float
    """
    y_pred = [np.argmax(x, axis=1) for x in y_pred]
    consistency = []
    for j in range(len(y_pred[0])):
        v = 1
        for i in range(len(y_pred) - 1):
            parent = 'L' + str(i) + '_' + str(y_pred[i][j])
            child = 'L' + str(i + 1) + '_' + str(y_pred[i + 1][j])
            if tree.parent(child).identifier != parent:
                v = 0
                break
        consistency.append(v)
    if only_array is True:
        return consistency
    return np.mean(consistency)


def get_mAP_Score(y_true: list, y_pred: list):
    if len(y_true) != len(y_pred):
        raise Exception('Size of the inputs should be the same.')
    # print(y_pred.shape)
    mAP_score = [average_precision_score(y_, y_pred_) for y_, y_pred_ in zip(y_true, y_pred)]
    return mAP_score

def get_h_mAP_score(y_true: list, y_pred: list):
    """
    This method computes the mean Average precision for all the hierarchical levels and take the harmonic mean.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :return: accuracy for each level of the taxonomy.
    :rtype: list
    """
    return hmean(get_mAP_Score(y_true, y_pred))

def get_m_mAP_score(y_true: list, y_pred: list):
    """
    This method computes the mean Average precision for all the hierarchical levels and take the arithmetic mean.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :return: accuracy for each level of the taxonomy.
    :rtype: list
    """
    return np.mean(get_mAP_Score(y_true, y_pred))


def get_hierarchical_metrics(y_true: list, y_pred: list, tree: Tree):
    """
    This method compute the hierarchical precision/recall/F1-Score. For more details, see:

    Kiritchenko S., Matwin S., Nock R., Famili A.F. (2006) Learning and Evaluation
    in the Presence of Class Hierarchies: Application to Text Categorization. In: Lamontagne L.,
    Marchand M. (eds) Advances in Artificial Intelligence. Canadian AI 2006. Lecture Notes in
    Computer Science, vol 4013. Springer, Berlin, Heidelberg. https://doi.org/10.1007/11766247_34

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :param tree: A tree of the taxonomy.
    :type tree: Tree
    :return: the hierarchical precision/recall/F1-Score values
    :rtype: float
    """
    y_pred = [np.argmax(x, axis=1) for x in y_pred]

    if len(y_true) != len(y_pred):
        raise Exception('Shape of the inputs should be the same')
    hP_list = []
    hR_list = []
    hF1_list = []
    for j in range(len(y_true[0])):
        y_true_aug = set()
        y_pred_aug = set()
        for i in range(len(y_true)):
            true_c = 'L' + str(i) + '_' + str(y_true[i][j])
            y_true_aug.add(true_c)
            while tree.parent(true_c) != None:
                true_c = tree.parent(true_c).identifier
                y_true_aug.add(true_c)

            pred_c = 'L' + str(i) + '_' + str(y_pred[i][j])
            y_pred_aug.add(pred_c)
            while tree.parent(pred_c) != None:
                pred_c = tree.parent(pred_c).identifier
                y_pred_aug.add(pred_c)

        y_true_aug.remove('root')
        y_pred_aug.remove('root')

        hP = len(y_true_aug.intersection(y_pred_aug)) / len(y_pred_aug)
        hR = len(y_true_aug.intersection(y_pred_aug)) / len(y_true_aug)
        if 2 * hP + hR != 0:
            hF1 = 2 * hP * hR / (hP + hR)
        else:
            hF1 = 0

        hP_list.append(hP)
        hR_list.append(hR)
        hF1_list.append(hF1)
    return np.mean(hP_list), np.mean(hR_list), np.mean(hF1_list)


def performance_report(y_true: list, y_pred: list, tree: Tree, title=None):
    """
        Build a text report showing the main classification metrics.

        :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
        :type y_pred: list
        :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
        :type y_true: list
        :param tree: A tree of the taxonomy.
        :type tree: Tree
        :param title: A title for the report.
        :type title: str
        :return: the hierarchical precision/recall/F1-Score values
        :rtype: float
        """
    y_true_argmax=[[] for x in range(len(y_true))]
    for x in range(len(y_true)):
        y_true_argmax[x]=np.argmax(y_true[x], axis=1).tolist()
    accuracy = get_top_k_taxonomical_accuracy(y_true_argmax, y_pred)
    exact_match = get_exact_match(y_true_argmax, y_pred)
    consistency = get_consistency(y_pred, tree)
    hP, hR, hF1 = get_hierarchical_metrics(y_true_argmax, y_pred, tree)
    HarmonicM_Accuracy_k1 = get_h_accuracy(y_true_argmax, y_pred, k=1)
    HarmonicM_Accuracy_k2 = get_h_accuracy(y_true_argmax, y_pred, k=2)
    HarmonicM_Accuracy_k5 = get_h_accuracy(y_true_argmax, y_pred, k=5)
    ArithmeticM_Accuracy_k1 = get_m_accuracy(y_true_argmax, y_pred, k=1)
    ArithmeticM_Accuracy_k2 = get_m_accuracy(y_true_argmax, y_pred, k=2)
    ArithmeticM_Accuracy_k5 = get_m_accuracy(y_true_argmax, y_pred, k=5)
    Harmonic_mAP_Score = get_h_mAP_score(y_true, y_pred)
    Arithmetic_mAP_Score = get_m_mAP_score(y_true, y_pred)

    out={}

    row = []
    for i in range(len(accuracy)):
        row.append('Accuracy L_' + str(i))
        row.append("{:.4f}".format(accuracy[i]))
        out['Accuracy L_' + str(i)] = accuracy[i]
    out = {**out, **{
                    'HarmonicM_Accuracy_k1': HarmonicM_Accuracy_k1,
                    'HarmonicM_Accuracy_k2': HarmonicM_Accuracy_k2,
                    'HarmonicM_Accuracy_k5': HarmonicM_Accuracy_k5,
                    'ArithmeticM_Accuracy_k1': ArithmeticM_Accuracy_k1,
                    'ArithmeticM_Accuracy_k2': ArithmeticM_Accuracy_k2,
                    'ArithmeticM_Accuracy_k5': ArithmeticM_Accuracy_k5,
                    'Harmonic_mAP_Score': Harmonic_mAP_Score,
                    'Arithmetic_mAP_Score': Arithmetic_mAP_Score,
                    'hP': hP, 
                    'hR': hR, 
                    'hF1': hF1,
                    'consistency': consistency,
                    'exact_match': exact_match
                    }
           }
    
    return out



def lvl_wise_metric(y_true: list, y_pred: list,savedir:str=None,show_graph:bool=True,show_report:bool=True):
    if len(y_true) < 1:
        raise ValueError("Invalid length of y_true. At least one level is required.")

    level_classes_numbers = [len(np.unique(np.argmax(y, axis=1))) for y in y_true]
    level_labels = [list(range(0, num_classes)) for num_classes in level_classes_numbers]
    level_target_names = [[str(x) for x in range(0, num_classes)] for num_classes in level_classes_numbers]

    for level, (y_true_level, y_pred_level) in enumerate(zip(y_true, y_pred)):
        # print('\033[91m', '\033[1m', "\u2022", f'Confusion_Matrix Level = {level}', '\033[0m')
        # print(confusion_matrix(np.argmax(y_true_level, axis=1), np.argmax(y_pred_level, axis=1)))

        print(f'\n\033[91m \033[1m \u2022 Classification Report & confusion matrix for Level = {level} Below:\033[0m' if show_graph is True else f'\n\033[91m \033[1m \u2022 Classification Report & confusion matrix for Level = {level} saving @ {savedir}\033[0m')
        confusion_matrixDisplay(np.argmax(y_true_level, axis=1), np.argmax(y_pred_level, axis=1), level_target_names[level],savedir,show_graph)
        report = classification_report(
                                        np.argmax(y_true_level, axis=1),
                                        np.argmax(y_pred_level, axis=1),
                                        target_names=level_target_names[level],
                                        digits=5
                                    )
        print(report if show_report is True else '')
        if savedir:
            report_file = os.path.join(savedir, f'classification_report_LVL_{level}.txt')
            with open(report_file, 'w') as f:
                f.write(f"Classification Report for Level {level}\n\n")
                f.write(report)
            print(f"Classification report for level {level} saved to {report_file}")

def confusion_matrixDisplay(y_true, y_pred, target_names,savedir,show_graph):
    size_of_fig = len(target_names)/2
    # size_of_fig = 100
    if size_of_fig < 7:
        size_of_fig = 7
    labels = target_names
    cm = confusion_matrix(y_true, y_pred)
    cmp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(size_of_fig,size_of_fig))
    cmp.plot(ax=ax)
    if savedir is not None:
        dpi_val = 1080/size_of_fig
        plt.savefig(os.path.join(savedir,f'LVL_Len_{len(target_names)}.png'), dpi=dpi_val)
    if show_graph is True:
        plt.show()
    else:
        plt.close()    

def get_in_or_consistent_chunk(Features: list, 
                                TrueLabels: list, 
                                Predictions: list, 
                                return_type: tuple = (0, 0), 
                                Number_samples=None,
                                sample_level: int = None,
                                specific_class: int = None,
                                correct_at_level: int = None,
                                samples_per_class: int = None,
                                label_tree: Tree = None,
                                ):
    """
    Get Consistent or Inconsistent Predictions and Features
    Inputs:
        Features: List of Features
        TrueLabels: List of True Labels
        Predictions: List of Predictions
        return_type: Tuple of two integers (consistent, inconsistent)
                        (0, 0) => Not Consistent and Not Exact Match
                        (0, 1) => Not Consistent and Exact Match # Not Possible
                        (1, 0) => Consistent and Not Exact Match
                        (1, 1) => Consistent and Exact Match
        Number_samples: Number of samples to return; None for all samples
        sample_level: Sample at a specific level; To be used with specific_class
        specific_class: Specific class at a specific level; To be used with sample_level
        correct_at_level: Get the samples where the prediction is correct at a specific level
        samples_per_class: Get the samples where each class has samples_per_class samples at a specific level
        label_tree: Tree of the Labels
    Outputs:
        Consistent/Inconsistent Features, TrueLabels, Predictions
    """
    # get consistent and inconsistent Predictions
    consistency = get_consistency(Predictions, label_tree, only_array=True)
    exact_match = get_exact_match(TrueLabels, Predictions, argmax_true = True, only_array=True)
    indices = np.where((np.array(consistency) == return_type[0]) & (np.array(exact_match) == return_type[1]))[0]
    if sample_level is not None and specific_class is not None:
        # sanity checkpoint
        assert sample_level < len(TrueLabels), "Level should be less than the number of levels"
        assert specific_class < len(np.unique(np.argmax(TrueLabels[sample_level], axis=1))), "Specific Class should be less than the number of classes at the level"
        # get the positions where the class is specific_class at level
        indices = np.where((np.array(consistency) == return_type[0]) & (np.array(exact_match) == return_type[1]) & (np.argmax(TrueLabels[sample_level], axis=1) == specific_class))[0]
    if Number_samples is not None:
        if len(indices) < Number_samples:
            print(f"Number of samples are less than the required samples. Returning all {len(indices)} samples")
        else:
            indices = indices[:Number_samples]
    if samples_per_class is not None:
        assert sample_level is not None, "Sample Level should be provided"
        assert Number_samples is None, "Number of samples should not be provided"
        assert specific_class is None, "Specific Class should not be provided"
        # Filter the indices so that TrueLabels[sample_level] have samples_per_class samples for each class
        # print(indices)
        filtered_indices = []
        unique_classes = np.unique(np.argmax(TrueLabels[sample_level], axis=1))
        for cls in unique_classes:
            class_indices = indices[np.where(np.argmax(TrueLabels[sample_level], axis=1)[indices] == cls)[0]]
            if len(class_indices) > samples_per_class:
                class_indices = class_indices[:samples_per_class]
                filtered_indices.extend(class_indices)
        indices = np.array(filtered_indices)

    # Correct @ Level
    if correct_at_level is not None:
        # Get the indenes for a specific level where the prediction is correct. true label and prediction should be same
        indices = indices[np.where(np.argmax(TrueLabels[correct_at_level], axis=1)[indices] == np.argmax(Predictions[correct_at_level], axis=1)[indices])[0]]
        
    features = [array[indices] for array in Features]
    true_labels = [array[indices] for array in TrueLabels]
    predictions = [array[indices] for array in Predictions]

    return features, true_labels, predictions

def get_image_in_or_consistent_chunk(Images: list, 
                                TrueLabels: list, 
                                Predictions: list, 
                                return_type: tuple = (0, 0), 
                                Number_samples=None,
                                sample_level: int = None,
                                specific_class: int = None,
                                correct_at_level: int = None,
                                samples_per_class: int = None,
                                label_tree: Tree = None,
                                ):
    """
    Get Consistent or Inconsistent Predictions and Features
    Inputs:
        Features: List of Features
        TrueLabels: List of True Labels
        Predictions: List of Predictions
        return_type: Tuple of two integers (consistent, inconsistent)
                        (0, 0) => Not Consistent and Not Exact Match
                        (0, 1) => Not Consistent and Exact Match # Not Possible
                        (1, 0) => Consistent and Not Exact Match
                        (1, 1) => Consistent and Exact Match
        Number_samples: Number of samples to return; None for all samples
        sample_level: Sample at a specific level; To be used with specific_class
        specific_class: Specific class at a specific level; To be used with sample_level
        correct_at_level: Get the samples where the prediction is correct at a specific level
        samples_per_class: Get the samples where each class has samples_per_class samples at a specific level
        label_tree: Tree of the Labels
    Outputs:
        Consistent/Inconsistent Features, TrueLabels, Predictions
    """
    # get consistent and inconsistent Predictions
    consistency = get_consistency(Predictions, label_tree, only_array=True)
    exact_match = get_exact_match(TrueLabels, Predictions, argmax_true = True, only_array=True)
    indices = np.where((np.array(consistency) == return_type[0]) & (np.array(exact_match) == return_type[1]))[0]
    if sample_level is not None and specific_class is not None:
        # sanity checkpoint
        assert sample_level < len(TrueLabels), "Level should be less than the number of levels"
        assert specific_class < len(np.unique(np.argmax(TrueLabels[sample_level], axis=1))), "Specific Class should be less than the number of classes at the level"
        # get the positions where the class is specific_class at level
        indices = np.where((np.array(consistency) == return_type[0]) & (np.array(exact_match) == return_type[1]) & (np.argmax(TrueLabels[sample_level], axis=1) == specific_class))[0]
    if Number_samples is not None:
        if len(indices) < Number_samples:
            print(f"Number of samples are less than the required samples. Returning all {len(indices)} samples")
        else:
            indices = indices[:Number_samples]
    if samples_per_class is not None:
        assert sample_level is not None, "Sample Level should be provided"
        assert Number_samples is None, "Number of samples should not be provided"
        assert specific_class is None, "Specific Class should not be provided"
        # Filter the indices so that TrueLabels[sample_level] have samples_per_class samples for each class
        # print(indices)
        filtered_indices = []
        unique_classes = np.unique(np.argmax(TrueLabels[sample_level], axis=1))
        for cls in unique_classes:
            class_indices = indices[np.where(np.argmax(TrueLabels[sample_level], axis=1)[indices] == cls)[0]]
            if len(class_indices) > samples_per_class:
                class_indices = class_indices[:samples_per_class]
                filtered_indices.extend(class_indices)
        indices = np.array(filtered_indices)

    # Correct @ Level
    if correct_at_level is not None:
        # Get the indenes for a specific level where the prediction is correct. true label and prediction should be same
        indices = indices[np.where(np.argmax(TrueLabels[correct_at_level], axis=1)[indices] == np.argmax(Predictions[correct_at_level], axis=1)[indices])[0]]
        
    images = Images[indices]
    true_labels = [array[indices] for array in TrueLabels]
    predictions = [array[indices] for array in Predictions]

    return images, true_labels, predictions

def hmeasurements(y_true: list,
                  y_pred: list,
                  tree):
    y_true_argmax=[[] for x in range(len(y_true))]
    for x in range(len(y_true)):
        y_true_argmax[x]=np.argmax(y_true[x], axis=1).tolist()
    h_measurements = get_hierarchical_metrics(y_true_argmax,y_pred,tree)
    consistency = get_consistency(y_pred, tree)
    exact_match = get_exact_match(y_true_argmax, y_pred)
    get_performance_report = performance_report(y_true, y_pred, tree)
    return h_measurements,consistency,exact_match, get_performance_report
    
    
if __name__ == '__main__':
    y = [[1, 0, 1, 0, 0], [1, 2, 3, 4, 0], [3, 4, 5, 8, 0]]

