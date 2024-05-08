import numpy as np
import math
from collections import Counter
import time


class DecisionNode:
    """Class to represent a nodes or leaves in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """
        Create a decision node with eval function to select between left and right node
        NOTE In this representation 'True' values for a decision take us to the left.
        This is arbitrary, but testing relies on this implementation.
        Args:
            left (DecisionNode): left child node
            right (DecisionNode): right child node
            decision_function (func): evaluation function to decide left or right
            class_label (value): label for leaf node
        """
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Determine recursively the class of an input array by testing a value
           against a feature's attributes values based on the decision function.

        Args:
            feature: (numpy array(value)): input vector for sample.

        Returns:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice index for data labels.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data contained in the ReadMe.
    It must be built fully starting from the root.

    Returns:
        The root node of the decision tree.
    """
    dt_root = None

    # I drew my decision tree by hand, here is how it is:
    # Start with A0 <= 0.0568 -> dt_root
    # if A0 <= 0.0568 is true then Y=0
    # if A0 <= 0.0568 is false then do, A1<= -0.3122 -> dt_node_1
    # if A1<= -0.3122 is true then do, A2 <= -0.7045 -> dt_node_2
    # if A1<= -0.3122 is false then do, A2 <= -0.7848 -> dt_node_3
    # if A2 <= -0.7045 is true then Y=2
    # if A2 <= -0.7045 is false then Y=0
    # if A2 <= -0.7848 is true then Y=2
    # if A2 <= -0.7848 is false then Y=1

    # Let's create the leaves like example above, by initializing DecisionNode class
    # The leaves are the values we want for each row, so Y=0, Y=1, Y=2

    leaf_node_y0 = DecisionNode(None, None, None, class_label=0)
    leaf_node_y1 = DecisionNode(None, None, None, class_label=1)
    leaf_node_y2 = DecisionNode(None, None, None, class_label=2)

    # unlike example above, we have 9 nodes, so we have to work backwards.
    # We can't simply do dt_root.left or dt_root.right like above

    # I am going to create decision nodes based of my tree I wrote above
    # We will have 3 decision nodes, under the decision tree root node.
    # Again that's why we can't do dt_root.left, etc.

    # Since we are working backwards, we have to initialize dt_node_2 and dt_node_3
    # since we already have the left and right values of those two decision tree nodes

    # What I am doing below is similar to the example given in readme: func = lambda feature : feature[2] <= 0.356
    # As it said there, feature[2] represents A2 attribute. The left node or true statement is both Y=2, from my tree
    # The right node or false statement is y=0 for dt_node_2 and y=1 for dt+node_3, again as shown above.

    dt_node_2 = DecisionNode(leaf_node_y2, leaf_node_y0, lambda feature: feature[2] <= -0.7045, None)
    dt_node_3 = DecisionNode(leaf_node_y2, leaf_node_y1, lambda feature: feature[2] <= -0.7848, None)

    # Now given these two I can get dt_node_1, as dt_node_2 is left of dt_node_1 and dt_node_3 is right.
    # Again here the feature[1] is used since dt_node_1 is checking if A1 <= -0.3122

    dt_node_1 = DecisionNode(dt_node_2, dt_node_3, lambda feature: feature[1] <= -0.3122, None)

    # Now we have all the decision nodes covered below the actual root node.
    # I can just initialize dt_root, with values from above, y=0 will be left of root and dt_node_1 will be right
    # Again here the condition is if A0 <= 0.0568 hence feature[0]

    dt_root = DecisionNode(leaf_node_y0, dt_node_1, lambda feature: feature[0] <= 0.0568, None)

    return dt_root


def confusion_matrix(true_labels, classifier_output, n_classes=2):
    """Create a confusion matrix to measure classifier performance.
   
    Classifier output vs true labels, which is equal to:
    Predicted  vs  Actual Values.
    
    Output will sum multiclass performance in the example format:
    (Assume the labels are 0,1,2,...n)
                                     |Predicted|
                     
    |A|            0,            1,           2,       .....,      n
    |c|   0:  [[count(0,0),  count(0,1),  count(0,2),  .....,  count(0,n)],
    |t|   1:   [count(1,0),  count(1,1),  count(1,2),  .....,  count(1,n)],
    |u|   2:   [count(2,0),  count(2,1),  count(2,2),  .....,  count(2,n)],'
    |a|   .............,
    |l|   n:   [count(n,0),  count(n,1),  count(n,2),  .....,  count(n,n)]]
    
    'count' function is expressed as 'count(actual label, predicted label)'.
    
    For example, count (0,1) represents the total number of actual label 0 and the predicted label 1;
                 count (3,2) represents the total number of actual label 3 and the predicted label 2.           
    
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
    Returns:
        A two dimensional array representing the confusion matrix.
    """
    c_matrix = None

    # from the method non_vectorized_loops method:
    # I was able to use it to build my matrix
    # non_vectorized = np.zeros(data.shape) is what you used
    # After looking at np.zeroes docs: https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    # I noticed what it's doing is creating an matrix filled with 0 values, given the shape

    # Here c_matrix is initialized at top to None, I have to reinitialize to zeroes
    # with the shape 2x2 matrix, so n_classes x n_classes and we want it to be an int as instructed

    c_matrix = np.zeros((n_classes, n_classes), int)
    #print(c_matrix)

    # For next part, it's similar to the same method, except I am not looping through rows and columns
    # Here we are iterating over the true_labels input list, which are correct classified labels
    # and classified_output list, which is correct classified labels
    # As mentioned at the top:
    #     Classifier output vs true labels, which is equal to: Predicted  vs  Actual Values.
    # based of that, Predicted is mapped to classifier output and actual is mapped to true labels

    # Also mentioned above, count(0,1), where count(actual label, predicted label)
    # So for the iterating variables in the loop, we need actual and predicted labels
    # need to assign the values in c_matrix to be [actual][predicted].

    # Once we iterate through actual labels in true_labels and predicted in classified_output,
    # we increment it by 1, when that statement is true at the respective iterative variable.
    # the actual_label and predicted_label have to be an integer, casting them to int just in case.

    for actual_label, predicted_label in zip(true_labels, classifier_output):
        c_matrix[int(actual_label)][int(predicted_label)] += 1
        # print(c_matrix)

    # print(c_matrix)
    return c_matrix


def precision(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the precision of a classifier compared to the correct values.
    In this assignment, precision for label n can be calculated by the formula:
        precision (n) = number of correctly classified label n / number of all predicted label n 
                      = count (n,n) / (count(0, n) + count(1,n) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of precision of each classifier output. 
        So if the classifier is (0,1,2,...,n), the output should be in the below format: 
        [precision (0), precision(1), precision(2), ... precision(n)].
    """

    # Precision: TP / TP + FP
    #  precision (n) = number of correctly classified label n / number of all predicted label n
    #                       = count (n,n) / (count(0, n) + count(1,n) + .... + count (n,n))

    # create an list of precisions, as the return mentioned above is that
    precision_list = []

    # I am going to check if pe_matrix which is supposedly the pre-existing numpy confusion matrix
    # is none, if it's none, then we create an new confusion matrix and set it to c_matrix
    # we have all the required inputs anyway
    # if it's not none, then we set that to be the confusion matrix
    # TLDR: we are checking if the pe_matrix has been passed, if not we create it.

    if pe_matrix is None:
        c_matrix = confusion_matrix(true_labels, classifier_output, n_classes)
        # print(c_matrix)
    else:
        c_matrix = pe_matrix
        # print(c_matrix)

    # now I am going to calculate precision for each class, we will loop over each class in n_classes
    for i in range(n_classes):
        # For each class, its diagonal will tell you how many times your classifier predicted correctly (TP)
        # For matrices, [0][0], [1][1], [2][2], etc. can be considered as an diagonal
        # The classifier predicted correctly is true positive, hence TP and we check our confusion matrix's diagonal

        tp = c_matrix[i][i]
        # print(tp)

        # its column (TP omitted) will inform you how many times your classifier predicted a positive value incorrectly (FP)
        # essentially what we can do to achieve the false positive (FP), is to take the sum of all values in column i
        # of confusion matrix and then subract the true positives from it, to get the difference which is False positives.
        # I used the following as an inspiration for summing all values in an specific column in an matrix:
        # https://www.geeksforgeeks.org/python-summation-of-kth-column-in-a-matrix/

        fp = np.sum(c_matrix[:, i]) - tp
        # print(fp)

        # now we have tp and fp, we can perform the calculation Precision: TP / TP + FP
        # only thing we have to check for is, tp + fp can't be 0, as we can't divide by 0
        # so if tp + fp is greater than 0, precision = tp / tp + fp, precision_value variable will hold the precision value
        # if (tp + fp) isn't greater than 0 for some reason, precision is 0. Again since we can't divide by 0.

        if (tp + fp) > 0:
            precision_value = tp / (tp + fp)
        else:
            precision_value = 0

        # we can now append the precision_value into the list of precisions, at this iteration in the loop.

        # print(precision_value)
        precision_list.append(precision_value)
        # print(precision_list)

    # return the list of precisions
    return precision_list





def recall(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the recall of a classifier compared to the correct values.
    In this assignment, recall for label n can be calculated by the formula:
        recall (n) = number of correctly classified label n / number of all true label n 
                   = count (n,n) / (count(n, 0) + count(n,1) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of recall of each classifier output..
        So if the classifier is (0,1,2,...,n), the output should be in the below format: 
        [recall (0), recall (1), recall (2), ... recall (n)].
    """

    # Recall: TP / TP + FN
    # Recall is pretty much exact same as my implementation for precision except for FN instead of FP
    # where its row (TP omitted) will inform you how many times your classifier did not correctly predict an actual positive (FN)
    # so instead of column in precision for FP, we have to check row here for FN. rest is same,
    # I won't explain the steps I used before in detail like before

    # we initialize the list of recalls similar to precision lists
    recall_list = []

    # we initialize the confusion matrix same as precision, if it hasn't been passed into pe_matrix yet

    if pe_matrix is None:
        c_matrix = confusion_matrix(true_labels, classifier_output, n_classes)
        # print(c_matrix)
    else:
        c_matrix = pe_matrix
        # print(c_matrix)

    # Similar to precision function, we will loop over each class in n_classes
    for i in range(n_classes):
        # going to calculate tp exact same way, no changes here
        tp = c_matrix[i][i]
        # print(tp)

        # only difference here is FN, which as mentioned above, we will take the sum of all values in row i
        # of confusion matrix and then subract the true positives from it, to get the difference which is False positives.

        fn = np.sum(c_matrix[i, :]) - tp
        # print(fn)

        # exact same structure as precision, going to check if tp + fn is greater than 0, to avoid dividing by 0
        # to find Recall: TP / TP + FN -> from readme, going to set it to variable recall_value here instead.

        if (tp + fn) > 0:
            recall_value = tp / (tp + fn)
        else:
            recall_value = 0

        # going to append recall_value to recall_list

        # print(recall_value)
        recall_list.append(recall_value)
        # print(recall_list)

    # return the list of recalls
    return recall_list


def accuracy(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """Get the accuracy of a classifier compared to the correct values.
    Balanced Accuracy Weighted:
    -Balanced Accuracy: Sum of the ratios (accurate divided by sum of its row) divided by number of classes.
    -Balanced Accuracy Weighted: Balanced Accuracy with weighting added in the numerator and denominator.

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The accuracy of the classifier output.
    """

    # Accuracy TP + TN / TP + TN + FP + FN
    # From Recall and precision, we have tp, fp and fn. We don't have TN
    # To calculate TN, from my understanding, we can take sum of confusion matrix then subtract it by (tp + fp + fn)
    # That's only difference, here rest will be same  in terms of how I generate confusion matrix, only difference is
    # we are returning the accuracy as an number not an list.

    accuracy_value = 0

    if pe_matrix is None:
        c_matrix = confusion_matrix(true_labels, classifier_output, n_classes)
        # print(c_matrix)
    else:
        c_matrix = pe_matrix
        # print(c_matrix)

    # # Similar to precision and recall function, we will loop over each class in n_classes
    # for i in range(n_classes):
    #     # going to calculate tp exact same way, no changes here
    #     tp = c_matrix[i][i]
    #     # print(tp)
    #
    #     # going to calculate both fn and fp here
    #
    #     fp = np.sum(c_matrix[:, i]) - tp
    #     # print(fp)
    #
    #     fn = np.sum(c_matrix[i, :]) - tp
    #     # print(fn)
    #
    #     # Accuracy TP + TN / TP + TN + FP + FN USE THIS -> from readme
    #
    #     # the only difference here is TN, I explained how we will find this above. sum of c_matrix minus (tp+fp+ fn)
    #
    #     tn = np.sum(c_matrix) - (tp+fp+fn)
    #     # print(tn)
    #
    #     # same as before, we are checking if denominator which is sum of the c_matrix a.k.a (tp+tn+fp+fn)
    #
    #     if np.sum(c_matrix) > 0:
    #         accuracy_value = tp + tn / np.sum(c_matrix)
    #     else:
    #         accuracy_value = 0
    #     print(accuracy_value)

    # *******EDIT: above doesn't work, updated version below******

    # NOTE: my above approach didn't work, kept getting error locally.
    # After looking into wikipedia for accuracy, found out that the above approach is for binary classification
    # We will need to do multiclass classification, it was also confirmed here:
    # wikipedia link: https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification
    # It shows in wikipedia, that accuracy = correct classifications / all classifications
    # To get the correct classifications, we need sum of TP and TN, or the sum along diagonals in the matrix
    # To get sum along diagonals we can use following: https://numpy.org/doc/stable/reference/generated/numpy.trace.html
    # Once we have that we can divide by total sum of matrix or in other words, all classifications.

    correct_classifications = np.trace(c_matrix)
    all_classifications = np.sum(c_matrix)
    # print(correct_classifications, all_classifications)

    # we now have both correct_classifications and all_classifications, can simply divide correct / all to get accuracy

    accuracy_value = correct_classifications / all_classifications
    # print(accuracy_value)

    # return the accuracy value
    return accuracy_value


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0, 1, 2, ...
    Returns:
        Floating point number representing the gini impurity.
    """

    # going to first initialize a value for gini impurity to a float
    gini_impurity_value = 0

    # To learn how gini_impurity works:
    # Reference: https://victorzhou.com/blog/gini-impurity/
    # From my understanding of what was provided,
    # Gini impurity is the probability of classifying the datapoint incorrectly
    # From wikipedia you provided in readme: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    # The final simplified equation of gini is 1 - sum of (probability of i squared).

    # so here is my approach:
    # first we randomly classify our data point according to the class distribution
    # then calculate the probabilities of each dataset at random
    # then we do the gini impurity using probabilities.

    # first step, to randomly classify our data point according to the class distribution
    # we have to see how many categories of elements there are in the class_vector input
    # for example from the reference given in the FAQ above, we have two types of colors in example 1.
    # To find out how many colors there are in the dataset, they had to find out what are the unique values in dataset
    # and how many times do they appear in the dataset to find the probability of that color.

    # essentially I will do the same here, I will find how many unique classes there are from vector of classes.
    # Then find the count of that class in the vector of classes given. So if an class named x appears 5 times, the count is 5.
    # For this I am using: https://numpy.org/doc/stable/reference/generated/numpy.unique.html
    # Note: this was used in my part 0, for the function: vectorized_flatten
    # unique, count = np.unique(positive_values, return_counts=True)
    # The unique represent the integer and count represents the number of occurrences
    # I will be using exact same format, unique will represent the unique classes in vector classes
    # and count will represent the number of times that class appears in vector classes. Both count and unique are arrays.
    # will replace positive_values with class_vector here

    unique, count = np.unique(class_vector, return_counts=True)
    # print(unique)
    # print(count)

    # second step, we can use the count variable, to find the probability.
    # using the example from above of class named x appears 5 times, the count being 5. if there are 10 classes in total
    # in the class_Vector input. the probability of class x appearing is 5/10 = 0.5. So to find probability we need to
    # divide the count variable by total sum of the count array.
    # EDIT: just making sure we don't divide by 0, missing marks part 3 on GS, making sure all base cases are good
    if count.sum() != 0:
        probability = count / count.sum()

    # print(count.sum())
    # print(probability)

    # third step, let's actually do the gini impurity now
    # From above, I said 1 - sum of (probability of i squared) is simplified formula, will use it here.
    # I will set that equation to variable gini_impurity_value
    # note in python (**) is squared.

    gini_impurity_value = 1 - np.sum(probability**2)
    # print(gini_impurity_value)


    # return the gini impurity value
    return gini_impurity_value


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0, 1, 2....
        current_classes (list(list(int): A list of lists where each list has
            0, 1, 2, ... values).
    Returns:
        Floating point number representing the gini gain.
    """

    # going to first initialize a value for gini gain to a float
    gini_gain_value = 0

    # I used the following resources for this function:
    # https://victorzhou.com/blog/information-gain/ -> this person had a follow up of gini_impurity for information gain
    # https://en.wikipedia.org/wiki/Information_gain_(decision_tree)
    # The Gini Gain follows a similar approach to information gain, replacing entropy with Gini Impurity.

    # so what they did for information gain is, calculate the entropy before split,
    # let's do same here with gini impurity, let's calculate gini impurity of previous_classes input
    # I am going to use the function I made above

    previous_gini_impurity = gini_impurity(previous_classes)
    # print(previous_gini_impurity)

    # After that they weighted the entropy of each branch by how many elements it has,
    # after they got entropies for both branches after the split.
    # I have to find weighted impurity for gini_gain instead
    # Let's first define an variable for weighted_gini_impurity

    weighted_gini_impurity = 0

    # Let me sum up what they did, they got weights of both sides of the split.
    # weight is determined by how many elements there are on each side of the split, if there are 4 greens and 6 blues
    # 4 on left and 6 on right, that means weights are 4 and 6 respectively with their entropy
    # then they multiply the 4 by it's entropy from left and 6 by it's entropy from right. That's the weighted entropy
    # in our case, let's replace entropy with impurity, we get weighted impurity
    # the current classes given in input, we have list of list(int)
    # let's loop through the each list of lists, which represent the group after split (left or right from above example)
    # then get impurity of that group, then get the weight by dividing the number of elements in that group
    # by the total number of elements in the previous class list, before the split. Example, 4/10 = 0.4.
    # then we multiply that weight by its impurity of subset.

    for group in current_classes:
        # let's get gini impurity of group variable first
        group_gini_impurity = gini_impurity(group)
        # print(group_gini_impurity)

        # EDIT: just making sure we don't divide by 0, making sure all base cases are good
        # the weights as mentioned above, by size of group variable by size of previous classes
        if len(previous_classes) != 0:
            weight_group = len(group) / len(previous_classes)

        # print(weight_group)

        # now we are multiplying the weight we got above by the impurity we got before that, then adding that result
        # to the weighted_gini_impurity for each list or group in the current_classes input, this is similar to adding
        # 0.4 * it's entropy + 0.6 * it's entropy from their example

        weighted_gini_impurity += group_gini_impurity * weight_group
        # print(weighted_gini_impurity)


    # after we have our weighted gini impurity, we need to get the gini gain value,
    # which is previous gini impurity minus weighted gini impurity

    gini_gain_value = previous_gini_impurity - weighted_gini_impurity
    # print(gini_gain_value)

    # return the gini gain value
    return gini_gain_value


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=22):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """

        # First the base cases, let's look at 1.1 and 1.2:
        # 1.1: If all input vectors have the same class, return a leaf node with the appropriate class label.
        # 1.2: If a specified depth limit is reached, return a leaf labeled with the most frequent class.
        # For 1.1, essentially if all elements in classes input vector is the same as first element, then we don't need
        # to split any further. so all we have to check is if all classes equals the first element of classes
        # we can use np.all for this: https://numpy.org/doc/stable/reference/generated/numpy.all.html
        # this tests whether all array elements along given axis is true

        if np.all(classes == classes[0]):
            # if the above statement is true then we can return the leaf node at classes[0]
            # They said that the leaf node is decision node, from above and from read me
            # 2. Use the DecisionNode class
            return DecisionNode(None, None, None, classes[0])

        # next base case, 1.2, is saying if the depth given in input
        # reaches the depth limit from the init then we can return the most frequent class.
        # from the text book, we have to call onto a PLURALITY-VALUE function,
        # as that's selects the most common output value among a set of examples, breaking ties randomly.
        # We can return the decision Node with the class label being plurality value function with classes as args

        if depth == self.depth_limit:
            return DecisionNode(None, None, None,  self.plurality_values(classes))

        # we have covered the main base cases, let's move to next part.
        # which is Initialize the class with useful variables and assignments
        # let's initialize the highest_gini_gain used later for highest gini gain achieved by an attribute split
        # let's initialize the "alpha_best" as mentioned in readme for the value with the highest information gain.
        # Let's initialize best_threshold that holds the threshold value for the best attribute split in the tree
        # finally, the best_subsets stores subsets of features and classes from the best attribute split
        # essentially we find the best attribute, alpha_best and it's threshold, best_threshold, that results in
        # highest gini gain, highest_gini_gain.

        highest_gini_gain = -1
        alpha_best = None
        best_threshold = None
        best_subsets = None

        # next from the pseudocode in page 660, in textbook figure 19.5
        # we need to do the else part of that pseudocode, which is:
        # A←argmaxa∈attributes IMPORTANCE(a, examples)
        # This part of pseudocode, from my understanding, evaluates the importance of each attribute
        # importance for us is gini gain.
        # to do what I said above, we need to iterate through each attribute, alpha, in features
        # features.shape[1] get's you the number of features in features array. As features are columbs in features array

        for alpha in range(features.shape[1]):
            # The next part of the pseudocode is: for each value v of A do
            # This part is asking to evaluate all possible threshold of alpha attribute. v represents threshold
            # and A represents alpha here
            # to get all possible thresholds, we need to get all unique values of the current attribute
            # and iterate through them, testing each potential threshold for making a split
            # to get all possible unique values, we can use numpy.unique, which was used before in this file
            # we are slicing inside the features list and getting all rows when the column is alpha.
            # this is how I made my decision tree by hand also, getting output of all rows that have certain column
            # once we have unique values then we can iterate over them, test each potential threshold for the split.

            values = np.unique(features[:, alpha])

            # so I decided to add an extra layer to selecting the splits, by randomly choose one of the values
            # I tried the median as threshold but it didn't generate the right response.
            # This new approach avoids timing out on gs and just long computational run time
            # as I am randomly choosing few thresholds to try from the unique values of the current attribute
            # I chose the size of the for this to be the minimum value of either 10 or length of values, if values is 9
            # then the size will be 9 of selected values. The replace=False, means sample is without replacement
            # This works locally now, getting more than 80% accuracy for part 3.

            selected_values = np.random.choice(values, size=min(10, len(values)), replace=False)

            # EDIT: before it was threshold in values, we change the iteration to selected_values now
            for threshold in selected_values:

                # the next part of the pseudocode is: exs← {e : e∈examples and e.A = v
                # the pseudocode is filtering the dataset, examples, to include only those examples where attribute A
                # has a particular value v
                # Here in code we will have left subset and right subset, how does the threshold work?
                # well from part 1, we know if the condition <= threshold, if that's true, then it goes to left subset
                # if not true it goes to right. Here my left subset will be called when condition is <= threshold
                # and right subset for when condition > threshold. What's condition?
                # well, Condition is going to be features[:alpha], like I had from above, we need to check all rows
                # when the column equals the index alpha in features array.

                left_subset = features[:, alpha] <= threshold
                right_subset = features[:, alpha] > threshold

                # print(left_subset)
                # print(right_subset)

                # once we have our splits, we have to go back to the base case, which is 1.3 and 1.4:
                # 1.3: Splits producing 0, 1 length vectors 1.4: Splits producing less or equivalent information
                # we can just check if the length of classes at index of subsets equals 0, it would satisfy those base cases
                # we are going to skip if that condition is true, avoid those

                if len(classes[left_subset]) == 0 or len(classes[right_subset]) == 0:
                    continue

                # let's calculate the gini gain for each possible split in both subsets
                # remember this ties back to the pseudcode: A←argmaxa∈attributes IMPORTANCE(a, examples)
                # instead of importance we are doing gini gain. To get each possible split, we need to group both
                # subsets into a list of subsets and then perform the gini gain on that using classes
                # remember gini gain function we implemented above, takes previous_classes and current_classes as args

                subsets = [classes[left_subset], classes[right_subset]]
                gini_gain_value = gini_gain(classes, subsets)
                # print(gini_gain_value)

                # now we go back to variables we initialized at top, here we are going to store:
                # highest_gini_gain, alpha_best, best_threshold, best_subset
                # we will only store values for this, if the current gini_gain_value is greater than highest_gini_gain

                if gini_gain_value > highest_gini_gain:
                    # highest gini gain will be set to what we found in the current split with the current subsets
                    # alpha_best is the attribute with highest gain, is set to be the current alpha
                    # best_threshold is set to be the current threshold that results in the higher gini gain value
                    # and best subsets are subsets that produce the best gini gain value for chosen threshold

                    highest_gini_gain = gini_gain_value
                    alpha_best = alpha
                    best_threshold = threshold
                    best_subsets = (features[left_subset], classes[left_subset]), (features[right_subset], classes[right_subset])
                    # print(highest_gini_gain, alpha_best, best_threshold, best_subsets)

        # Next we need to go back to the pseudo code: if attributes is empty then return PLURALITY-VALUE(examples)
        # how we translate here is that if no attribute, alpha provides a proper gain, then return the most common class
        # similarly to how they did in pseudo code, we will call the plurality values function pass in classes again
        # our condition will check not if attributes is empty since we aren't looping through anything but if the
        # highest_gini_gain is less than 0, because if this is true, that means in the loop above, we weren't able to
        # get a gini_gain_value for any of the attributes, thresholds or subsets that results in higher value than default

        if highest_gini_gain <= 0:
            return DecisionNode(None, None, None, self.plurality_values(classes))

        # we need to build the decision function for the decision node
        # from part 1, given in the readme: func = lambda feature : feature[2] <= 0.356
        # This will choose the left node if the A2 attribute is <= 0.356.
        # that's how they said in read me to create an decision function
        # let's follow the same format, instead of feature[2], we will do feature[alpha_best]
        # as we want to build the decision node with our best attribute and best threshold
        # instead of 0.356, I will use best threshold we set above

        # func = lambda feature : feature[2] <= 0.356
        func = lambda feature: feature[alpha_best] <= best_threshold
        # print(func)

        # next we need to create the left and right subtrees with recursive calls
        # IN the pseudcode: subtree←LEARN-DECISION-TREE(exs, attributes−A, examples
        # so let's follow that, lets make an recursive call to this function __build_tree__ everytime we create an subtree
        # remember best_subsets from above, we need to call those and pass inside in this function
        # _build_tree_ function takes in features, classes and depth
        # features will be the [0][0] index of best_subsets for left_subtree and [1][0] for right_subtree
        # classes will be the [0][1] index of best_subsets for left_subtree and [1][1] for right_subtree
        # depth will be incremented by 1, since we are doing an recursive call.

        left_subtree = self.__build_tree__(best_subsets[0][0], best_subsets[0][1], depth+1)
        right_subtree = self.__build_tree__(best_subsets[1][0], best_subsets[1][1], depth + 1)

        # we have our recursive statements, and decison function, let's return the decision node now with this information

        return DecisionNode(left_subtree, right_subtree, func)


    def plurality_values(self, classes):
        # The textbook pseduo code on page 660 said:
        # The function PLURALITY-VALUE selects the most common output value among a set of examples
        # We can do this by using collections.Counter, given in readme and imported by default
        # we can count how many times the input classes appears using the counter, then check for which of the counter
        # has the highest values, using the max function in python. If there are more than 1, then randomly choose 1.

        count = Counter(classes)
        max_count = max(count.values())

        # create an list that contains that stores the highest values
        highest_values = []

        # let's iterate over each item in count dictionary, from above
        # label represents the class label and counter represents the frequency of that class


        # count.items() returns an tuple for each class, as (class,count)
        for label, counter in count.items():
            # we are only trying to get the highest values
            if counter == max_count:
                # we then append the current class label associated with that counter variable
                highest_values.append(label)

        # we need to randomly choose one of the values in highest values now, as those are most frequent classes
        # in the classes input list. Let's return the leaf node using this value in classes part of args.
        # I use this: https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        # It essentially choose an random sample given an 1-d array
        # returning the leaf node with most frequent class

        return np.random.choice(highest_values)

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        class_labels = []

        # write a function to produce classifications for a list of features once your decision tree has been built.

        # essentially in this function I am going to iterate over the list of example features,
        # and classify single feature at a time recursively and add the values of that to the class_label given.
        # I used the hint for this part from the read, hint 8:
        # "Use recursion to build your tree, by using the split lists, remember true goes left using decide"
        # the recursion call will be in decide function in part 1 from decision node class, that's exactly what we want
        # that's where we determine recursively the class of an input array by testing a value against a
        # feature's attributes values based on the decision function.
        # It also returns a Class label if a leaf node, otherwise a child node.
        # I start from the root of the tree and the classify everything. Once the features list is iterated,
        # we return the list of class labels that has been classified by decide function.

        for feature in features:
            class_label = self.root.decide(feature)
            # print(class_label)
            class_labels.append(class_label)
            # print(class_labels)

        # print(class_labels)
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """
    folds = []

    # for this I used the tutorial you gave me in readme: https://machinelearningmastery.com/k-fold-cross-validation/
    # However, here is the general procedure from that website from readme:
    # 1) Shuffle the dataset randomly.
    # 2) Split the dataset into k groups
    # 3) For each unique group:
    # 3.1) Take the group as a hold out or test data set
    # 3.2) Take the remaining groups as a training data set
    # 3.3) Fit a model on the training set and evaluate it on the test set
    # 3.4) Retain the evaluation score and discard the model
    # 4) Summarize the skill of the model using the sample of model evaluation scores

    # let's do step 1, using: https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html
    # will be using np.random.shuffle with the data set given in the input

    # np.random.shuffle(dataset)

    # NOTE: I got following error locally when I just do np.random.shuffle with dataset input:
    #   line 866, in generate_k_folds -> np.random.shuffle(dataset)
    #   line 866, in generate_k_folds -> np.random.shuffle(dataset)
    #   File "mtrand.pyx", line 4570, in numpy.random.mtrand.RandomState.shuffle
    #   File "mtrand.pyx", line 4573, in numpy.random.mtrand.RandomState.shuffle
    # TypeError: 'tuple' object does not support item assignment

    # The reason for the error is due to the dataset is a tuple of features, classes from the description above
    # we can't just do np.random.shuffle(dataset) without combining the tuple of features and classes into one,
    # from the numpy docs for random shuffle, that's an requirement.
    # so first we will check if the dataset input is of tuple type, using the python built in function isinstance
    # then we assign the variables features and classes with the respective values from dataset
    # then we use numpy.column_stack similar to what I used in the vectorized_glue function from part 0
    # it stacks the 1-d arrays into a 2-d array

    if isinstance(dataset, tuple):
        features, classes = dataset
        dataset = np.column_stack((features, classes))

    # now we can do random.shuffle function, it didn't throw error in locally for me

    np.random.shuffle(dataset)

    # Then the next step says Split the dataset into k groups
    # to do this we can take the length of dataset and divide by k
    # from the local test cases, I saw that they did this: test_set_count = example_count // k
    # the reason for // k is the floor division operator in python, essentially takes floor of resultant (i.e. 5/2 is 2)
    # since the result of // in python is an integer. We don't get the reminder.
    # This would be the fold size, based of description of function itself, Split dataset into folds.
    # EDIT: just making sure we don't divide by 0, missing marks part 3 on GS, making sure all base cases are good
    if k != 0:
        fold_size = len(dataset) // k
    # print(fold_size)

    # Next we do the step 3.1: Take the group as a hold out or test data set
    # We need to iterate K number times to create each fold, as we split data into k equal subsets
    # remember, k represents the number of groups or folds into which we are splitting the dataset. Hence why we iterate
    # over k number of times in this loop
    # the iterator variable i represents the index of current fold
    # They said to test data set, to do that we need to calculate the starting index for the test set in current fold
    # to get that, we have to multiply the fold_size by i, as that would give the index in the dataset where current fold
    # begins from. For example: fold_size is 10, and index is 0 (that's first i value), the start index of the fold
    # will be 0. the second index is 1, fold_size remains 10, hence the start index of fold is now 10, for next one 11, etc.
    # We need to also get the end index of the test set in the current fold. to do that
    # we can just add start_index to the fold_size
    # once we have the start index of the fold and end index of the fold, we can take the test set for the current fold
    # that's what 3.1 is saying, we can create test_dataset by using slicing as I did in other parts of part 0,
    # essentially it get's data from the start_index_fold in the dataset to but not including end_index_fold.

    for i in range(k):
        start_index_fold = fold_size * i
        end_index_fold = start_index_fold + fold_size
        test_dataset = dataset[start_index_fold: end_index_fold]
        # print(test_dataset)

        # now we have test_dataset, which fulfils expectations for 3.1
        # we can move onto 3.2, which is to: Take the remaining groups as a training data set
        # to do this we can select all other indexes in dataset that isn't in test_dataset for the train_dataset
        # so what's left from dataset, that's not in test_dataset?
        # well we have start_index_fold, we can get data from the dataset before the start_index_fold
        # We can also get data after the end_index_fold, including that index as it's not included in test_dateset
        # dataset[:start_index_fold] takes all data before start_index_fold as end point of slicing isn't inclusive
        # dataset[end_index_fold:] takes all data starting end_index_fold as start point of slicing is inclusive

        train_start_index = dataset[:start_index_fold]
        train_end_index = dataset[end_index_fold:]

        # once we have the start and end index for the train dataset, we need to concatenate the data at both these indexes
        # to do that we can use: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html

        train_dataset = np.concatenate((train_start_index, train_end_index))
        # print(train_dataset)

        # that fulfils expectations for 3.2
        # let's move onto the 3.3, which is Fit a model on the training set and evaluate it on the test set
        # we just need to append to the folds list, remember Fold is a tuple (training_set, test_set).
        # and Set is a tuple (features, classes). Each fold is a tuple of sets and Each Set is a tuple of numpy arrays.
        # so the first tuple inside the fold represents the training_set and the second represents test_set
        # training_Set contains features then classes, same with test_set
        # So we need to append this to the folds list

        # let's define the train_dataset's features
        # for the features, we cannot include the last column as it's usually the class label in any dataset
        # to include all columns except last, we can use python slicing technique [:,:-1] the -1 avoids last column

        train_dataset_features = train_dataset[:, :-1]

        # get the classes from train dataset, this is just getting the last column from the training dataset
        # to do that, we can use slicing again, [:-1] again the -1 represents only return that value here.

        train_dataset_classes = train_dataset[:, -1]

        # do the same for test dataset for both classes and features

        test_dataset_features = test_dataset[:, :-1]

        test_dataset_classes = test_dataset[:, -1]

        # let's now create an training_set and test_set like this tuple (features, classes)

        training_dataset_tuple = (train_dataset_features, train_dataset_classes)
        # print(training_dataset_tuple)

        test_dataset_tuple = (test_dataset_features, test_dataset_classes)
        # print(test_dataset_tuple)

        # let's now create the tuple for fold like this tuple (training_set, test_set)

        fold_tuple = (training_dataset_tuple, test_dataset_tuple)
        # print(fold_tuple)

        # we can finally append the fold tuple into the list of folds now

        folds.append(fold_tuple)

    # return the folds list
    # print(folds)
    return folds


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees=80, depth_limit=5, example_subsample_rate=.1,
                 attr_subsample_rate=.3):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        # note changed the values in init, to match what readme says:
        # To test, we will be using a forest with 80 trees, with a depth limit of 5,
        # example subsample rate of 0.3 and attribute subsample rate of 0.3

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # They are usually short (depth limited)
        # They use smaller (but more of them) random datasets for training
        # They use a subset of attributes drawn randomly from the training set
        # They fit the tree to the sampled dataset and are considered specialized to the set

        # I am going to generate_k_folds using features and classes given, with k=10, as said in textbook and lectures
        # This will get us the training dataset from that function which we created there
        # We can then pull features and classes from that training dataset, then can fit the tree to sampled dataset
        # this follows all the steps given in readme

        # my k_folds will have k of 10 as they said in lectures and textbook to keep it usually around 10
        # also it says they are usually short (Depth limited), so our K has to be small, don't have to create too many
        # subsets

        k_folds = generate_k_folds((features, classes), 10)

        # once k_folds is done, we can iterate through it to get the training dataset,
        # remember, fold_tuple = (training_dataset_tuple, test_dataset_tuple). So we just need first variable from
        # the k_folds, second element in the tuple is test_dataset, which we don't need here

        for train_dataset, _ in k_folds:
            # then we can pull the training dataset features and classes at each iteration
            train_dataset_features, train_dataset_classes = train_dataset

            # we have to create an decision tree as that's what the function is asking for:
            # "Build a random forest of decision trees using Bootstrap Aggregation"
            tree = DecisionTree(depth_limit=self.depth_limit)

            # once we have the tree we can fit the tree with the training dataset features and classes
            tree.fit(train_dataset_features, train_dataset_classes)

            # just append the result of the fitting to the tree in init now. No return required in this function.
            self.trees.append(tree)

            # print(tree)
            # print(self.trees)


    def classify(self, features):
            """Classify a list of features based on the trained random forest.
            Args:
                features (m x n): m examples with n features.
            Returns:
                votes (list(int)): m votes for each element
            """
            votes = []

            # Advanced learners may use weighting of their classifier trees to improve performance
            # They use majority voting (every tree in the forest votes) to classify a sample

            # we have to iterate over each feature in features, as votes list returns the list of votes for each feature

            for feature in features:

                # weightings of the classifier trees is needed to be found here and need to then take max votes
                # using those weightings, by seeing the most common weighting.

                # let's first create a list of weightings for all tress for the current feature in the input features
                weightings = []

                # let's iterate over each decision tree in random forest class
                # as they said "every tree in forest votes" in readme
                for tree in self.trees:
                    # then we classify the feature of the current tree, the [0] will give a single weighting at a time
                    # for the given feature which we passed as input inside the classify function
                    tree_weighting = tree.classify([feature])[0]
                    # print(tree_weighting)

                    # we can then append the tree_weighting to the weightings list we created above
                    weightings.append(tree_weighting)

                # this is where we use majority voting to classify a sample
                # We find the max number of votes by getting the weighting that appears the most.
                vote = max(set(weightings), key=weightings.count)
                # print(vote)

                # we can then append that vote of the current feature in features list to the votes list
                votes.append(vote)

            #print(votes)
            #return the votes list
            return votes


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self, n_clf=0, depth_limit=0, example_subsample_rt=0.0, \
                 attr_subsample_rt=0.0, max_boost_cycles=0):
        """Create a boosting class which uses decision trees.
        Initialize and/or add whatever parameters you may need here.
        Args:
             num_clf (int): fixed number of classifiers.
             depth_limit (int): max depth limit of tree.
             attr_subsample_rate (float): percentage of attribute samples.
             example_subsample_rate (float): percentage of example samples.
        """
        self.num_clf = n_clf
        self.depth_limit = depth_limit
        self.example_subsample_rt = example_subsample_rt
        self.attr_subsample_rt=attr_subsample_rt
        self.max_boost_cycles = max_boost_cycles
        # TODO: finish this.͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
        raise NotImplemented()

    def fit(self, features, classes):
        """Build the boosting functions classifiers.
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        # TODO: finish this.͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
        raise NotImplemented()

    def classify(self, features):
        """Classify a list of features.
        Predict the labels for each feature in features to its corresponding class
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """
        # TODO: finish this.͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
        raise NotImplemented()


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """

        # just following what instructions say above
        # multiplies by itself and then adds to itself

        return data * data + data

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return (max_sum, max_sum_index)

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        # search throught first 100 rows, look for max sum

        # args says to slice and sum data (add all values in the row together)
        # used this to learn slicing: https://www.w3schools.com/python/python_strings_slicing.asp
        # used this to learn add all values in the row together: https://numpy.org/doc/stable/reference/generated/numpy.sum.html

        max_row_sum = data[:100].sum(axis=1)

        # used this to find index of row with max sum: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html

        max_row_sum_index = np.argmax(max_row_sum)

        # return row with max sum value and it's index.

        return max_row_sum[max_row_sum_index], max_row_sum_index

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """
        unique_dict = {}
        flattened = data.flatten()
        for item in flattened:
            if item > 0:
                if item in unique_dict:
                    unique_dict[item] += 1
                else:
                    unique_dict[item] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """

        # flatten down data into a 1d array
        # used following for that: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html

        flattened = data.flatten()

        # create a dictionary of how often a positive number appears in data

        # see below my approach with dictionary was wrong initially, so this is useless
        # dictionary = {}

        # used following positive flattened indices: https://numpy.org/doc/stable/reference/generated/numpy.where.html

        positive_number = np.where(flattened > 0)

        # we need to get the values where positive flattened numbers

        positive_values = flattened[positive_number]

        # need to get the values of the positive values that give unique values
        # used the following for that: https://numpy.org/doc/stable/reference/generated/numpy.unique.html

        unique, count = np.unique(positive_values, return_counts=True)

        # print(unique, count)
        # print(unique)
        # print(count)

        # The unique represent the integer and count represents the number of occurrences
        # we need to store this in our dictionary

        # to do that we can use zip function from python, this groups both tuples respectively,
        # while initializing to dictionary, we have to set zip object to an dict
        # got zip from the following: https://www.w3schools.com/python/ref_func_zip.asp

        # dictionary = dict(zip(unique, count))
        # this didn't work
        # print(dictionary)

        # this worked, the dictionary didn't:
        # print(list(zip(unique, count)))

        return list(zip(unique, count))



    def non_vectorized_glue(self, data, vector, dimension='c'):
        """Element wise array arithmetic with loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        if dimension == 'c' and len(vector) == data.shape[0]:
            non_vectorized = np.ones((data.shape[0],data.shape[1]+1), dtype=float)
            non_vectorized[:, -1] *= vector
        elif dimension == 'r' and len(vector) == data.shape[1]:
            non_vectorized = np.ones((data.shape[0]+1,data.shape[1]), dtype=float)
            non_vectorized[-1, :] *= vector
        else:
            raise ValueError('This parameter must be either c for column or r for row')
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row, col] = data[row, col]
        return non_vectorized

    def vectorized_glue(self, data, vector, dimension='c'):
        """Array arithmetic without loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        vectorized = None

        # takes an multi-dimensional array and vector then combines both into a new one.
        # need to handle both column and row-wise additions

        # I found the following functions from numpy to the additions for me:
        # column wise: https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html
        # row wise: https://numpy.org/doc/stable/reference/generated/numpy.row_stack.html

        # we just check if dimension is c for column or r for row, as given in args

        if dimension == 'c':
            vectorized = np.column_stack((data, vector))
        elif dimension == 'r':
            vectorized = np.row_stack((data, vector))
        else:
            raise ValueError("Dimension must be either c for column or r for row")

        # return the vectorized array

        return vectorized

    def non_vectorized_mask(self, data, threshold):
        """Element wise array evaluation with loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared.
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        non_vectorized = np.zeros_like(data, dtype=float)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                val = data[row, col]
                if val >= threshold:
                    non_vectorized[row, col] = val
                    continue
                non_vectorized[row, col] = val**2

        return non_vectorized

    def vectorized_mask(self, data, threshold):
        """Array evaluation without loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared. You are required to use a binary mask for this problem
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        vectorized = None

        # function takes multi-dimensional array then populates a new multi-dimnesionsal array.
        # if values in data is below threshold it will be squared.
        # required to use binary mask for this,
        # essentially it's if data < threshold, the mask is true and false otherwise.
        # the mask contains true or false values.

        mask = data < threshold
        # print(mask)

        # let's take our existing array and populate new array
        # used this for that: https://numpy.org/doc/stable/reference/generated/numpy.copy.html

        vectorized = np.copy(data)

        # we are taking the vectorized array which is a copy of data,
        # then we are squaring it, using: https://numpy.org/doc/stable/reference/generated/numpy.square.html
        # we are squaring the value in vectorized at mask index, which is when
        # data < threshold, so everywhere that's true, we will get the vectorized values
        # then we will square it

        vectorized[mask] = np.square(vectorized[mask])

        # return the vectorized array

        return vectorized
