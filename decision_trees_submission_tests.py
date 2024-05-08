import math
import unittest
import decision_tree as dt
import numpy as np
import time


class DecisionTreePart1Tests(unittest.TestCase):
    """Test tree example, confusion matrix, precision, recall, and accuracy.

    Attributes:
        hand_tree (DecisionTreeNode): root node of the built example tree.
        ht_examples (list(list(int)): features for example tree.
        ht_classes (list(int)): classes for example tree."""

    def setUp(self):
        """Setup test data.
        """
        self.hand_tree = dt.build_decision_tree()
        self.ht_examples = [[ 1.1125,-0.0274,-0.0234, 1.3081],
                            [ 0.0852, 1.2190,-0.7848,-0.7603],
                            [-1.1357, 0.5843,-0.3195, 0.8563],
                            [ 0.9767, 0.8422, 0.2276, 0.1197],
                            [ 0.8904,-1.7606, 0.3619,-0.8276],
                            [ 2.3822,-0.3122,-2.0307,-0.5065],
                            [ 0.7194,-0.4061,-0.7045,-0.0731],
                            [-2.9350, 0.7810,-2.5421, 3.0142],
                            [ 2.4343,-1.5380,-2.7953, 0.3862],
                            [ 0.8096,-0.2601, 0.5556, 0.6288],
                            [ 0.8577,-0.2217,-0.6973,-0.1095],
                            [ 0.0568, 0.0696, 1.1153,-1.1753]]
        self.ht_classes = [1, 2, 0, 1, 0, 2, 2, 0, 2, 1, 1, 0]

    def test_hand_tree_accuracy(self):
        """Test accuracy of the tree example.

        Asserts:
            decide return matches true class for all classes.
        """
        for index in range(0, len(self.ht_examples)):
            # print(self.ht_examples[index])͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
            decision = self.hand_tree.decide(self.ht_examples[index])
            # print(decision)

            assert decision == self.ht_classes[index]


    def test_confusion_matrix(self):
        """Test confusion matrix for the example tree.

        Asserts:
            confusion matrix is correct.
        """
        true_label = [1, 1, 1, 0, 0, 0, 0]
        answer = [1, 0, 0, 1, 0, 0, 0]
        test_matrix = [[3, 1], [2, 1]]
        n_classes = 2
        assert np.array_equal(test_matrix, dt.confusion_matrix(true_label, answer, n_classes))

    def test_confusion_matrix_multiclass(self):
        """Test confusion matrix for multiclass input.
        Asserts:
            confusion matrix is correct.
        """

        answer = [1, 0, 0, 1, 0, 0, 0, 2, 3, 3, 3, 2, 0, 1]
        true_label = [1, 1, 1, 0, 0, 0, 0, 2, 3, 1, 3, 2, 2, 2]
        test_matrix = [[3, 1, 0, 0], [2, 1, 0, 1], [1, 1, 2, 0], [0, 0, 0, 2]]
        n_classes = 4
        assert np.array_equal(test_matrix, dt.confusion_matrix(true_label, answer, n_classes))

    def test_precision_calculation(self):
        """Test precision calculation.

        Asserts:
            Precision matches for all true labels.
        """
        
        true_label =    [ 3, 0, 4, 4, 1, 0, 1, 0, 4, 2, 3, 2]
        output_answer = [ 4, 2, 0, 4, 3, 1, 4, 3, 1, 3, 3, 3]
        test_list = [0.000, 0.000, 0.000, 0.200, 0.333]
        calculated_list = dt.precision(true_label, output_answer, n_classes = 5)
        assert np.array_equal(test_list, [round(elem, 3) for elem in calculated_list])

    def test_precision_calculation_multiclass(self):
        n_classses = 4
        answer = [1, 0, 0, 1, 0, 0, 0, 2, 3, 3, 3, 2, 0, 1]
        true_label = [1, 1, 1, 0, 0, 0, 0, 2, 3, 1, 3, 2, 2, 2]
        precision_value = dt.precision(true_label, answer, n_classses)
        precision_value = [round(num, 3) for num in precision_value]
        assert precision_value == [0.5 , 0.333 , 1.0, 0.667]

    def test_recall_calculation(self):
        """Test recall calculation.

        Asserts:
            Recall matches for all true labels. Checks for passing a matrix
        """
        true_label =    [ 3, 0, 4, 4, 1, 0, 1, 0, 4, 2, 3, 2]
        output_answer = [ 4, 2, 0, 4, 3, 1, 4, 3, 1, 3, 3, 3]

        test_matrix = np.array([[0, 1, 1, 1, 0],
                                [0, 0, 0, 1, 1],
                                [0, 0, 0, 2, 0],
                                [0, 0, 0, 1, 1],
                                [1, 1, 0, 0, 1]])

        n_classes = 5
        test_list = [0.000, 0.000, 0.000, 0.500, 0.333]
        calculated_list = dt.recall(true_label, output_answer, n_classes, test_matrix)
        assert np.array_equal(test_list, [round(elem, 3) for elem in calculated_list])

    def test_recall_multiclass(self):
        answer = [1, 0, 0, 1, 0, 0, 0, 2, 3, 3, 3, 2, 0, 1]
        true_label = [1, 1, 1, 0, 0, 0, 0, 2, 3, 1, 3, 2, 2, 2]
        n_classes = 4
        recall_value = dt.recall(true_label, answer, n_classes)
        recall_value = [round(num, 3) for num in recall_value]
        assert recall_value == [0.75, 0.25, 0.5 , 1.]

    def test_accuracy_calculation(self):
        """Test accuracy calculation.

        Asserts:
            Accuracy matches for all true labels.
        """
        answer = [0, 0, 0, 0, 0]
        true_label = [1, 1, 1, 1, 1]
        true_class_cnt = 2
        total_count = len(answer)
        accuracy = [0.] * len(answer)
        for index in range(0, len(answer)):
            answer[index] = 1
            accuracy[index] = dt.accuracy(true_label, answer, true_class_cnt)
            assert round(accuracy[index], 3) == round(((index + 1) / total_count), 3)

    def test_accuracy_multiclass(self):
        answer = [3, 5, 3, 5, 2, 0, 4, 2, 1, 1, 5, 6, 3, 3, 4, 6, 5, 4, 5, 5]
        true_label = [6, 1, 3, 6, 0, 4, 4, 5, 1, 0, 1, 6, 6, 6, 1, 5, 5, 6, 3, 6]
        n_classes = 7
        assert round(0.25, 3) == round(dt.accuracy(true_label, answer, n_classes), 3)

class DecisionTreePart2Tests(unittest.TestCase):
    """Tests for Decision Tree Learning.

    Attributes:
        restaurant (dict): represents restaurant data set.
        dataset (data): training data used in testing.
        train_features: training features from dataset.
        train_classes: training classes from dataset.
    """

    def setUp(self):
        """Set up test data.
        """
        data_dir = './data/'
        self.restaurant = {'restaurants': [0] * 6 + [1] * 6,
                           'split_patrons': [[0, 0],
                                             [1, 1, 1, 1],
                                             [1, 1, 0, 0, 0, 0]],
                           'split_food_type': [[0, 1],
                                               [0, 1],
                                               [0, 0, 1, 1],
                                               [0, 0, 1, 1]]}

        self.dataset = dt.load_csv(data_dir + 'mod_complex_multi.csv')
        self.train_features, self.train_classes = self.dataset

    def test_gini_impurity_max(self):
        """Test maximum gini impurity.

        Asserts:
            gini impurity is 0.5.
        """

        gini_impurity = dt.gini_impurity([1, 1, 1, 0, 0, 0])

        assert .500 == round(gini_impurity, 3)

    def test_gini_impurity_min(self):
        """Test minimum gini impurity.

        Asserts:
            entropy is 0.
        """

        gini_impurity = dt.gini_impurity([1, 1, 1, 1, 1, 1])
        assert 0 == round(gini_impurity, 3)
        gini_impurity = dt.gini_impurity([1, 1, 0, 0, 0, 0, 2, 2, 2])
        assert round(0.642, 3) == round(gini_impurity, 3)


    def test_gini_impurity(self):
        """Test gini impurity.

        Asserts:
            gini impurity is matched as expected.
        """

        gini_impurity = dt.gini_impurity([1, 1, 0, 0, 0, 0])
        assert round(4. / 9., 3) == round(gini_impurity, 3)
        
        labels = [0, 1, 2, 1, 0, 2, 2, 2]
        assert round(dt.gini_impurity(labels), 3) == 0.625

        gini_impurity = dt.gini_impurity([1, 1, 1, 2, 2, 2, 0, 0, 0])
        assert  .667 == round(gini_impurity, 3)

    def test_gini_gain_max(self):
        """Test maximum gini gain.

        Asserts:
            gini gain is 0.5.
        """

        gini_gain = dt.gini_gain([1, 1, 1, 0, 0, 0],
                                 [[1, 1, 1], [0, 0, 0]])
        assert .500 == round(gini_gain, 3)

        gini_gain = dt.gini_gain([1, 1, 1, 0, 0, 0, 2, 2, 2],
                                 [[1, 1, 0, 2, 2, 2], [1, 0, 0]])
        assert 0.111 == round(gini_gain, 3)

    def test_gini_gain(self):
        """Test gini gain.

        Asserts:
            gini gain is within acceptable bounds
        """

        labels = ([1, 1, 1, 0, 0, 0], [[1, 1, 0], [1, 0, 0]])
        gini_gain = dt.gini_gain(labels[0],labels[1])
        assert 0.056 == round(gini_gain, 3)
        
        labels = ([0, 0, 1, 1, 2, 2, 2, 2], [[0, 0, 1, 2], [1, 2, 2, 2]])
        assert round(dt.gini_gain(labels[0],labels[1]), 3) == 0.125

    def test_gini_gain_restaurant_patrons(self):
        """Test gini gain using restaurant patrons.

        Asserts:
            gini gain rounded to 3 decimal places matches as expected.
        """

        gain_patrons = dt.gini_gain(
            self.restaurant['restaurants'],
            self.restaurant['split_patrons'])
        assert round(gain_patrons, 3) == 0.278

    def test_gini_gain_restaurant_type(self):
        """Test gini gain using restaurant food type.

        Asserts:
            gini gain is 0.
        """

        gain_type = round(dt.gini_gain(
            self.restaurant['restaurants'],
            self.restaurant['split_food_type']), 2)
        assert gain_type == 0.00

    def test_decision_tree_all_data(self):
        """Test decision tree classifies all data correctly.

        Asserts:
            classification is 100% correct.
        """

        tree = dt.DecisionTree()
        tree.fit(self.train_features, self.train_classes)
        output = tree.classify(self.train_features)
        assert (output == self.train_classes).all()

    def test_k_folds_test_set_count(self):
        """Test k folds returns the correct test set size.

        Asserts:
            test set size matches as expected.
        """

        example_count = len(self.train_features)
        k = 10
        test_set_count = example_count // k
        ten_folds = dt.generate_k_folds(self.dataset, k)

        for fold in ten_folds:
            training_set, test_set = fold
            assert len(test_set[0]) == test_set_count

    def test_k_folds_training_set_count(self):
        """Test k folds returns the correct training set size.

        Asserts:
            training set size matches as expected.
        """

        example_count = len(self.train_features)
        k = 10
        training_set_count = example_count - (example_count // k)
        ten_folds = dt.generate_k_folds(self.dataset, k)

        for fold in ten_folds:
            training_set, test_set = fold
            assert len(training_set[0]) == training_set_count



class DecisionTreePart3Tests(unittest.TestCase):
    """Tests for RandomForest Decision Tree Learning.

    Attributes:
        restaurant (dict): represents restaurant data set.
        dataset (data): training data used in testing.
        train_features: training features from dataset.
        train_classes: training classes from dataset.
    """

    def setUp(self):
        """Set up test data.
        """
        data_dir = './data/'
        self.comp_bin_dataset = dt.load_csv(data_dir + 'mod_complex_binary.csv')
        self.comp_multi_dataset = dt.load_csv(data_dir + 'mod_complex_multi.csv')
        self.bin_tests = dt.generate_k_folds(self.comp_bin_dataset, 10)
        self.multi_tests = dt.generate_k_folds(self.comp_multi_dataset, 10)
        self.rfb = None
        self.rfm = None

    def test_binary_random_forest(self):
        """Test random forest on binary data.

        Asserts:
            Accuracy is greater than 75%.
        """
        results = []
        for train, test in self.bin_tests:
            train_features, train_classes = train
            test_features, test_classes = test
            self.rfb = dt.RandomForest(80, 5, 0.3, 0.3)
            self.rfb.fit(train_features, train_classes)
            votes = self.rfb.classify(test_features)
            results.append(float(sum([1 if vote == test_classes[i] else 0 for i,
                                     vote in enumerate(votes)])) / float(len(test_classes)))
        print(sum(results) / 10)
        assert sum(results) / 10. > .75

    def test_multi_random_forest(self):
        """Test random forest on binary data.

        Asserts:
            Accuracy is greater than 80%.
        """
        results = []
        for train, test in self.multi_tests:
            train_features, train_classes = train
            test_features, test_classes = test
            self.rfm = dt.RandomForest(80, 5, 0.3, 0.3)
            self.rfm.fit(train_features, train_classes)
            votes = self.rfm.classify(test_features)
            results.append(float(sum([1 if vote == test_classes[i] else 0 for i,
                                 vote in enumerate(votes)])) / float(len(test_classes)))
            print(sum(results) / 10)
        assert sum(results) / 10. >= .80


class DecisionTreePart4Tests(unittest.TestCase):
    """Tests for Boost Decision Tree Learning.

    Attributes:
        restaurant (dict): represents restaurant data set.
        dataset (data): training data used in testing.
        train_features: training features from dataset.
        train_classes: training classes from dataset.
    """

    def setUp(self):
        """Set up test data.
        """
        data_dir = './data/'
        self.comp_bin_dataset = dt.load_csv(data_dir + 'complex_binary.csv')
        self.comp_multi_dataset = dt.load_csv(data_dir + 'complex_multi.csv')
        self.bin_tests = dt.generate_k_folds(self.comp_bin_dataset, 10)
        self.multi_tests = dt.generate_k_folds(self.comp_multi_dataset, 10)
        self.rfbb = None
        self.rfmb = None

    def test_binary_boosting(self):
        """Test random forest on binary data.

        Asserts:
            Accuracy is greater than 80%.
        """
        results = []
        for train, test in self.bin_tests:
            train_features, train_classes = train
            test_features, test_classes = test
            self.rfbb = dt.ChallengeClassifier()
            self.rfbb.fit(train_features, train_classes)
            votes = self.rfbb.classify(test_features)
            results.append(float(sum([1 if vote == test_classes[i] else 0 for i,
                                     vote in enumerate(votes)])) / float(len(test_classes)))
        assert sum(results) / 10. >= .80

    def test_multi_boosting(self):
        """Test random forest on binary data.

        Asserts:
            The more accuracy the better
        """
        results = []
        for train, test in self.multi_tests:
            train_features, train_classes = train
            test_features, test_classes = test
            self.rfmb = dt.ChallengeClassifier()
            self.rfmb.fit(train_features, train_classes)
            votes = self.rfmb.classify(test_features)
            results.append(float(sum([1 if vote == test_classes[i] else 0 for i,
                                     vote in enumerate(votes)])) / float(len(test_classes)))
        assert sum(results) / 10. >= .80


class VectorizationWarmUpTests(unittest.TestCase):
    """Tests the Warm Up exercises for Vectorization.

    Attributes:
        vector (Vectorization): provides vectorization test functions.
        data: vectorize test data.
    """

    def setUp(self):
        """Set up test data.
        """
        data_dir = './data/'
        self.vector = dt.Vectorization()
        self.data = dt.load_csv(data_dir + 'vectorize.csv', 1)

    def test_vectorized_loops(self):
        """Test if vectorized arithmetic.

        Asserts:
            vectorized arithmetic matches looped version.
        """

        real_answer = self.vector.non_vectorized_loops(self.data)
        my_answer = self.vector.vectorized_loops(self.data)

        assert np.array_equal(real_answer, my_answer)

    def test_vectorized_slice(self):
        """Test if vectorized slicing.

        Asserts:
            vectorized slicing matches looped version.
        """

        real_sum, real_sum_index = self.vector.non_vectorized_slice(self.data)
        my_sum, my_sum_index = self.vector.vectorized_slice(self.data)

        assert real_sum == my_sum
        assert real_sum_index == my_sum_index

    def test_vectorized_flatten(self):
        """Test if vectorized flattening.

        Asserts:
            vectorized flattening matches looped version.
        """

        answer_unique = sorted(self.vector.non_vectorized_flatten(self.data))
        my_unique = sorted(self.vector.vectorized_flatten(self.data))

        assert np.array_equal(answer_unique, my_unique)

    def test_vectorized_glue(self):
        """Test if vectorized flattening.

        Asserts:
            vectorized array matches looped version.
        """
        answer_glue = self.vector.non_vectorized_glue(self.data[:,0:-1], self.data[:,-1], 'c')
        my_glue = self.vector.vectorized_glue(self.data[:,0:-1], self.data[:,-1], 'c')

        assert np.array_equal(answer_glue, my_glue)

        answer_glue = self.vector.non_vectorized_glue(self.data[0:-1,:], self.data[-1,:], 'r')
        my_glue = self.vector.vectorized_glue(self.data[0:-1,:], self.data[-1,:], 'r')

        assert np.array_equal(answer_glue, my_glue)

    def test_vectorized_mask(self):
        """Test if vectorized flattening.

        Asserts:
            vectorized mask matches looped version.
        """
        val = 99.
        answer_mask = self.vector.non_vectorized_mask(self.data, val)
        my_mask = self.vector.vectorized_mask(self.data, val)

        assert np.array_equal(answer_mask, my_mask)

    def test_vectorized_loops_time(self):
        """Test if vectorized arithmetic speed.

        Asserts:
            vectorized arithmetic is faster than expected gradescope time.
        """

        start_time = time.time() * 1000
        self.vector.vectorized_loops(self.data)
        end_time = time.time() * 1000

        assert (end_time - start_time) <= 0.09

    def test_vectorized_slice_time(self):
        """Test if vectorized slicing speed.

        Asserts:
            vectorized slicing is faster than expected gradescope time.
        """

        start_time = time.time() * 1000
        self.vector.vectorized_slice(self.data)
        end_time = time.time() * 1000

        assert (end_time - start_time) <= 0.1

    def test_vectorized_flatten_time(self):
        """Test if vectorized flatten speed.

        Asserts:
            vectorized flatten is faster than expected gradescope time.
        """
        start_time = time.time() * 1000
        self.vector.vectorized_flatten(self.data)
        end_time = time.time()  * 1000

        assert (end_time - start_time) <= 5.0

    def test_vectorized_glue_time(self):
        """Test vectorized glue speed.

        Asserts:
            vectorized flatten is faster than expected gradescope time.
        """

        start_time = time.time() * 1000
        self.vector.vectorized_glue(self.data[0:-1,:], self.data[-1,:], 'r')
        end_time = time.time()  * 1000

        assert (end_time - start_time) <= 3.0

    def test_vectorized_mask_time(self):
        """Test if vectorized flatten speed.

        Asserts:
            vectorized flatten is faster than expected gradescope time.
        """
        val = 100.
        start_time = time.time() * 1000
        self.vector.vectorized_mask(self.data, val)
        end_time = time.time()  * 1000

        assert (end_time - start_time) <= 4.0


class NameTests(unittest.TestCase):
    def setUp(self):
        """Set up test data.
        """
        self.to_compare = "George P. Burdell"

    def test_name(self):
        """Test if vectorized arithmetic.

        Asserts:
            Non Matching Name
        """

        self.name = dt.return_your_name()
        assert self.name != None
        assert self.name != self.to_compare

if __name__ == '__main__':
    unittest.main()
