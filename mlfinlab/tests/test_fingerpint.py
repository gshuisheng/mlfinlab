"""
Test RegressionModelFingerprint and ClassificationModelFingerprint implementations.
"""

import unittest
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from mlfinlab.feature_importance import RegressionModelFingerprint, ClassificationModelFingerprint


# pylint: disable=invalid-name
# pylint: disable=unsubscriptable-object

class TestModelFingerprint(unittest.TestCase):
    """
    Test model fingerprint functions
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """

        # Use California housing dataset instead of removed Boston dataset
        housing = fetch_california_housing()
        self.X = pd.DataFrame(housing.data[:100])
        self.y = pd.Series(housing.target[:100])

        self.reg_rf = RandomForestRegressor(n_estimators=10, random_state=42)
        self.reg_linear = LinearRegression(fit_intercept=True)
        self.reg_rf.fit(self.X, self.y)
        self.reg_linear.fit(self.X, self.y)

        self.reg_fingerprint = RegressionModelFingerprint()

    def test_linear_effect(self):
        """
        Test get_linear_effect for various regression models and num_values.
        """

        self.reg_fingerprint.fit(self.reg_rf, self.X, num_values=20)
        linear_effect, _, _ = self.reg_fingerprint.get_effects()

        # Test that linear effects are computed and normalized values sum to 1
        self.assertTrue(len(linear_effect['norm']) == self.X.shape[1])
        self.assertAlmostEqual(sum(linear_effect['norm'].values()), 1.0, delta=1e-6)

        self.reg_fingerprint.fit(self.reg_linear, self.X, num_values=20)
        linear_effect, _, _ = self.reg_fingerprint.get_effects()

        # Test that linear effects are computed for linear model
        self.assertTrue(len(linear_effect['norm']) == self.X.shape[1])
        self.assertAlmostEqual(sum(linear_effect['norm'].values()), 1.0, delta=1e-6)

        # Test fingerprints with bigger num_values
        self.reg_fingerprint.fit(self.reg_linear, self.X, num_values=70)
        linear_effect_70, _, _ = self.reg_fingerprint.get_effects()

        # Increasing the number of samples doesn't change feature effect massively
        for feature in range(min(5, self.X.shape[1])):
            self.assertAlmostEqual(linear_effect['norm'][feature],
                                   linear_effect_70['norm'][feature], delta=0.1)

    def test_non_linear_effect(self):
        """
        Test get_non_linear_effect for various regression models and num_values.
        """

        self.reg_fingerprint.fit(self.reg_rf, self.X, num_values=20)
        _, non_linear_effect, _ = self.reg_fingerprint.get_effects()

        # Test that non-linear effects are computed
        self.assertTrue(len(non_linear_effect['norm']) == self.X.shape[1])

        self.reg_fingerprint.fit(self.reg_linear, self.X, num_values=20)
        _, non_linear_effect, _ = self.reg_fingerprint.get_effects()

        # Non-linear effect to linear model is zero
        for effect_value in non_linear_effect['raw'].values():
            self.assertAlmostEqual(effect_value, 0, delta=1e-8)

        self.reg_fingerprint.fit(self.reg_linear, self.X, num_values=70)
        _, non_linear_effect_70, _ = self.reg_fingerprint.get_effects()

        # Increasing the number of samples doesn't change feature effect massively for linear model
        for feature in range(min(5, self.X.shape[1])):
            self.assertAlmostEqual(non_linear_effect['raw'][feature],
                                   non_linear_effect_70['raw'][feature], delta=0.05)

    def test_pairwise_effect(self):
        """
        Test compute_pairwise_effect for various regression models and num_values.
        """

        combinations = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
        self.reg_fingerprint.fit(self.reg_rf, self.X, num_values=20, pairwise_combinations=combinations)
        _, _, pair_wise_effect = self.reg_fingerprint.get_effects()

        # Test that pairwise effects are computed for all combinations
        self.assertTrue(len(pair_wise_effect['raw']) == len(combinations))

        combinations = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
        self.reg_fingerprint.fit(self.reg_linear, self.X, num_values=20, pairwise_combinations=combinations)
        _, _, pair_wise_effect = self.reg_fingerprint.get_effects()

        # Pairwise effect for linear model should be zero
        for pair in combinations:
            self.assertAlmostEqual(pair_wise_effect['raw'][str(pair)], 0, delta=1e-9)

    def test_classification_fingerpint(self):
        """
        Test model fingerprint values (linear, non-linear, pairwise) for classification model.
        """

        X, y = load_breast_cancer(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        clf_fingerpint = ClassificationModelFingerprint()
        clf_fingerpint.fit(clf, X, num_values=20, pairwise_combinations=[(0, 1), (2, 3), (8, 9)])

        linear_effect, non_linear_effect, pair_wise_effect = clf_fingerpint.get_effects()

        # Test that effects are computed
        self.assertTrue(len(linear_effect['raw']) == X.shape[1])
        self.assertTrue(len(non_linear_effect['raw']) == X.shape[1])
        self.assertTrue(len(pair_wise_effect['raw']) == 3)

    def test_plot_effects(self):
        """
        Test plot_effects function.
        """

        self.reg_fingerprint.fit(self.reg_rf, self.X, num_values=20)
        self.reg_fingerprint.plot_effects()

        self.reg_fingerprint.fit(self.reg_rf, self.X, num_values=20, pairwise_combinations=[(1, 2), (3, 5)])
        self.reg_fingerprint.plot_effects()
