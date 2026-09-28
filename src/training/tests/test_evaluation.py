"""Tests for binary-classification evaluation helpers."""

import unittest

import numpy as np

from training.job.utils.ml.evaluation import evaluate_binary_predictions


class EvaluationTests(unittest.TestCase):
    """Exercise aggregate and top-fraction metrics."""

    def test_binary_metrics_and_top_fraction_metrics(self):
        """Return expected metrics from valid positive probabilities."""
        metrics = evaluate_binary_predictions(
            y_true=np.array([0, 0, 1, 1]),
            y_probability=np.array([0.1, 0.4, 0.35, 0.8]),
            top_fractions=(0.25,),
        )

        self.assertEqual(metrics["observations"], 4)
        self.assertEqual(metrics["positives"], 2)
        self.assertAlmostEqual(metrics["prevalence"], 0.5)
        self.assertAlmostEqual(metrics["roc_auc"], 0.75)
        self.assertAlmostEqual(metrics["average_precision"], 5 / 6)

        top_metrics = metrics["top_fraction_metrics"][0]
        self.assertEqual(top_metrics["selected"], 1)
        self.assertEqual(top_metrics["positives"], 1)
        self.assertAlmostEqual(top_metrics["precision"], 1.0)
        self.assertAlmostEqual(top_metrics["recall"], 0.5)
        self.assertAlmostEqual(top_metrics["lift"], 2.0)

    def test_rejects_two_dimensional_probabilities(self):
        """Require callers to provide only positive-class probabilities."""
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            evaluate_binary_predictions(
                y_true=np.array([0, 1]),
                y_probability=np.array([[0.9, 0.1], [0.2, 0.8]]),
            )


if __name__ == "__main__":
    unittest.main()
