import unittest

from experiments.train_full_more import temporal_indices


class TemporalSplitTests(unittest.TestCase):
    def test_validation_never_precedes_training(self):
        dates = [
            "2018-06",
            "2018-01",
            "2018-08",
            "2018-02",
            "2018-04",
            "2018-07",
            "2018-03",
            "2018-09",
            "2018-05",
            "2018-10",
        ]
        fit, validation = temporal_indices(dates, per_class=5)
        self.assertEqual((len(fit), len(validation)), (8, 2))
        self.assertLessEqual(
            max(dates[index] for index in fit), min(dates[index] for index in validation)
        )


if __name__ == "__main__":
    unittest.main()
