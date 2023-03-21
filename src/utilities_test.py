
from utilities import *
import pandas as pd
import numpy as np
import unittest
import os


class HelpersTest(unittest.TestCase):
    data = pd.DataFrame({'number': np.arange(-6,6,1).tolist(), 
                            'text': [f"{n}" for n in np.arange(-6,6,1).tolist()],
                            'date': [pd.to_datetime('1/1/2000')]*12})
    datafile = "test_data"
    pickle_name = f"{datafile}.pkl"
    expected_shape = (12,3)
    config_path = "./src/config.json"

    def test_to_pickle(self):
        save_result = to_pickle(self.data, self.datafile)
        self.assertEqual(save_result, True, "Saving to pickle failed")
        os.remove(self.pickle_name)
    
    def test_from_pickle(self):
        with open(self.pickle_name, 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        read_result = from_pickle(self.datafile)
        self.assertEqual(read_result.shape, self.expected_shape, "Reading from pickle failed")
        os.remove(self.pickle_name)

    def test_get_dtypes_dict(self):
        actual = get_dtypes_dict(self.data)
        expected = {'date':['date'],
                    'str':['text'],
                    'numeric':['number']}
        self.assertEqual(expected, actual, "Getting dtypes failed")
    
    def test_transform_to_positive(self):
        actual = transform_to_positive(self.data['number'])
        expected = pd.Series(np.arange(0,12,1).tolist())
        compare_result = actual.equals(expected)
        self.assertEqual(compare_result, True, "Transforming to positive failed")

    def test_filter_high_variability(self):
        actual_all_colls = filter_high_variability(self.data, min_values=1)
        compare_all_cols = actual_all_colls.shape == self.expected_shape
        actual_two_cols = filter_high_variability(self.data, min_values=2)
        compare_two_cols = actual_two_cols.shape == (12,2)
        actual_no_cols = filter_high_variability(self.data, min_values=13)
        compare_no_cols = actual_no_cols.shape == (12,0)
        compare_result = compare_all_cols & compare_two_cols & compare_no_cols
        self.assertEqual(compare_result, True, "Filter high variability failed")

if __name__ == "__main__":
    unittest.main(verbosity=2)
