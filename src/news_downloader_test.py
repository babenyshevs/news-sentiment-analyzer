from utilities import load_config
from news_downloader import NewsDownloader

import datetime as dt
import pandas as pd
import numpy as np
import unittest

class NewsDownloaderTest(unittest.TestCase):

    config = load_config("./src/config.json")

    # initiate instance to be tested
    instance = NewsDownloader(config, debug=False)
    instance.apikey = 1

    def test_get_today_minus_n(self):
        date = dt.date.today()
        n = 4
        expected = date - dt.timedelta(n)
        actual = self.instance._get_today_minus_n(n)
        self.assertEqual(expected, actual, f"get_today_minus_n - FAILED")
        # self.assertTrue(compare_results, "get_ids_count returns unexpected dataset")

    def test_build_request_query(self):
        expected = "https://newsapi.org/v2/everything?q=q&from=01.01.2023&to=01.01.2023&language=en&sortBy=publishedAt&apiKey=1"
        actual = self.instance.build_request_query("q", "en", "01.01.2023")
        self.assertEqual(expected, actual, f"build_request_query - FAILED")

    def test_get_response(self):
        expected = "https://api.github.com/user"
        actual = self.instance.get_response("https://api.github.com")['current_user_url']
        self.assertEqual(expected, actual, f"get_response - FAILED")

    # fit & transform aren't tested

if __name__ == "__main__":
    print("\n\n\n\n\n\n NEW OUTPUT **********************************************")
    unittest.main(verbosity=2)
