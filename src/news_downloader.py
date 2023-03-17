import requests
import datetime
import time
from utilities import *

class NewsDownloader:
    """
    Downloads and saves news data from open-sources
    Possible error: You have made too many requests recently. 
                    Developer accounts are limited to 100 requests over a 24 hour period 
                    (50 requests available every 12 hours)
    """
    def __init__(self, config:dict, read_from_disk=False) -> None:
        """
        setup main variables, provide apikey in separate file (not included in repo for security reasons)
        """
        self.config = config
        # a little workaround to avoid exposing apikey
        apikey_file = self.config['apikey_file']
        self.apikey = load_config(apikey_file)['apikey']
        self.queries = self.config['queries']
        self.languages = self.config['languages']
        self.days = self.config['days']
        start_from = 1
        start_to = start_from + self.days + 1
        self.dates_list = [self._get_today_minus_n(n) for n in range(start_from,start_to)]
        self.sleep = self.config['request_sleep_time']
        self.responses_file = self.config['responses_file']

        # other
        self.read_from_disk=read_from_disk

    @staticmethod
    def _get_today_minus_n(n:int) -> datetime.date:
        """
        gets date shifted for n days back from today's date
        """
        today = datetime.date.today()
        today_minus_n = today - datetime.timedelta(n)
        return today_minus_n

    def build_request_query(self, query:str, language:str, date_request:datetime.date) -> str:
        """
        builds request query from input parameters
        """
        request_string = f"https://newsapi.org/v2/everything?q={query}&from={date_request}&to={date_request}&language={language}&sortBy=publishedAt&apiKey={self.apikey}"
        return request_string

    def get_response(self, request:str) -> dict:
        """
        gets response from opens-ourse API based on request query
        """
        time.sleep(self.sleep) #to avoid too many requests error
        response = requests.get(request, verify=False)
        response_json = response.json()
        return response_json

    def fit(self, X=None, y=None) -> object:
        """
        Method and (X, y) added for compatibility with scikit pipelines
        """
        return self

    def transform(self, X=None, y=None) -> list[dict]:        
        """
        Main method, (X, y) added for compatibility with scikit pipelines
        """
        query_params = [[q,l,d] for q in self.queries for l in self.languages for d in self.dates_list]

        # configured for the time of development&debugging to avoid extensive requests above daily quota
        if self.read_from_disk:
            responses = from_pickle(self.responses_file)
        else:
            responses = [{(q,l,d): self.get_response(self.build_request_query(q,l,d))} for q,l,d in query_params]

        return responses

if __name__ == "__main__":
    config = load_config("./src/config.json")
    downloader = NewsDownloader(config, read_from_disk=False)
    request_downloaded = downloader.transform()
    to_pickle(request_downloaded, config["responses_file"])