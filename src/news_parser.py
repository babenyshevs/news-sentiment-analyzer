import pandas as pd
from helpers import *

class NewsParser:
    """
    Reads, parses and saves news data from open-sources
    """
    def __init__(self) -> None:
        """
        setup main variables
        """
        pass

    
    @staticmethod
    def _newsline_generator(response: list[dict]) -> pd.DataFrame:
        """
        generates newslines for each query-language-date triplet; implemented as generator to save memory, when big pieces of text (=news) will come
        """
        for q_l_d_triplet in response:
            for (q,l,d), response in q_l_d_triplet.items():
                for news_line in response['articles']:
                    yield (q,l,d), news_line

    def fit(self, X=None, y=None) -> object:
            """
            (X, y) added for compatibility with scikit pipelines
            """
            return self

    def transform(self, X:list[dict]=None, y=None):        
        """
        main method, (X, y) added for compatibility with scikit pipelines
        """
        parsed_df = pd.DataFrame()
        for (q,l,d), news_line in self._newsline_generator(X):
            tmp_df = pd.DataFrame(news_line)
            tmp_df = tmp_df.reset_index().query("index == 'name'")
            tmp_df.drop(columns=['index','url','urlToImage','publishedAt'], inplace=True)
            tmp_df[["query", "language", "date"]] = q,l,d

            parsed_df = pd.concat([parsed_df, tmp_df], axis=0)
            parsed_df.reset_index(inplace=True, drop=True)

        return parsed_df

if __name__ == "__main__":
    from helpers import *
    config = load_config("./src/config.json")
    downloaded = from_pickle(config["responses_file"])

    parser = NewsParser()
    parsed = parser.transform(downloaded)
    to_pickle(parsed, config["parsed_file"])