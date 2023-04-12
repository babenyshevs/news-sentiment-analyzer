import pandas as pd
import numpy as np

class PostProcessing:
    """
    Makes some postprocessing of scores, code is model - specific, i.e. expects scores as one of the options:
        -1, 0, 1
        negative, neutral, positive,
        1 star, 2 stars, 3 stars, 4 stars, 5 stars
    """
    def __init__(self, verbose=True):
        self.verbose = verbose

    def normalize_scores(self, score):
        """
        transforms score into integer without merging categories 
            i.e. positive, neutral, negative will become 1,0,-1, 
            while 1 star, 2 stars, 3 stars, 4 stars, 5 stars will become 1, 2, 3, 4, 5
        """
        if type(score) == float:
            return round(score)
        # string score - just return the respective number (i.e)
        elif score.lower().startswith(tuple(str(i) for i in range(1,6))):
            return int(score[0])
        elif score.lower().startswith("positive"):
            return 1
        elif score.lower().startswith("negative"):
            return -1
        elif score.lower().startswith("neutral"):
            return 0
        else:
            return np.nan
        
    def rescale_scores(self, score):
        """
        transforms score into string form and rescale it (usefull, when need to compare scores from different models),
            i.e. -1, 0, 1 will become negative, neutral, positve
            also 1 star and 2 stars are merged into negative category,
            4, 5 stars - into positive

        """
        if type(score) == float:
            rounded_score = round(score)
            if rounded_score == 1:
                return "positive"
            elif rounded_score == -1:
                return "negative"
            elif rounded_score == 0:
                return "neutral"
            else:
                return "none"
        # string score
        else:
            score = score.lower()
            postive_vals = ("positive","4","5")
            negative_vals = ("negative","1","2")
            neutral_vals = ("neutral","3")
            if score.startswith(postive_vals):
                return "positive"
            elif score.startswith(negative_vals):
                return "negative"
            elif score.startswith(neutral_vals):
                return "neutral"
            else:
                return "none"

    def score_int2char(self, score):
        """
        makes an interger score from textual name (positive is 1, neutral is 0, negative is -1)
        """
        if score == "positive":
            return 1
        elif score == "neutral":
            return 0
        elif score == "negative":
            return -1
        else:
            return np.nan
            
    def fit(self, X=None, y=None) -> object:
        """
        dummy method: added for compatibility with scikit pipelines
        """
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """
        main method: named for compatibility with scikit pipelines,
        makes transformation
        """
        normalized = X.copy()
        rescaled_string = X.copy()
        rescaled_integer = X.copy()
        self.sentiment_cols = [col for col in X.columns if "sentiment" in col]
        
        if self.verbose:
            print('POSTPROCESSING:\n normalaizing scores...')
        normalized[self.sentiment_cols] = normalized[self.sentiment_cols].applymap(self.normalize_scores)
        if self.verbose:
            print('  done!\n rescaling scores (string)...')
        rescaled_string[self.sentiment_cols] = rescaled_string[self.sentiment_cols].applymap(self.rescale_scores)
        if self.verbose:
            print('  done!\n rescaling scores (integer)...')
        rescaled_integer[self.sentiment_cols] = rescaled_string[self.sentiment_cols].applymap(self.score_int2char)
        if self.verbose:
            print('  done!')

        return normalized, rescaled_string, rescaled_integer