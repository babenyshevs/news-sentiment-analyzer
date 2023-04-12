from helpers import *
from sklearn.pipeline import Pipeline
from sentiment_preprocessing import TextPreprocessing
from sentiment_postprocessing import PostProcessing
from sentiment_scoring import SentimentScoring

if __name__ == "__main__":
    config = load_config("./src/config.json")    
    parsed = from_pickle(config["parsed_file"])

    VERBOSE = True
    KEEP_ORIGINAL_COLS = True

    pipe = Pipeline(steps=[("preprocess", TextPreprocessing(config, verbose=VERBOSE, keep_original_cols=KEEP_ORIGINAL_COLS)),
                           ("score", SentimentScoring(config, verbose=VERBOSE)),
                           ("postprocess", PostProcessing(verbose=VERBOSE))])

    normalized, rescaled_str, rescaled_int = pipe.fit_transform(parsed)

    #save result
    to_pickle(file=normalized, filename=config["normalized_scores"])
    to_pickle(file=rescaled_str, filename=config["scaled_str_scores"])
    to_pickle(file=rescaled_int, filename=config["scaled_int_scores"])

    if VERBOSE:
        print(f"{'*'*15} Normalized scores file (shape:{normalized.shape}), top 5 rows:{'*'*15}\n {normalized.head()}")
        print(f"{'*'*15} Rescaled (str) scores file (shape:{rescaled_str.shape}), top 5 rows:{'*'*15}\n {rescaled_str.head()}")
        print(f"{'*'*15} Rescaled (int) scores file (shape:{rescaled_int.shape}), top 5 rows:{'*'*15}\n {rescaled_int.head()}")