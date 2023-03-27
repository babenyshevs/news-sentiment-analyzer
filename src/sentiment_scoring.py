from transformers import pipeline
import torch
import tensorflow

import pandas as pd
from textblob import TextBlob
from textblob_de import TextBlobDE

# models:
# citizenlab/twitter-xlm-roberta-base-sentiment-finetunned
# nlptown/bert-base-multilingual-uncased-sentiment - fails, when called repeatedly

class SentimentScoring:
    """
    runs sentiment analysis
    """
    def __init__(self, config:dict, verbose=True):
        text_cols = config["feature_columns"]
        language_col = config["language_column"]

        self.textblob_language_models = {'de':TextBlobDE, 'en':TextBlob}
        self.transformer_models_names = config['transformers']
        self.verbose = verbose
        self.linefiller = "="*30
        self.text_cols = text_cols
        self.language_col = language_col

    def predict_w_text_blob(self, text:str, language:str):
        """
        predict based on TextBlobModel
        """
        if type(text) == list:
            text = " ".join(text)
        language_model = self.textblob_language_models[language]
        sentiment = language_model(text).sentiment[0]
        return sentiment
    

    def fit(self, X=None, y=None) -> object:
        """
        dummy method: added for compatibility with scikit pipelines
        """
        return self
    
    def transform(self, X, y= None):
        """
        main method: named for compatibility with scikit pipelines, predicts sentiment
        """
        data_out = X.copy()
        data_out[self.text_cols] = data_out[self.text_cols].fillna("None")

        if self.verbose:
            print('\nSCORING ', self.linefiller)
        for col in self.text_cols:
            if self.verbose:
                print(f' column: {col}\n  Textblob model - RUNNING')
            # rule-based models application (language dependence)
            for language in set(data_out[self.language_col]):
                mask = data_out[self.language_col] == language
                output_var_name = f"{col}_tb_sentiment"
                data_out.loc[mask, output_var_name] = data_out.loc[mask, col].apply(lambda cell: self.predict_w_text_blob(cell, language))
            if self.verbose:
                print(f'  Textblob model - DONE!\n  Transfomer models:')

            # apply transformer-model
            for model_name in self.transformer_models_names:
                if self.verbose:
                    print(f'   {model_name} - Running')
                # inference
                inputs = data_out[col].to_list()
                classifier = pipeline("sentiment-analysis", model=model_name)
                tmp_result = classifier(inputs)
                # build output
                new_col_name = f"{col}_" + str(model_name.split("/")[0]) + "_sentiment"
                df_tmp = pd.DataFrame(tmp_result)[['label']].rename(columns={'label':new_col_name})
                data_out = pd.concat((data_out, df_tmp), axis=1)
                if self.verbose:
                    print(f'   {model_name} - DONE!')
        
        return data_out