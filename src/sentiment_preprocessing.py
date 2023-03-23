import string
import pandas as pd

import spacy
from spacy.lang.en import English
from spacy.lang.de import German

from nltk.corpus import stopwords as sw

class TextPreprocessing:
    """
     makes text preprocessing (so far only in English, German languages)
    """
    def __init__(self, config:dict, verbose=True, keep_original_cols=True) -> None:

        text_cols = config["feature_columns"]
        language_col = config["language_column"]

        self.punctuation = string.punctuation
        self.languages = {'de':'german', 'en':'english'}
        self.verbose = verbose
        self.linefiller = "="*30

        self.text_cols = text_cols
        if keep_original_cols:
            self.keep_original_cols = keep_original_cols
            self.text_cols_original = [f"{col}_original" for col in text_cols]
        self.language_col = language_col
        
    def lower(self, data: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
        """
        transforms given text columns into lowercase
        """
        data_out = data.copy()
        data_out.loc[:,text_cols] = data_out.loc[:,text_cols].astype(str).applymap(str.lower)
        if self.verbose:
            print(f"Lower-cased, shapes before:{data.shape} / after:{data_out.shape}")
            print(f"example: {data_out.iloc[0][text_cols[0]]}")
            print(self.linefiller)
        return data_out

    def _tokenize_string(self, text:str, language_model: spacy.lang) -> list[str]:
        """
        transforms given piece of text into tokenized list
        """
        document = language_model(text)
        tokenized_document = [token.text for token in document]
        return tokenized_document

    def tokenize(self, data: pd.DataFrame, text_cols:list[str], language_col:str):
        """
        transforms given list of columns into tokenized form
        """
        data_out = data.copy()
        language_models = {'de':German(), 'en':English()}

        # apply tokenizer of a respective language
        for language in set(data_out[language_col]):
            mask = data_out[language_col] == language
            language_model = language_models[language]
            data_out.loc[mask, text_cols] = data_out.loc[mask, text_cols].applymap(lambda cell: self._tokenize_string(cell, language_model))
        if self.verbose:
            print(f"Tokenized, shapes before:{data.shape} / after:{data_out.shape}")
            print(f"example: {data_out.iloc[0][text_cols[0]]}")
            print(self.linefiller)
        return data_out

    def remove_punctuation(self, data:pd.DataFrame, text_cols:list[str]):
        """
        removes punctuation from given columns in given dataframe
        """
        data_out = data.copy()
        data_out[text_cols] = data_out[text_cols].applymap(lambda cell: [word for word in cell if word not in string.punctuation])
        if self.verbose:
            print(f"Remove punctuation, shapes before:{data.shape} / after:{data_out.shape}")
            print(f"example: {data_out.iloc[0][text_cols[0]]}")
            print(self.linefiller)
        return data_out

    def remove_stopwords(self, data:pd.DataFrame, text_cols:list[str], language_col:str):
        """
        removes stopwords of a given language
        """
        data_out = data.copy()
        stopwords_dict = {'de': sw.words('german'), 'en':sw.words('english')}

        # apply tokenizer of a respective language
        for language in set(data_out[language_col]):
            mask = data_out[language_col] == language
            stopwords_list = stopwords_dict[language]
            data_out.loc[mask, text_cols] = data_out.loc[mask, text_cols].applymap(lambda cell: [word for word in cell if word not in stopwords_list])
        if self.verbose:
            print(f"Remove stopwords, shapes before:{data.shape} / after:{data_out.shape}")
            print(f"example: {data_out.iloc[0][text_cols[0]]}")
            print(self.linefiller)
        return data_out

    def _lemmatize_string(self, text: str, language_model: spacy.lang) -> str:
        """
        lemmitize a given word
        """
        spacy_doc = language_model(text)
        output = [token.lemma_ for token in spacy_doc]
        return " ".join(output)

    def lemmatize(self, data:pd.DataFrame, text_cols:list[str], language_col:str):
        """
        transform given columns of dataset into lemmas        
        """
        data_out = data.copy()
        language_models = {'de':spacy.load('de_dep_news_trf'),
                            'en':spacy.load('en_core_web_lg')}

        # apply tokenizer of a respective language
        for language in set(data_out[language_col]):
            mask = data_out[language_col] == language
            language_model = language_models[language]
            data_out.loc[mask, text_cols] = data_out.loc[mask, text_cols].applymap(lambda cell: self._lemmatize_string(cell, language_model))
        if self.verbose:
            print(f"Lemmatisation, shapes before:{data.shape} / after:{data_out.shape}")
            print(f"example: {data_out.iloc[0][text_cols[0]]}")
            print(self.linefiller)
        return data_out

    def fit(self, X=None, y=None) -> object:
        """
        dummy method: added for compatibility with scikit pipelines
        """
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """
        main method: named for compatibility with scikit pipelines
        """
        transformed = X.copy()

        if self.keep_original_cols:
            transformed[self.text_cols_original] = transformed[self.text_cols]

        if self.verbose:
            print(f"Step 1 - lowercasing")
        transformed = self.lower(transformed, self.text_cols)

        if self.verbose:
            print(f"Step 2 - tokenisation")
        transformed = self.tokenize(transformed, text_cols=self.text_cols, language_col=self.language_col)

        if self.verbose:
            print(f"Step 3 - remove punctuation")
        transformed = self.remove_punctuation(transformed, self.text_cols)

        if self.verbose:
            print(f"Step 4 - remove stopwords")
        transformed = self.remove_stopwords(transformed, text_cols=self.text_cols, language_col=self.language_col)

        if self.verbose:
            print(f"Step 5 - joining tokens into single string")
        transformed[self.text_cols] = transformed[self.text_cols].applymap(lambda cell: " ".join(cell))
        transformed.reset_index(drop=True, inplace=True)

        if self.verbose:
            print(f"Step 6 - lematisation")
        transformed = self.lemmatize(transformed, text_cols=self.text_cols, language_col=self.language_col)

        return transformed