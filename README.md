# News sentiment analyzer

## **Project purpose** 

make some news parser on relevant topics:

## **Scripts**
There are essentially **4 types** of scripts/notebooks in the project. All are stored either in *src* folder (python scripts) or right in the *root directory* (jupyter notebooks)

### 1. Data downloading scripts
These scripts utilize **newsapi.org** to download latest news for respective date, query and language.
- news_downloader.py

Scripts is configured by **config file**, which is saved in the same (src) directory. Config structure:
- "apikey_file": *apikeyfile with apikey*
- "queries": *list of queries to sent*
- "languages": *list of languages*
- "days": *number days in the past to retrieve data*
- "request_sleep_time": *technical parameter to avoid "too many requests" error*
- "responses_file": *destination for responses json (binarized as pickle, retrieved only when run *news_downloader.py* directly)*
result of *news_main.py* run is file *parced_<VERSION>.pkl* (binarazed pandas-dataframe, which preserves dtypes, etc)

### 2. Scoring scripts
TBD

### 3. Helping
some additional helping functionalities, which don't belong to other categories
- fix_ssl.py - *helps to check and add (if needed) certificate for Huggingface.org (used as as source of pretrained transformers)*
- helpers.py - *some handy functions (namely picklezed save/load is imported from there)*

### 4. Notebooks
Analyses and visualisations for **EDA** and final **scores analysis**:
- eda.ipynb - *exploratory data analysis* TBD
- visualisatons.ipynb - *retrieved scores analysis* TBD
Some results in a form of *PNG* images are available in **reports** folder.

# VALIDATION DATA
models are validated on following data sets:
- English (taken from https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis and https://huggingface.co/datasets/sara-nabhani/ML-news-sentiment): 
  <pre><code>data/validation_data.csv</pre></code>

# DEPENDENCIES:
### NLTK
you may face an nltk-packages related error, which will require downloading data files ("brown", "stopwords", etc). There are 2 solutions for that:
- use standard download method:
  <pre><code>import nltk
  nltk.download("stopwords") </pre></code>
- download and place necessary data-files manually (especially, when facing *url* errors with standard method) into the folder:
  <pre><code> c:\Users\%USER%\AppData\Roaming\nltk_data\</pre></code>
  or 
  <pre><code>c:\nltk_data</pre></code>
  read more on this method here: https://www.nltk.org/data.html#manual-installation. Respective files can be found here: https://www.nltk.org/nltk_data/

### Spacy
wheel-packages, which are necessary for Spacy run of a current model are *de_dep_news_trf* and *en_core_web_lg*. They can be found in *./src/spacy_models* folder of a current project.
To install them, simply run following (e.g. for *en_core_web_lg* package):
<pre><code> pip install src/spacy_models/de_core_news_lg-3.5.0-py3-none-any.whl</pre></code>
rest of the wheel packages you may find in the same folder *./src/spacy_models*
# RESEARCH DIRECTIONS:
- [ ] Topic modelling
- [ ] Classification / Regression tasks with text embedings as regressors
- [ ] Wordcloud aggregation
- [ ] Sentiment barometer
- [ ] Sentiment time series

# TODO:
- [x] Download data
- [ ] Conduct EDA
- [ ] Engineer features
- [ ] Build a model
