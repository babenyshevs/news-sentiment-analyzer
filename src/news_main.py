from helpers import *
from sklearn.pipeline import Pipeline
from news_downloader import NewsDownloader
from news_parser import NewsParser

if __name__ == "__main__":
    config = load_config("./src/config.json")

    pipe = Pipeline(steps=[("download", NewsDownloader(config, debug=False)),
                            ("parse", NewsParser(debug=False))])
    
    parsed = pipe.fit_transform(None)
    to_pickle(parsed, config["parsed_file"])
    print(f"{'*'*15} parsed file (shape:{parsed.shape}), top 5 rows:{'*'*15}\n {parsed.head()}")