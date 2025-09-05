from DataPreprocessor import   DataPreprocessor


preprocess = DataPreprocessor("Data/aapl-merged-T.csv")

x = preprocess.run("AAPL","4s")


x