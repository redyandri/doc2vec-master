from nltk.tokenize import RegexpTokenizer
import pandas as pd

# tokenizer = RegexpTokenizer(r'\w+')
# print(tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!'))

s=[["1","2","3"],["why","cloud","gloomy"],["why","sun","shady"]]
df=pd.DataFrame(s,columns=["1","2","3"],index=None)
print(list(df["2"]))
