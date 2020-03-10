from gensim.summarization.summarizer import summarize
import pandas as pd
import numpy as np
from victorinox import victorinox

class MyIter(object):
    path=""
    def __init__(self,fp):
        self.path=fp

    def __iter__(self):
        # path = datapath(self.path)
        with open(self.path, 'r', encoding='utf-8') as fin:
            for line in fin:
               yield line

tool=victorinox()


csvsrc=r"data/dataset_lower_clean_stem_staff_group_with_periods.csv"
csvsummary=r"data/dataset_lower_clean_stem_staff_group_with_periods_summary.csv"


# df=pd.read_csv(csvsrc,sep=";")
# dftarget=df[df.values=="196808172003121001_syafarrudin"]#196302201983111001_dedysulistiarto"]
# text="\n".join(dftarget.iloc[:,0])
# print(text)
# print("########################################################")
# summ=summarize(text)
# print(summ)
# sumsentences=str(summ).splitlines()
# print("COUNT ori:%d" %len(dftarget))
# print("COUNT summary:%d" %len(sumsentences))


# df=pd.read_csv(csvsrc,sep=";")
# lengths=[]
# for idx,row in df.iterrows():
#     lengths.append(len(str(row["KOMPETENSI"]).split()))
# print(np.mean(lengths))
###9.421897018021408

corpus=MyIter(csvsrc)
with open(csvsummary,"a+") as f:
    i=1
    for line in corpus:
        parts=line.split(";")
        id = parts[0]
        if(id=="ID_PEGAWAI"):
            continue
        doc=parts[1]
        lineddoc="\n".join(doc.split("."))
        summary=summarize(lineddoc)
        summary=". ".join(summary.split("\n"))
        l=";".join([id,summary])
        f.write(l+"\n")
        print("\rwrite %d / 114253"%(i),end="",flush=True)
        i+=1


