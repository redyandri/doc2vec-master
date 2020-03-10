from collections import Counter
import pandas as pd
import numpy as np
import re
import csv
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory,StopWordRemover,ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models import FastText
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity


class victorinox(object):
    def __init__(self):
        return

    def check_column_redundancy(self,
                              column_id=3,
                              fp="\a.csv"):
        records=[]
        with open(fp,"r") as file:
            lines=file.readlines()
            i=0
            for row in lines:
                if i==0:
                    i+=1
                    continue
                fields=str(row).split(";")
                try:
                    email = fields[column_id]
                    records.append(email)
                except Exception as x:
                    print(fields)
                    break
        c=Counter(records)
        redundant=[]
        for k,v in c.items():
            if int(v)>1:
                redundant.append(k)
        print(redundant)

    def check_column_multivalue(self,
                              column_id=3,
                              delimiter=" ",
                              fp="\a.csv"):
        records=[]
        with open(fp,"r") as file:
            lines=file.readlines()
            i=0
            for row in lines:
                if i==0:
                    i+=1
                    continue
                fields=str(row).split(";")
                try:
                    vals = fields[column_id]
                    flds=str(vals).split(delimiter)
                    if len(flds)>1:
                        records.append(vals)
                except Exception as x:
                    print(fields)
                    break
        print(records)

    def replace_column_string(self,
                              column_id=3,
                              old=",",
                              new="",
                              fp="\a.csv",
                              dest_fp="\b.csv"
                              ):
        records=[]
        with open(fp,"r") as file:
            lines=file.readlines()
            i=0
            for row in lines:
                if i==0:
                    i+=1
                    continue
                fields=str(row).split(";")
                try:
                    val = fields[column_id]
                    val=str(val).replace(old,new)
                    fields[column_id]=val
                    l=";".join(fields)
                    records.append(l)
                except Exception as x:
                    print(fields)
                    break
        # print(records)
        with open(dest_fp, "w+") as f:
            f.writelines(records)

    def split_csv_into_batches(self,
                              batch_length=400,
                              fp="\a.csv",
                              dest_fp="\b.csv",
                              sep=",",
                              insert_header=False,
                               replace_string=False
                              ):

        df=pd.read_csv(fp,sep=sep,delimiter=sep)
        npd=np.array(df)
        max=len(npd)
        err=0
        for i in range(0,max,batch_length):
            try:
              if i > max:
                break
              d = str(dest_fp).replace(".csv", "_" + str(i) + ".csv")
              to = i + batch_length
              arr = npd[i:to, :]
              # arr=np.chararray.encode(arr,"utf-8")
              np.savetxt(d, arr, fmt="%s", delimiter=sep,encoding="utf-8")
              if insert_header:
                self.insert_header(d)
              if replace_string:
                self.replace_string(";nan", sep, d)
              print("saved: ", str(d))
            except Exception as e:
                print("Error:", str(e))
                err+=1
                continue
        print("Done. %d erro found"%(err))


    def replace_string(self,
                              old=";nan",
                              new=";",
                              fp="\a.csv",
                              header=None
                              ):
        records = []
        with open(fp, "r") as file:
            lines = file.readlines()
            i = 0
            for row in lines:
                if header is not None:
                    i += 1
                    continue
                try:
                    val = str(row).replace(old, new)
                    records.append(val)
                except Exception as x:
                    print(val)
                    break
        # print(records)
        with open(fp, "w+") as f:
            f.writelines(records)

    def insert_header(self,fp="\a.csv",
                     header="NAME;LAST_NAME;PASSWORD;EMAIL;LOGIN;"
                            "IBLOCK_SECTION_NAME_1;IBLOCK_SECTION_NAME_2;IBLOCK_SECTION_NAME_3;"
                            "IBLOCK_SECTION_NAME_4;IBLOCK_SECTION_NAME_5\n"):
        records = []
        with open(fp, "r") as file:
            lines = file.readlines()
            i = 0
            records.append(header)
            for row in lines:
                try:
                    records.append(row)
                except Exception as x:
                    print(row)
                    break
        # print(records)
        with open(fp, "w+") as f:
            f.writelines(records)


    def select_by_column_string(self,
                              column_id=6,
                              column_val="Sekretariat Jenderal",
                              fp="\a.csv",
                              dest_fp="\b.csv",
                              header=None
                              ):
        records=[]
        with open(fp,"r") as file:
            lines=file.readlines()
            i=0
            for row in lines:
                if header is not None:
                    if i==0:
                        i+=1
                        continue
                fields=str(row).split(";")
                try:
                    val = fields[column_id]
                    if val==column_val:
                        l=";".join(fields)
                        records.append(l)
                except Exception as x:
                    print(fields)
                    break
        # print(records)
        with open(dest_fp, "w+") as f:
            f.writelines(records)


    def select_leaders(self,
                              echelon_id=1,
                              fp="\a.csv",
                              dest_fp="\b.csv",
                              header=None
                              ):
        records=[]
        with open(fp,"r") as file:
            lines=file.readlines()
            suffix=""
            if echelon_id==3:
                suffix=";"
            elif echelon_id==2:
                suffix=";;"
            elif echelon_id==1:
                suffix=";;;"
            else:
                suffix=""

            i=0
            for row in lines:
                if header is not None:
                    if i==0:
                        i+=1
                        continue
                # fields=str(row).split(";")
                try:
                   if str(row).__contains__(suffix):
                        records.append(row)
                        # l=";".join(fields)
                        # records.append(l)
                except Exception as x:
                    print(row)
                    break
        # print(records)
        with open(dest_fp, "w+") as f:
            f.writelines(records)

    def convert_xls_to_csv(self,xls_path="",
                           csv_path="",
                           sheet_name="",
                           index_col=None,
                           row_start_idx=1,
                           column_idx=[1,5]):
        data_xls = pd.read_excel(xls_path, sheet_name, index_col=index_col)
        data_xls=data_xls.iloc[row_start_idx:,column_idx]
        data_xls.to_csv(csv_path,
                        encoding='utf-8',
                        index=False,
                        sep=";",
                        header=None)
        print("done on %d rows"%data_xls.shape[0])

    def merge_csv(self,csv_path1="",
                           csv_path2="",
                    csv_dest="",
                            sep=";",
                            join_op="left",
                           join_col="",
                            out_cols=["training","nip"]):
        data_xls1 = pd.read_csv(csv_path1,sep=sep)
        data_xls2 = pd.read_csv(csv_path2, sep=sep)
        data_xls1 = data_xls1.astype({col: str for col in data_xls1.columns})
        data_xls2 = data_xls2.astype({col: str for col in data_xls2.columns})
        data_xls3=pd.merge(data_xls1,
                           data_xls2,
                           how=join_op,
                           on=join_col)
        data_xls3["nip"]=data_xls3["nip"]+"_"+data_xls3["name"]
        if out_cols!= None:
            data_xls3[out_cols].to_csv(csv_dest,encoding='utf-8',index=False,sep=";")
        else:
            data_xls3.to_csv(csv_dest,encoding='utf-8',index=False,sep=";")
        print("done on %d rows"%data_xls3.shape[0])

    def replace_string_in_file(self,
                               file_path="",
                               file_path_dest="",
                               string_to_replace="",
                               replacement_string=""):
        regex = re.compile(r"\d{18},", re.IGNORECASE)
        regex2 = re.compile(r"\d{18}", re.IGNORECASE)
        res=[]
        with open(file_path,encoding="utf8") as f:
            lines =f.readlines()
            for line in lines:
                try:
                    nip = regex2.findall(line)[0]
                    line = regex.sub(nip + ";", line)
                    line=line.replace("\n","")
                    res.append([x for x in line.split(";")])
                except Exception as e:
                    print("error line:%s"%line)
                    continue
        nparr=np.array(res)
        df=pd.DataFrame(nparr)
        df.to_csv(file_path_dest,
                                   header=None,
                                   index=None,
                                    sep=";")
        print("done saving %d rows"%len(res))


    def switch_columns(self,
                       csv_path="",
                       csv_dest_path="",
                       sep=";"):
        df=pd.read_csv(csv_path,delimiter=sep,error_bad_lines=False)
        df=df.iloc[:,[-1,-2]]
        df.to_csv(csv_dest_path,
                  sep=sep,
                  header=None,
                                   index=None)

    def lower_contents(self,
                       csv_src="",
                       csv_dest="",
                       sep=";"):
        df=pd.read_csv(csv_src,sep=sep)
        for c in df.columns:
            df[c]=df[c].str.lower()
        df.to_csv(csv_dest,sep=sep,index=None)
        print("lower %d rows"%len(df))


    def remove_stopwords(self,csv_src="",
                         csv_dest="",
                         cols_to_clean=["KOMPETENSI"],
                         sep=";"):
        #factory = StopWordRemoverFactory()
        default_stopwords = StopWordRemoverFactory().get_stop_words()
        additional_stopwords=["(",")","senin","selasa","rabu","kamis","jumat","sabtu","minggu"]
        dictionary=ArrayDictionary(default_stopwords+additional_stopwords)
        stopword = StopWordRemover(dictionary)#factory.create_stop_word_remover(dictionary = dictionary)
        tokenizer = RegexpTokenizer(r'\w+')
        df = pd.read_csv(csv_src, sep=sep)
        for c in cols_to_clean:
            df[c] = df[c].map(lambda x: " ".join(tokenizer.tokenize(x)))    #get only words without symbols
            df[c]=df[c].map(lambda x:stopword.remove(x))                    #remove stop words
        df.to_csv(csv_dest, sep=sep, index=None)
        print("lower %d rows" % len(df))


    def stem(self,csv_src="",
                         csv_dest="",
                         cols_to_clean="KOMPETENSI",
                         sep=";"):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        df = pd.read_csv(csv_src, sep=sep)
        df[cols_to_clean]=df[cols_to_clean].astype(str)
        df[cols_to_clean] = df[cols_to_clean].map(lambda x: stemmer.stem(x))
        df.to_csv(csv_dest, sep=sep, index=None)
        print("lower %d rows" % len(df))

    def create_sentence_list(self,
                         csv_src="",
                         csv_dest="",
                         cols_to_write="KOMPETENSI",
                         sep=";"):
        df = pd.read_csv(csv_src, sep=sep)
        df[cols_to_write].to_csv(csv_dest, sep=sep, index=None)
        print("lower %d rows" % len(df))

    def document_vector(self,word2vec_model, doc):
        # remove out-of-vocabulary words
        doc = [word for word in doc if word in word2vec_model.wv.vocab]
        return np.mean(word2vec_model[doc], axis=0)

    def measure_similarity(self,vec1,vec2):
        vec1=np.array(vec1).reshape(1,-1)
        vec2 = np.array(vec2).reshape(1, -1)
        return cosine_similarity(vec1,vec2)

    def group_sentences(self,
                        csv_src="",
                         csv_dest="",
                         col_to_groupby="ID_PEGAWAI",
                         col_to_group="KOMPETENSI",
                         sep=";",
                        sentence_link=" "):
        df=pd.read_csv(csv_src,sep=sep)
        df2= df.groupby(col_to_groupby)
        ids=[]
        datas = []
        for group_name, dfgroup in df2:
            groupcontent=""
            for idx, row in dfgroup.iterrows():
                groupcontent+=str(row[col_to_group])+sentence_link
            datas.append(groupcontent)
            ids.append(row[col_to_groupby])
        result={col_to_groupby:ids,
                col_to_group:datas}
        dfresult=pd.DataFrame(result)
        dfresult.to_csv(csv_dest, sep=sep, index=None)
        print("group into %d rows"%len(df))

    def get_list_from_txt(self,
                         txt_src="",
                        sep=";"):
        with open(txt_src) as f:
            mylist=f.read().splitlines()
        return mylist

    def concat_dataframe(self,df1,df2,axis=1,csv_dest=""):
        df3 = pd.concat([df1, df2], axis=1)
        df3.to_csv(csv_dest,sep=";",index=None)
        print("done merging %d rows"%len(df3))

    def preprocess_sentence(self,q=""):
        #tokenize, lower, stopword,stem
        default_stopwords = StopWordRemoverFactory().get_stop_words()
        additional_stopwords = ["(", ")", "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu"]
        dictionary = ArrayDictionary(default_stopwords + additional_stopwords)
        stopword = StopWordRemover(dictionary)
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        tokenizer = RegexpTokenizer(r'\w+')
        res=" ".join(tokenizer.tokenize(q))
        res=res.lower()
        res=stopword.remove(res)
        res=factory =stemmer.stem(res)
        return res







