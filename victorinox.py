from collections import Counter
import pandas as pd
import numpy as np
import re
import csv


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







