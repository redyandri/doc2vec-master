from victorinox import victorinox
import nltk
import pandas as pd
import logging
import pickle

#enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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

datapath_staffs=r"data/dataset_lower_clean_stem_staff.csv"
datapath_staffs_nip=r"data/dataset_lower_clean_stem_staff_with_nip.csv"
dataset_vector=r"data/tfidf_per_sentence_vectors.csv"
dataset_vector_nip=r"data/tfidf_per_sentence_vectors_nip.csv"
dataset_vector_idseq=r"data/tfidf_per_sentence_vectors_idseq.csv"
staff_dictionary=r"data/staff_dictionary.pkl"
staff_dictionary_by_sequence=r"data/staff_dictionary_by_sequence.pkl"
staff_dictionary_by_sequence_reveresed=r"data/staff_dictionary_by_sequence_reversed.pkl"
# xls_p=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\training.xlsx"
# csv_p=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\training.csv"
# xls_pegawai=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\pegawai_pusintek.xlsx"
# csv_pegawai=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\pegawai.csv"

# # get name,training
# tool.convert_xls_to_csv(xls_path=xls_p,
#                         csv_path=csv_p,
#                         sheet_name="training",
#                         row_start_idx=1,
#                         column_idx=[1,5])
#
# # get nip, name
# tool.convert_xls_to_csv(xls_path=xls_pegawai,
#                         csv_path=csv_pegawai,
#                         sheet_name="Data",
#                         row_start_idx=1,
#                         column_idx=[-2,-1])

# csv_nip_training=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\training_nipname.csv"
# tool.merge_csv(csv_path1=csv_p,
#                csv_path2=csv_pegawai,
#                csv_dest=csv_nip_training,
#                sep=";",
#                join_op="left",
#                join_col="name")

# name_nip=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\name_nip.csv"
# disposisi_nip=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\disposisi_normalized_pusintek2.csv"
# disposisi_nipname=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\disposisi_nipname.csv"
# tool.merge_csv(csv_path1=disposisi_nip,
#                csv_path2=name_nip,
#                csv_dest=disposisi_nipname,
#                sep=";",
#                join_op="left",
#                join_col="nip",
#                out_cols=["disposisi","nip"])

# disposisi_path=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\disposisi_pusintek.csv"
# disposisi_normalized_path=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\disposisi_normalized_pusintek.csv"
# tool.replace_string_in_file(file_path=disposisi_path,
#                             file_path_dest=disposisi_normalized_path)


# disposisi_normalized_path=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\disposisi_normalized_pusintek.csv"
# disposisi_normalized_path2=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\disposisi_normalized_pusintek2.csv"
# tool.switch_columns(disposisi_normalized_path,disposisi_normalized_path2)

# dataset=r"data/id_kompetensi_flag.csv"
# dataset_lower=r"data/id_kompetensi_flag_lower.csv"
# tool.lower_contents(csv_src=dataset,
#                     csv_dest=dataset_lower,
#                     sep=";")

# dataset_lower=r"data/id_kompetensi_flag_lower.csv"
# dataset_lower_clean=r"data/dataset_lower_clean.csv"
# tool.remove_stopwords(csv_src=dataset_lower,
#                          csv_dest=dataset_lower_clean,
#                          cols_to_clean=["KOMPETENSI"],
#                          sep=";")

#nltk.download('wordnet')

# dataset_lower_clean=r"data/dataset_lower_clean.csv"
# dataset_lower_clean_stem=r"data/dataset_lower_clean_stem.csv"
# tool.stem(csv_src=dataset_lower_clean,
#                          csv_dest=dataset_lower_clean_stem,
#                          cols_to_clean="KOMPETENSI",
#                          sep=";")

# dataset_lower_clean_stem=r"data/dataset_lower_clean_stem.csv"
# dataset_lower_clean_stem_sentence=r"data/dataset_lower_clean_stem_sentence.csv"
# tool.create_sentence_list(csv_src=dataset_lower_clean_stem,
#                          csv_dest=dataset_lower_clean_stem_sentence,
#                          cols_to_write="KOMPETENSI",
#                          sep=";")


# dataset_lower_clean_stem=r"data/dataset_lower_clean_stem.csv"
# dataset_lower_clean_stem_group=r"data/dataset_lower_clean_stem_group.csv"
# tool.group_sentences(csv_src=dataset_lower_clean_stem,
#                          csv_dest=dataset_lower_clean_stem_group,
#                          col_to_groupby="ID_PEGAWAI",
#                          sep=";")


# xls_pegawai=r"data/monitoring data pegawai april 2018 -share.xlsx"
# csv_pegawai=r"data/pegawai.csv"
# tool.convert_xls_to_csv(xls_path=xls_pegawai,
#                         csv_path=csv_pegawai,
#                         sheet_name="monev",
#                         row_start_idx=0,
#                         column_idx=[0,1,2,3,4,5,6,7,8,9])


# csv_employees=r"data/dataset_lower_clean_stem_group.csv"
# csv_leaders=r"data/leaders.csv"
# csv_emploees_noleader=r"data/dataset_lower_clean_stem_group_staffs.csv"
# leaders=[]
# with open(csv_leaders) as f:
#     leaders=f.read().splitlines()
# df=pd.read_csv(csv_employees,sep=";")
# ID_PEGAWAI=[]
# KOMPETENSI=[]
# for idx,row in df.iterrows():
#     nip=str(row["ID_PEGAWAI"]).split("_")[0]
#     if(nip in leaders):
#         continue
#     ID_PEGAWAI.append(row["ID_PEGAWAI"])
#     KOMPETENSI.append(row["KOMPETENSI"])
# newdf_json={"ID_PEGAWAI":ID_PEGAWAI,
#        "KOMPETENSI":KOMPETENSI}
# newdf=pd.DataFrame(newdf_json)
# newdf.to_csv(csv_emploees_noleader,sep=";",index=None)


# csv_emploees_noleader=r"data/dataset_lower_clean_stem_group_staffs.csv"
# csv_emploees_noleader_sentences=r"data/dataset_lower_clean_stem_group_staffs_sentences.csv"
# df=pd.read_csv(csv_emploees_noleader,sep=";")
# df=df.KOMPETENSI
# df.to_csv(csv_emploees_noleader_sentences,index=None)




# corpus=MyIter(dataset_vector)
# with open(dataset_vector_nip,"a+") as f:
#     for line in corpus:
#         parts=line.split(";")
#         vec = parts[0:-1]
#         id = parts[-1].replace("\n", "")
#         nip=id.split("_")[0]
#         l=vec+[nip]
#         f.write(";".join(l))
#         f.write("\n")


# corpus=MyIter(dataset_vector)
# dct={}
# with open(dataset_vector_nip,"a+") as f:
#     for line in corpus:
#         parts=line.split(";")
#         id = parts[-1].replace("\n", "")
#         idparts=id.split("_")
#         nip=idparts[0]
#         name = idparts[-1]
#         dct[nip]=name
# with open(staff_dictionary,"wb+") as f:
#     pickle.dump(dct,f)
# with open(staff_dictionary,"rb") as f:
#     kamus=pickle.load(f)
# print("nip:198401112009011004, name:%s"%(kamus["198401112009011004"]))


# corpus=MyIter(dataset_vector)
# dct={}
# i=0
# with open(staff_dictionary,"rb") as f:
#     kamus=pickle.load(f)
# for k,v in kamus.items():
#     dct[i]=k+"_"+v
#     i+=1
# with open(staff_dictionary_by_sequence,"wb+") as f:
#     pickle.dump(dct,f)
# with open(staff_dictionary_by_sequence,"rb") as f:
#     kamus2=pickle.load(f)
# print("employee id 0=%s"%(kamus2[0]))



# dct={}
# with open(staff_dictionary_by_sequence,"rb") as f:
#     kamus=pickle.load(f)
# for k,v in kamus.items():
#     dct[v]=k
# with open(staff_dictionary_by_sequence_reveresed,"wb+") as f:
#     pickle.dump(dct,f)
# with open(staff_dictionary_by_sequence_reveresed,"rb") as f:
#     kamus2=pickle.load(f)
# print("employee id 198401112009011004_redyandriyansah=%s"%(kamus2["198401112009011004_redyandriyansah"]))



# with open(staff_dictionary_by_sequence_reveresed,"rb") as f:
#     kamus=pickle.load(f)
# corpus=MyIter(dataset_vector)
# with open(dataset_vector_idseq,"a+") as f:
#     for line in corpus:
#         parts=line.split(";")
#         vec = parts[0:-1]
#         id = parts[-1].replace("\n", "")
#         newid=str(kamus[id])
#         l=vec+[newid]
#         f.write(";".join(l))
#         f.write("\n")



dataset_lower_clean_stem=r"data/dataset_lower_clean_stem_staff.csv"
dataset_lower_clean_stem_group_in_lines=r"data/dataset_lower_clean_stem_staff_group_with_periods.csv"
tool.group_sentences(csv_src=dataset_lower_clean_stem,
                         csv_dest=dataset_lower_clean_stem_group_in_lines,
                         col_to_groupby="ID_PEGAWAI",
                         sep=";",
                     sentence_link=". ")







