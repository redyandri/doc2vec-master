from victorinox import victorinox
import nltk

tool=victorinox()
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

dataset_lower_clean_stem=r"data/dataset_lower_clean_stem.csv"
dataset_lower_clean_stem_sentence=r"data/dataset_lower_clean_stem_sentence.csv"
tool.create_sentence_list(csv_src=dataset_lower_clean_stem,
                         csv_dest=dataset_lower_clean_stem_sentence,
                         cols_to_write="KOMPETENSI",
                         sep=";")