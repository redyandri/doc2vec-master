from victorinox import victorinox

tool=victorinox()
xls_p=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\training.xlsx"
csv_p=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\training.csv"
xls_pegawai=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\pegawai_pusintek.xlsx"
csv_pegawai=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\pegawai.csv"

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

# csv_nip_training=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\pegawai_training.csv"
# tool.merge_csv(csv_path1=csv_p,
#                csv_path2=csv_pegawai,
#                csv_dest=csv_nip_training,
#                sep=";",
#                join_op="left",
#                join_col="name")

# disposisi_path=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\disposisi_pusintek.csv"
# disposisi_normalized_path=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\disposisi_normalized_pusintek.csv"
# tool.replace_string_in_file(file_path=disposisi_path,
#                             file_path_dest=disposisi_normalized_path)


disposisi_normalized_path=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\disposisi_normalized_pusintek.csv"
disposisi_normalized_path2=r"C:\Users\redy.andriyansah\Documents\project\competence_analytics\doc2vec\doc2vec-master\data\disposisi_normalized_pusintek2.csv"
tool.switch_columns(disposisi_normalized_path,disposisi_normalized_path2)