import os
import numpy as np

list_file = [
        "acc_matrix_dn-sd14.csv",
        "auc_matrix_dn-sd14.csv",
        "acc_matrix_dn-glide.csv",
        "auc_matrix_dn-glide.csv",
        "acc_matrix_dn-mj.csv",
        "auc_matrix_dn-mj.csv",
        "acc_matrix_dn-dallemini.csv",
        "auc_matrix_dn-dallemini.csv",
        "acc_matrix_dn-tt.csv",
        "auc_matrix_dn-tt.csv",
        "acc_matrix_dn-sd21.csv",
        "auc_matrix_dn-sd21.csv",
        "acc_matrix_dn-cips.csv",
        "auc_matrix_dn-cips.csv",
        "acc_matrix_dn-biggan.csv",
        "auc_matrix_dn-biggan.csv",
        "acc_matrix_dn-vqdiff.csv",
        "auc_matrix_dn-vqdiff.csv",
        "acc_matrix_dn-diffgan.csv",
        "auc_matrix_dn-diffgan.csv",
        "acc_matrix_dn-sg3.csv",
        "auc_matrix_dn-sg3.csv",
        "acc_matrix_dn-gansformer.csv",
        "auc_matrix_dn-gansformer.csv",
        "acc_matrix_dn-dalle2.csv",
        "auc_matrix_dn-dalle2.csv",
        "acc_matrix_dn-ld.csv",
        "auc_matrix_dn-ld.csv",
        "acc_matrix_dn-eg3d.csv",
        "auc_matrix_dn-eg3d.csv",
        "acc_matrix_dn-projgan.csv",
        "auc_matrix_dn-projgan.csv",
        "acc_matrix_dn-sd1.csv",
        "auc_matrix_dn-sd1.csv",
        "acc_matrix_dn-ddg.csv",
        "auc_matrix_dn-ddg.csv",
        "acc_matrix_dn-ddpm.csv",
        "auc_matrix_dn-ddpm.csv",
]
acc_matrix = np.zeros(shape=(20, 20))
auc_matrix = np.zeros(shape=(20, 20))


for i, (acc_path, auc_path) in enumerate(zip(list_file[::2], list_file[1::2])):
    if os.path.exists("csvs/" + acc_path):
        small_acc_mat = np.genfromtxt("csvs/" + acc_path, delimiter=',')
        small_auc_mat = np.genfromtxt("csvs/" + auc_path, delimiter=',')
    else:
        break
    
    if i == 0:
        acc_matrix[0][0] = small_acc_mat[0][0]
        auc_matrix[0][0] = small_auc_mat[0][0]
        
    try:
        acc_matrix[i+1][0] = small_acc_mat[1][0]
        acc_matrix[i+1][i+1] = small_acc_mat[1][1]
        
        auc_matrix[i+1][0] = small_auc_mat[1][0]
        auc_matrix[i+1][i+1] = small_auc_mat[1][1]
    except:
        break


np.savetxt("csvs/acc_matrix.csv", acc_matrix, delimiter=",")
np.savetxt("csvs/auc_matrix.csv", auc_matrix, delimiter=",")
print("done")