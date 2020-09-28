import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import umap
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

# List of functions:::

# Cluster details >>> No. of clusters, No. of Points in each cluster
def clusters_det(hdb_labels):
  m = 0
  no_of_clusters = 0
  for i in range(10):
    m += np.count_nonzero(hdb_labels == i)
    if np.count_nonzero(hdb_labels == i) != 0:
      no_of_clusters += 1
    print(f'{i}\t', np.count_nonzero(hdb_labels == i))

  print('total Data point\t', m)
  print('total No. of Clusters\t', no_of_clusters)

  return no_of_clusters

# Clusters Embedding
def embedding_1D(no_cluster, data, hdb_labels, y_labels, p_out = True):
  cluster_list = list() #Creat a list contains all clusters after Embedding
  label_list = list() #Creat a list contains all labels after Embedding
  original_image_list = list() #Original Images List

  for k in range(no_cluster):
    if p_out:
      print('\n\nEmbedding Cluster No. >>>\t', k)
    cluster_list.append(umap.UMAP(random_state=42, n_neighbors=30, min_dist=0, n_components=1).fit_transform(data[hdb_labels==k]))
    label_list.append(y_labels[hdb_labels==k])
    original_image_list.append(data[hdb_labels==k])

  return cluster_list, label_list, original_image_list

def exp_regions(avg, sub, rare, p_out = True):
  if avg<=0 or sub<=0 or rare<=0:
    print('Regions Error = At least One Region is Negative')
    return None
  elif avg + sub + rare != 100:
    print('Regions Error = Regions Sum is Not Equal to 100%')
    return None
  avg, sub, rare = avg/100, sub/100, rare/100
  R1_h = 0.5 + (avg/2)
  R1_l = 0.5 - (avg/2)
  R2_h = 1 - (rare/2)
  R2_l = 0 + (rare/2)
  av = R2_h - R2_l
  R = [R1_h, R1_l, R2_h, R2_l]
  if p_out:
    print(f'Average Region:\t{R[0]}-{R[1]}')
    print(f'Sub-Avg Region:\t{R[2]}-{R[0]} & {R[3]}-{R[0]}')
    print(f'Rare Region:\t{R[2]}-{1} & {0}-{R[3]}')
  return R

# Data Splitting
def split_data(n, avg, sub, rare, cluster_list, y_list, original_list, p_out = True):
  #creat seperate lists for splitted datasets
  R_1, R_2, R_3 = [[] for i in range(n)], [[] for i in range(n)], [[] for i in range(n)] #creat a list contains all Region 1 datasets
  R_1_labels, R_2_labels, R_3_labels = [[] for i in range(n)], [[] for i in range(n)], [[] for i in range(n)] #creat a list contains all Region 1 labels
  R_1_original, R_2_original, R_3_original = [[] for i in range(n)], [[] for i in range(n)], [[] for i in range(n)] #creat a list contains all Region 1 Original Images
  
  # Regions / Quantile Limits
  # Regions Limitis:
  regions_perc = exp_regions(avg, sub, rare)
  R1_high = regions_perc[0]
  R1_low = regions_perc[1]
  R2_high = regions_perc[2]
  R2_low = regions_perc[3]

  # Number of Rows list is Required for splitting/looping on all rows of each cluster

  no_rows = list() #creat a list for Number of Rows values of clusters

  co1 = 0 # creat counter to be used inside the loop

  for cluster_i in cluster_list:
    no_rows.append(cluster_i.shape[0]) #Cluster number of rows
    co1 += 1 #end of the loop, move to the next cluster, increase the counter by 1

  # Cluster splitting > 3 divisions (Average (Region 1), Sub-Average (Region 2), Rare (Region 3))
  co2 = 0 # creat counter to be used inside the loop

  for cluster_q in cluster_list:
    for r in range(no_rows[co2]):
      if cluster_q[r][0] <= np.quantile(cluster_q, R1_high, axis=0) and cluster_q[r][0] >= np.quantile(cluster_q, R1_low, axis=0): 
        R_1[co2] = np.append(R_1[co2], np.array(cluster_q[r], dtype= np.float32))
        R_1_labels[co2] = np.append(R_1_labels[co2], y_list[co2][r])
        R_1_original[co2].append(original_list[co2][r])
      elif cluster_q[r][0] <= np.quantile(cluster_q, R2_high, axis=0) and cluster_q[r][0] >= np.quantile(cluster_q, R2_low, axis=0): 
        R_2[co2] = np.append(R_2[co2], np.array(cluster_q[r], dtype= np.float32))
        R_2_labels[co2] = np.append(R_2_labels[co2], y_list[co2][r])
        R_2_original[co2].append(original_list[co2][r])
      else:
        R_3[co2] = np.append(R_3[co2], np.array(cluster_q[r], dtype= np.float32))
        R_3_labels[co2] = np.append(R_3_labels[co2], y_list[co2][r])
        R_3_original[co2].append(original_list[co2][r])

    R_1_original[co2] = np.array(R_1_original[co2])
    R_2_original[co2] = np.array(R_2_original[co2])
    R_3_original[co2] = np.array(R_3_original[co2])

    if p_out:
      #Printing splitted datasets (cluster regions) details
      print("\n************************************************************************************************************\n\n")
      print(f"\n>>>Cluster {co2} Splitted datasets:::<<<\n")
      print("\t\tData Size \tLabels Size")   
      print("Average \t", R_1[co2].size, "\t\t", R_1_labels[co2].size) if len(R_1[co2]) != 0 else print("Average \t","empty\t\t empty")
      print("Sub-Average \t", R_2[co2].size, "\t\t", R_2_labels[co2].size) if len(R_2[co2]) != 0 else print("Sub-Average \t","empty\t\t empty")
      print("Rare \t\t", R_3[co2].size, "\t\t", R_3_labels[co2].size) if len(R_3[co2]) != 0 else print("Rare \t\t","empty\t\t empty")

    co2 += 1 #end of the loop, increase the counter by 1

  return R_1, R_2, R_3, R_1_labels, R_2_labels, R_3_labels, R_1_original, R_2_original, R_3_original

#Plotting splitted Datasets
def show_split(R_1, R_2, R_3, cluster_no):
  #Plotting splitted datasets (cluster regions) histograms without KDE, with Borders
  print("\n************************************************************************************************************\n")
  print(f">>>Cluster {cluster_no} Splitted datasets:::<<<\n")   
  print("Average \t", R_1[cluster_no].size) if len(R_1[cluster_no]) != 0 else print("Average \t","empty")
  print("Sub-Average \t", R_2[cluster_no].size) if len(R_2[cluster_no]) != 0 else print("Sub-Average \t","empty")
  print("Rare \t\t", R_3[cluster_no].size) if len(R_3[cluster_no]) != 0 else print("Rare \t\t","empty")

  plt.figure(figsize=(12,7))

  #Plotting Region1 data
  sns.distplot(R_1[cluster_no], kde=False, hist=True, rug=True, bins=5, color='green', hist_kws = {'color':'#35dab6', 'edgecolor':'#d6c417',
                        'linewidth':1, 'linestyle':'--', 'alpha':0.9}, norm_hist= True);
  #Plotting Region2 data
  sns.distplot(R_2[cluster_no], kde=False, hist=True, rug=True, bins=5, color='blue', hist_kws = {'color':'#0096fe', 'edgecolor':'#e1f7d5',
                        'linewidth':1, 'linestyle':'--', 'alpha':0.9}, norm_hist= True);
  #Plotting Region3 data
  sns.distplot(R_3[cluster_no], kde=False, hist=True, rug=True, bins=5, color='red', hist_kws = {'color':'#ff0000', 'edgecolor':'#e6dbdc',
                        'linewidth':2, 'linestyle':'--', 'alpha':0.9}, norm_hist= True);

  print("\n************************************************************************************************************\n")

  return None

#Data Preparation and Train, Test, Validate splitting for training
def prepare_data(no_cluster, R_1_original, R_1_labels, R_2_original, R_2_labels, R_3_original, R_3_labels):
  #Region 1 arrays initiating
  x_tr_R01, x_v_R01, y_tr_R01, y_v_R01 = [], [], [], []
  #Region 2 arrays initiating
  x_tr_R02, x_v_R02, y_tr_R02, y_v_R02 = [], [], [], []
  #Region 3 arrays initiating
  x_tr_R03, x_v_R03, y_tr_R03, y_v_R03 = [], [], [], []


  for z1 in range(no_cluster):
    #############################
    ###~~~ 70% Training DS ~~~###
    #############################
    #REGION >>>1
    x_t1, x_v1, y_t1, y_v1 = train_test_split(R_1_original[z1], R_1_labels[z1], train_size= 0.7, test_size= 0.3)
    x_tr_R01.append(x_t1) # 70% Data
    x_v_R01.append(x_v1)
    y_tr_R01.append(y_t1) # 70% Labels
    y_v_R01.append(y_v1)

    #REGION >>>2
    x_t2, x_v2, y_t2, y_v2 = train_test_split(R_2_original[z1], R_2_labels[z1], train_size= 0.7, test_size= 0.3)
    x_tr_R02.append(x_t2) # 70% Data
    x_v_R02.append(x_v2)
    y_tr_R02.append(y_t2) # 70% Labels
    y_v_R02.append(y_v2)
    
    #REGION >>>3
    x_t3, x_v3, y_t3, y_v3 = train_test_split(R_3_original[z1], R_3_labels[z1], train_size= 0.7, test_size= 0.3)
    x_tr_R03.append(x_t3) # 70% Data
    x_v_R03.append(x_v3)
    y_tr_R03.append(y_t3) # 70% Labels
    y_v_R03.append(y_v3)

  #>>> 70% (Train) of Region 1 + Region 2 + Region 3
  x_tr, y_tr = [], []
  for x in range(no_cluster):
    x_tr.extend(x_tr_R01[x])
    y_tr.extend(y_tr_R01[x])
    x_tr.extend(x_tr_R02[x])
    y_tr.extend(y_tr_R02[x])
    x_tr.extend(x_tr_R03[x])
    y_tr.extend(y_tr_R03[x])

  x_tr = np.array(x_tr)
  y_tr = np.array(y_tr)

  #############################################
  ###~~~ 20% Validation DS & 10% Test DS ~~~###
  #############################################

  x_v, y_v = [], []
  for x in range(no_cluster):
    x_v.extend(x_v_R01[x])
    y_v.extend(y_v_R01[x])
    x_v.extend(x_v_R02[x])
    y_v.extend(y_v_R02[x])
    x_v.extend(x_v_R03[x])
    y_v.extend(y_v_R03[x])

  x_v = np.array(x_v)
  y_v = np.array(y_v)


  #>>> 20% (Valid) & 10% (Train) of Region 1
  x_va, x_te, y_va, y_te = train_test_split(x_v, y_v, train_size= 2/3, test_size= 1/3)

  return x_tr, y_tr, x_va, y_va, x_te, y_te

# >>> VGG FUNCTION <<< #
def prepareDataset (DS):
  DS = np.dstack([DS] * 3)
  DS = DS.reshape(-1, 28,28,3)
  DS = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in DS])
  DS = DS / 255.
  DS = DS.astype('float32')
  return DS