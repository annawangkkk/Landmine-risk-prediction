import pandas as pd
import numpy as np

def combine_and_seperate(root):
    # all/train_labeled/test_labeled
    train_org = pd.read_csv(root + '/processed_dataset/train/grid_features_labels.csv')
    test_org = pd.read_csv(root + '/processed_dataset/test/test_features_labels.csv')
    train_cols = set(train_org.columns)
    train_org.loc[train_org['rwi_missing'] == 1,'rwi'] = np.nan # set nan based on rwi_missing
    del train_org['rwi_missing']
    test_cols = set(test_org.columns)
    all_cols = list(train_cols.intersection(test_cols))
    all = pd.merge(train_org, test_org, how='outer', on=all_cols)
    del all['Abandono o Despojo Forzado de Tierras']
    del all['Confinamiento']
    del all['SIN INFORMACION ']
    all_categorical = all.loc[:,all.columns != 'rwi']
    all_categorical = all_categorical.fillna(0,downcast='infer')
    all_categorical['rwi'] = all['rwi']
    all_categorical.to_csv(root + '/processed_dataset/all/all.csv')
    train_all = all_categorical.iloc[:7019,:]
    test_all = all_categorical.iloc[7019:,:]
    
    train_labeled = train_all[train_all['mines_outcome'] != -1]
    train_unlabeled = train_all[train_all['mines_outcome'] == -1]
    test_labeled = test_all[test_all['mines_outcome'] != -1]
    test_unlabeled = test_all[test_all['mines_outcome'] == -1]
    all_labeled = all_categorical[all_categorical['mines_outcome'] != -1]
    all_unlabeled = all_categorical[all_categorical['mines_outcome'] == -1]
    
    train_all.to_csv(root + '/processed_dataset/train/train_all.csv')
    train_labeled.to_csv(root + '/processed_dataset/train/train_labeled.csv')
    train_unlabeled.to_csv(root + '/processed_dataset/train/train_unlabeled.csv')
    test_all.to_csv(root + '/processed_dataset/test/test_all.csv')
    test_labeled.to_csv(root + '/processed_dataset/test/test_labeled.csv')
    test_unlabeled.to_csv(root + '/processed_dataset/test/test_unlabeled.csv')
    all_labeled.to_csv(root + '/processed_dataset/all/all_labeled.csv')
    all_unlabeled.to_csv(root + '/processed_dataset/all/all_unlabeled.csv')
    return

def random_split(root):
    all = pd.read_csv(root + "/processed_dataset/all/all.csv",index_col=0)
    caldas_test = pd.read_csv(root + "/processed_dataset/caldas/test/test_all.csv",index_col=0)
    all_size = all.shape[0]
    test_size = caldas_test.shape[0]
    test_idx = sorted(list(np.random.choice(all_size, test_size, replace=False)))
    train_idx = list(set(list(range(all_size))) - set(test_idx))
    train_all = all.iloc[train_idx,:]
    test_all = all.iloc[test_idx,:]
    train_labeled = train_all[train_all['mines_outcome'] != -1]
    train_unlabeled = train_all[train_all['mines_outcome'] == -1]
    test_labeled = test_all[test_all['mines_outcome'] != -1]
    test_unlabeled = test_all[test_all['mines_outcome'] == -1]
    train_all.to_csv(root + '/processed_dataset/random/train/train_all.csv')
    train_labeled.to_csv(root + '/processed_dataset/random/train/train_labeled.csv')
    train_unlabeled.to_csv(root + '/processed_dataset/random/train/train_unlabeled.csv')
    test_all.to_csv(root + '/processed_dataset/random/test/test_all.csv')
    test_labeled.to_csv(root + '/processed_dataset/random/test/test_labeled.csv')
    test_unlabeled.to_csv(root + '/processed_dataset/random/test/test_unlabeled.csv')
    return


def sonson_split(root):
    all = pd.read_csv(root + "/processed_dataset/all/all.csv",index_col=0)
    test_all = all[all['Municipio'] == 'SONSON']
    train_all = all[all['Municipio'] != 'SONSON']
    train_labeled = train_all[train_all['mines_outcome'] != -1]
    train_unlabeled = train_all[train_all['mines_outcome'] == -1]
    test_labeled = test_all[test_all['mines_outcome'] != -1]
    test_unlabeled = test_all[test_all['mines_outcome'] == -1]
    train_all.to_csv(root + '/processed_dataset/sonson/train/train_all.csv')
    train_labeled.to_csv(root + '/processed_dataset/sonson/train/train_labeled.csv')
    train_unlabeled.to_csv(root + '/processed_dataset/sonson/train/train_unlabeled.csv')
    test_all.to_csv(root + '/processed_dataset/sonson/test/test_all.csv')
    test_labeled.to_csv(root + '/processed_dataset/sonson/test/test_labeled.csv')
    test_unlabeled.to_csv(root + '/processed_dataset/sonson/test/test_unlabeled.csv')
    return
        
    
def create_cv_fold(root, split):
    # create CV fold
    train_labeled = pd.read_csv(root + f"/processed_dataset/{split}/train/train_labeled.csv", index_col=0)
    train_labeled = train_labeled.sample(frac=1)
    group_indices = []
    intvl = train_labeled.shape[0]//5
    for i in range(5):
        if (i+1)*intvl > train_labeled.shape[0]:
            group_indices.append(list(range(i*intvl,train_labeled.shape[0])))
        else:
            group_indices.append(list(range(i*intvl,(i+1)*intvl)))
    for i in range(5):
        cv_valid = train_labeled.iloc[group_indices[i],:]
        cv_train_indices = []
        for j in range(5):
            if j != i:
                cv_train_indices.extend(group_indices[j])
        cv_train = train_labeled.iloc[cv_train_indices,:]
        cv_valid = cv_valid.sort_index()
        cv_train = cv_train.sort_index()
        cv_valid.to_csv(root + f'/processed_dataset/{split}/train/cv{i}/val.csv')
        cv_train.to_csv(root + f'/processed_dataset/{split}/train/cv{i}/train.csv')

    return

def create_geo_cv_fold(root, split):
    train_labeled = pd.read_csv(root + f"/processed_dataset/{split}/train/train_labeled.csv", index_col=0)
    cities, counts = np.unique(train_labeled['Municipio'],return_counts=True)
    city2count = dict(zip(cities, counts))
    count2city = dict()
    for idx, c in enumerate(counts):
        if c not in count2city:
            count2city[c] = [cities[idx]]
        else:
            count2city[c].append(cities[idx])

    # formulate as a k way number partition problem
    desc_count = sorted(counts)[::-1]
    res_count = {i:[] for i in range(5)}
    res_city = {i:[] for i in range(5)}
    # init
    for i in range(len(res_count)):
        largest = desc_count.pop(0)
        res_count[i].append(largest)
    # greedy
    while len(desc_count) > 0:
        largest = desc_count.pop(0)
        min_key = -1
        min_sum = sum(counts) + 1
        for i in range(len(res_count)):
            current_sum = sum(res_count[i])
            if current_sum < min_sum:
                min_key = i
                min_sum = current_sum
        res_count[min_key].append(largest)
    # map counts back to city
    for key in res_count:
        val = res_count[key]
        print(f"fold: {key}, card: {sum(val)}")
        city_fold = []
        for v in val:
            targets = count2city[v]
            if len(targets) > 1:
                city_fold.append(targets.pop(0))
            else:
                city_fold.append(targets[0])
        res_city[key] = city_fold
    print(res_city)
    # generate folds
    for i in range(5):
        cv_valid = train_labeled[train_labeled['Municipio'].isin(res_city[i])]
        other_cities = []
        for j in range(5):
            if j != i:
                other_cities.extend(res_city[j])
        cv_train = train_labeled[train_labeled['Municipio'].isin(other_cities)]
        cv_valid = cv_valid.sort_index()
        cv_train = cv_train.sort_index()
        cv_valid.to_csv(root + f'/processed_dataset/{split}/train/geoCV/cv{i}/val.csv')
        cv_train.to_csv(root + f'/processed_dataset/{split}/train/geoCV/cv{i}/train.csv')
    return