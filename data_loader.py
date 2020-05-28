import numpy as np
import pandas as pd
import torch
import os
import random
import numpy as np
import gzip
import shutil
import bz2
import requests
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from category_encoders import LeaveOneOutEncoder
from sklearn.preprocessing import LabelEncoder
from google_drive_downloader import GoogleDriveDownloader as gdd


'''
Rewrite the data processing functions, and add two more new data sets for testing.
'''

dataset_url = {
    'A9A':{'train':{'url':'https://www.dropbox.com/s/9cqdx166iwonrj9/a9a?dl=1',
                    'extension':''},
           'test' :{'url': 'https://www.dropbox.com/s/sa0ds895c0v4xc6/a9a.t?dl=1',
                    'extension':''},
           'train_idx':{'url':'https://www.dropbox.com/s/xy4wwvutwikmtha/stratified_train_idx.txt?dl=1',
                        'extension':'.txt'},
           'valid_idx':{'url':'https://www.dropbox.com/s/nthpxofymrais5s/stratified_test_idx.txt?dl=1',
                        'extension':'.txt'}},
    
    'EPSILON':{ 'train_archive':{'url': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2',
                    'extension':'.bz2'},
                'test_archive' :{'url': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2',
                    'extension':'.bz2'},
                'train_idx':{'url':'https://www.dropbox.com/s/wxgm94gvm6d3xn5/stratified_train_idx.txt?dl=1',
                        'extension':'.txt'},
                'valid_idx':{'url':'https://www.dropbox.com/s/fm4llo5uucdglti/stratified_valid_idx.txt?dl=1',
                        'extension':'.txt',}},

    'PROTEIN':{ 'train':{'url': 'https://www.dropbox.com/s/pflp4vftdj3qzbj/protein.tr?dl=1',
                    'extension':''},
                'test' :{'url': 'https://www.dropbox.com/s/z7i5n0xdcw57weh/protein.t?dl=1',
                    'extension':''},
                'train_idx':{'url':'https://www.dropbox.com/s/wq2v9hl1wxfufs3/small_stratified_train_idx.txt?dl=1',
                        'extension':'.txt'},
                'valid_idx':{'url':'https://www.dropbox.com/s/7o9el8pp1bvyy22/small_stratified_valid_idx.txt?dl=1',
                        'extension':'.txt'}},

                
    
    'YEAR': { 'data':{'url': 'https://www.dropbox.com/s/l09pug0ywaqsy0e/YearPredictionMSD.txt?dl=1',
                    'extension':'.txt'},
                'train_idx':{'url':'https://www.dropbox.com/s/00u6cnj9mthvzj1/stratified_train_idx.txt?dl=1',
                        'extension':'.txt'},
                'valid_idx':{'url':'https://www.dropbox.com/s/420uhjvjab1bt7k/stratified_valid_idx.txt?dl=1',
                        'extension':'.txt'}},


    'HIGGS': { 'data_archive':{'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz',
                    'extension':'.csv.gz'},
                'train_idx':{'url':'https://www.dropbox.com/s/i2uekmwqnp9r4ix/stratified_train_idx.txt?dl=1',
                        'extension':'.txt'},
                'valid_idx':{'url':'https://www.dropbox.com/s/wkbk74orytmb2su/stratified_valid_idx.txt?dl=1',
                        'extension':'.txt'}},

    'MICROSOFT': {'train':{'url':'https://www.dropbox.com/s/izpty5feug57kqn/msrank_train.tsv?dl=1',
                    'extension':'.tsv'},
                'test' :{'url': 'https://www.dropbox.com/s/tlsmm9a6krv0215/msrank_test.tsv?dl=1',
                    'extension':'.tsv'},
                'train_idx':{'url':'https://www.dropbox.com/s/pba6dyibyogep46/train_idx.txt?dl=1',
                        'extension':'.txt'},
                'valid_idx':{'url':'https://www.dropbox.com/s/yednqu9edgdd2l1/valid_idx.txt?dl=1',
                        'extension':'.txt'}},

    'YAHOO':{'train_archive':{'url':'https://www.dropbox.com/s/7rq3ki5vtxm6gzx/yahoo_set_1_train.gz?dl=1',
                    'extension':'.gz'},
            'valid_archive' :{'url': 'https://www.dropbox.com/s/3ai8rxm1v0l5sd1/yahoo_set_1_validation.gz?dl=1',
                    'extension':'.gz'},
            'test_archive':{'url':'https://www.dropbox.com/s/3d7tdfb1an0b6i4/yahoo_set_1_test.gz?dl=1',
                        'extension':'.gz'}},
    
    'CLICK':{'data_csv':{'url':'https://www.dropbox.com/s/w43ylgrl331svqc/click.csv?dl=1',
                    'extension':'.csv'}},   
    
    'SHELTER':{'train':{'file_id':'15fHBRQM5ztc1LUiAhPiYTOggi5lyw4C-', 'extension':'.csv'},
               'test' :{'file_id':'1I0QqJOUM_lhAdcUZnmninHEqi68nMeju', 'extension':'.csv'}} ,
    
    'ADULT':{'adult'  :{'file_id':'1_gsKRhFKae2JIfkGjeKNPifI-5LAt_TS', 'extension':'.csv'},
             'archive':{'extension':'.zip'}}                                             
    }


def download(url, file_name, delete_if_error=True, chunk_size=4096):
    if os.path.exists(file_name):
            print("file exist!")
            return file_name
    try:
        with open(file_name, "wb") as f:
            print("Downloading {} > {}".format(url, file_name))
            response = requests.get(url, stream=True)
            content_len = response.headers.get('content-length')

            if content_len is None:  # no content length header
                f.write(response.content)
            else:
                content_len = int(content_len)
                with tqdm(total=content_len) as progressbar:
                    for d in response.iter_content(chunk_size=chunk_size):
                        if d:  # filter-out keep-alive chunks
                            f.write(d)
                            progressbar.update(len(d))
    except Exception as e:
        if delete_if_error:
            print("Removing incomplete download {}.".format(file_name))
            os.remove(file_name)
        raise e
    return file_name

'''
download data corrsponding to the map we defined.
'''
def download_data(data_set):
    data_folder = os.path.join(os.getcwd(),"dataset")
    print("---downloading:" + data_set+"---")
    if data_set in dataset_url:
        path = os.path.join(data_folder, data_set)
        info = dataset_url[data_set]
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        for name in info:
            f_path = os.path.join(path, name+info[name]['extension'])
            download(info[name]['url'],f_path)


def get_file_path(data_set, name):
    if data_set not in dataset_url or name not in dataset_url[data_set]:
        return print('Unkown file!')
    data_folder = os.path.join(os.getcwd(),"dataset")
    path =  os.path.join(data_folder, data_set) 
    file_name = name + dataset_url[data_set][name]['extension']
    path = os.path.join(path, file_name) 
    return path

'''
Read the data set form files and split the dataset.
'''
def data_processing(data_set,random_state, train_size=None, valid_size=None, test_size=None):
    tr_size,val_size,t_size = train_size,valid_size,test_size
    if data_set not in dataset_url:
        return print('Unkown file!')
   
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    random.seed(random_state)
    
    def tr_val_idx():
        if all(sizes is None for sizes in (train_size, valid_size, test_size)):
            tr_idx = pd.read_csv(get_file_path(data_set,'train_idx'), header=None)[0].values
            val_idx = pd.read_csv(get_file_path(data_set,'valid_idx'), header=None)[0].values
        else:
            assert tr_size, "please provide either train_size or none of sizes"
            if val_size is None:
                val_size = len(X_train) - tr_size
                assert val_size > 0
            if tr_size + val_size > len(X_train):
                print('train_size + valid_size = {} > dataset size: {}.'.format(tr_size + val_size, len(X_train)))
                    
            shuffled_indices = np.random.permutation(np.arange(len(X_train)))
            tr_idx = shuffled_indices[:tr_size]
            val_idx = shuffled_indices[tr_size: tr_size + val_size]
        return tr_idx, val_idx

    def A9A():
        tr_path,t_path = get_file_path(data_set,'train'),get_file_path(data_set,'test')
        if not all(os.path.exists(f) for f in (tr_path, t_path)):
            download(data_set)

        X_train, y_train = load_svmlight_file(get_file_path(data_set,'train'), dtype=np.float32, n_features=123)
        X_test, y_test = load_svmlight_file(get_file_path(data_set,'test'), dtype=np.float32, n_features=123)
        X_train, X_test = X_train.toarray(), X_test.toarray()
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0
        y_train, y_test = y_train.astype(np.int), y_test.astype(np.int)
        train_idx, valid_idx = tr_val_idx()
        return dict(
        X_train=X_train[train_idx], y_train=y_train[train_idx],
        X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test) 
        
        
    def YEAR():
        if not os.path.exists(get_file_path(data_set,'data')):
            download_data(data_set)

        test_size=51630
        n_features = 91
        types = {i: (np.float32 if i != 0 else np.int) for i in range(n_features)}
        data = pd.read_csv(get_file_path(data_set,'data'), header=None, dtype=types)
        data_train, data_test = data.iloc[:-test_size], data.iloc[-test_size:]

        X_train , y_train = data_train.iloc[:, 1:].values, data_train.iloc[:, 0].values
        X_test, y_test = data_test.iloc[:, 1:].values, data_test.iloc[:, 0].values
        train_idx, valid_idx = tr_val_idx() 
        return dict(
        X_train=X_train[train_idx], y_train=y_train[train_idx],
        X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test)

    def EPSILON():
        path = os.path.join('dataset',dataset)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        tr_path = get_file_path(data_set,'train')
        t_path  = get_file_path(data_set,'test')

     
        tr_archive, t_archive = get_file_path(data_set,'train_archive'),get_file_path(data_set,'test_archive')
        
        if not all(os.path.exists(f) for f in (tr_path, t_path)):

            if not all(os.path.exists(fa) for fa in (tr_archive, t_archive)):
                download_data(data_set)
                
            for fn, archive_name in zip((tr_path, t_path), (tr_archive, t_archive)):
                zipfile = bz2.BZ2File(archive_name)
                with open(fn, 'wb') as f:
                    f.write(zipfile.read())
                print('finish unzip')
        print("reading dataset (it may take a long time)")
        X_train, y_train = load_svmlight_file(tr_path, dtype=np.float32, n_features=2000)
        X_test, y_test = load_svmlight_file(t_path, dtype=np.float32, n_features=2000)
        X_train, X_test = X_train.toarray(), X_test.toarray()
        y_train, y_test = y_train.astype(np.int), y_test.astype(np.int)
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0

        train_idx, valid_idx = tr_val_idx() 

        return dict(
        X_train=X_train[train_idx], y_train=y_train[train_idx],
        X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test
        )


    def PROTEIN():
        tr_path = get_file_path(data_set,'train')
        t_path  = get_file_path(data_set,'test')

        if not all(os.path.exists(f) for f in (tr_path, t_path)):
            download_data(data_set)
        
        X_train, y_train = load_svmlight_file(tr_path, dtype=np.float32, n_features=357)
        X_test, y_test = load_svmlight_file(t_path, dtype=np.float32, n_features=357)
        X_train, X_test = X_train.toarray(), X_test.toarray()
        y_train, y_test = y_train.astype(np.int), y_test.astype(np.int)

        train_idx, valid_idx = tr_val_idx() 
        return dict(
            X_train=X_train[train_idx], y_train=y_train[train_idx],
            X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
            X_test=X_test, y_test=y_test
        )

    def CLICK():
        valid_size=100_000
        validation_seed=None
        csv_path = get_file_path(data_set,'data_csv')
        if not os.path.exists(csv_path):
            download_data(data_set)

        data = pd.read_csv(csv_path, index_col=0)
        X, y = data.drop(columns=['target']), data['target']
        X_train, X_test = X[:-100_000].copy(), X[-100_000:].copy()
        y_train, y_test = y[:-100_000].copy(), y[-100_000:].copy()

        y_train = (y_train.values.reshape(-1) == 1).astype('int64')
        y_test = (y_test.values.reshape(-1) == 1).astype('int64')

        cat_features = ['url_hash', 'ad_id', 'advertiser_id', 'query_id',
                    'keyword_id', 'title_id', 'description_id', 'user_id']

        X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=valid_size, random_state=validation_seed)

        cat_encoder = LeaveOneOutEncoder()
        cat_encoder.fit(X_train[cat_features], y_train)
        X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
        X_val[cat_features] = cat_encoder.transform(X_val[cat_features])
        X_test[cat_features] = cat_encoder.transform(X_test[cat_features])
        return dict(
            X_train=X_train.values.astype('float32'), y_train=y_train,
            X_valid=X_val.values.astype('float32'), y_valid=y_val,
            X_test=X_test.values.astype('float32'), y_test=y_test
        )

    def HIGGS():
        test_size=5 * 10 ** 5
        path = os.path.join(os.getcwd(),"dataset/HIGGS")
        data_path  = os.path.join(path, 'higgs.csv')
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        data_archive = get_file_path(data_set,'data_archive')
        
        if not os.path.exists(data_path):
            if not os.path.exists(data_archive):
                download_data(data_set)
           
            with gzip.open(data_archive, 'rb') as f_in:
                with open(data_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        n_features = 29
        types = {i: (np.float32 if i != 0 else np.int) for i in range(n_features)}
        data = pd.read_csv(data_path, header=None, dtype=types)
        train, test = data.iloc[:-test_size], data.iloc[-test_size:]

        X_train, y_train = train.iloc[:, 1:].values, train.iloc[:, 0].values
        X_test,  y_test = test.iloc[:, 1:].values, test.iloc[:, 0].values

        train_idx, valid_idx = tr_val_idx()
        
        return dict(
            X_train=X_train[train_idx], y_train=y_train[train_idx],
            X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
            X_test=X_test, y_test=y_test,
        )

    def MICROSOFT():
        for fname in (get_file_path(data_set,'train'), get_file_path(data_set,'test')):
            raw = open(fname).read().replace('\\t', '\t')
            with open(fname, 'w') as f:
                f.write(raw)

        train = pd.read_csv(get_file_path(data_set,'train'), header=None, skiprows=1, sep='\t')
        test = pd.read_csv(get_file_path(data_set,'test'), header=None, skiprows=1, sep='\t')
      
        train_idx, valid_idx = tr_val_idx()

        X_train, y_train, query_train = train.iloc[train_idx, 2:].values, train.iloc[train_idx, 0].values, train.iloc[train_idx, 1].values
        X_valid, y_valid, query_valid = train.iloc[valid_idx, 2:].values, train.iloc[valid_idx, 0].values, train.iloc[valid_idx, 1].values
        X_test, y_test, query_test = test.iloc[:, 2:].values, test.iloc[:, 0].values, test.iloc[:, 1].values

        return dict(
        X_train=X_train.astype(np.float32), y_train=y_train.astype(np.int64), query_train=query_train,
        X_valid=X_valid.astype(np.float32), y_valid=y_valid.astype(np.int64), query_valid=query_valid,
        X_test=X_test.astype(np.float32), y_test=y_test.astype(np.int64), query_test=query_test,
        )

    def YAHOO():

        tr_path,val_path,t_path = get_file_path(data_set,'train'),get_file_path(data_set,'valid'),get_file_path(data_set,test)
        tr_archive, val_archive, t_archive = get_file_path(data_set,'train_archive'),get_file_path(data_set,'valid_archive'),get_file_path(data_set,'test_archive')
        
        if not all(os.path.exists(f) for f in (tr_path, val_path, t_path)):
            if not all(os.path.exists(fa) for fa in (tr_archive, val_archive, t_archive)):
                download_data(data_set)
                
            for file_name, archive_name in zip((tr_path, val_path, t_path), (tr_archive, val_archive, t_archive)):
                with gzip.open(archive_name, 'rb') as f_in:
                    with open(file_name, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
 
        for fname in (tr_path, val_path, t_path):
            raw = open(fname).read().replace('\\t', '\t')
            with open(fname, 'w') as f:
                f.write(raw)

        train = pd.read_csv(tr_path, header=None, skiprows=1, sep='\t')
        valid = pd.read_csv(val_path, header=None, skiprows=1, sep='\t')
        test = pd.read_csv(t_path, header=None, skiprows=1, sep='\t')
        X_train, y_train, query_train = train.iloc[:, 2:].values, train.iloc[:, 0].values, train.iloc[:, 1].values
        X_valid, y_valid, query_valid = valid.iloc[:, 2:].values, valid.iloc[:, 0].values, valid.iloc[:, 1].values
        X_test, y_test, query_test    = test.iloc[:, 2:].values, test.iloc[:, 0].values, test.iloc[:, 1].values

        return dict(
            X_train=X_train.astype(np.float32), y_train=y_train, query_train=query_train,
            X_valid=X_valid.astype(np.float32), y_valid=y_valid, query_valid=query_valid,
            X_test=X_test.astype(np.float32), y_test=y_test, query_test=query_test,
            )

    '''
    new data set we added for evaluation
    # based on: https://www.kaggle.com/c/shelter-animal-outcomes/data
    #https://jovian.ml/aakanksha-ns/shelter-outcome
    '''
    def SHELTER():
        path = os.path.join('dataset','SHELTER')
        tr_path , t_path = get_file_path(data_set,'train'),get_file_path(data_set,'test')
        
        if not all(os.path.exists(fn) for fn in (tr_path, t_path)):
            os.makedirs(path, exist_ok=True)

            tr_archive = os.path.join(path, 'train.zip')
            t_archive = os.path.join(path, 'test.zip')

            if not all(os.path.exists(f) for f in (tr_archive, t_archive)):
                gdd.download_file_from_google_drive(file_id = dataset_url[data_set]['train']['file_id'],
                                    dest_path=tr_archive,
                                    unzip=True)
                gdd.download_file_from_google_drive(file_id = dataset_url[data_set]['test']['file_id'],
                                    dest_path=t_archive,
                                    unzip=True)
    
        train = pd.read_csv(tr_path)
        test = pd.read_csv(t_path)

        train_X = train.drop(columns= ['OutcomeType', 'OutcomeSubtype', 'AnimalID'])
        Y = train['OutcomeType']
        test_X = test
        stacked_df = train_X.append(test_X.drop(columns=['ID']))
        stacked_df = stacked_df.drop(columns=['DateTime'])
        for col in stacked_df.columns:
            if stacked_df[col].isnull().sum() > 10000:
                #print("dropping", col, stacked_df[col].isnull().sum())
                stacked_df = stacked_df.drop(columns = [col])

        for col in stacked_df.columns:
            if stacked_df.dtypes[col] == "object":
                stacked_df[col] = stacked_df[col].fillna("NA")
            else:
                stacked_df[col] = stacked_df[col].fillna(0)
            stacked_df[col] = LabelEncoder().fit_transform(stacked_df[col])

        for col in stacked_df.columns:
            stacked_df[col] = stacked_df[col].astype('category')


        X = stacked_df[0:26729]
        test_processed = stacked_df[26729:]

        Y = LabelEncoder().fit_transform(Y)

        #sanity check to see numbers match and matching with previous counter to create target dictionary
        #print(Counter(train['OutcomeType']))
        #print(Counter(Y))
        target_dict = {
            'Return_to_owner' : 3,
            'Euthanasia': 2,
            'Adoption': 0,
            'Transfer': 4,
            'Died': 1
        }
        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.33, random_state=0)

        return dict(
            X_train=X_train.values.astype(np.float32), y_train=y_train,
            X_valid=X_val.values.astype(np.float32) , y_valid=y_val,
            X_test=test_processed.values.astype(np.float32), y_test=np.zeros(len(test_processed))
        )
    '''
    New data set for testing
    '''
    def ADULT():
        tr_path = get_file_path(data_set,'adult')

        if not all(os.path.exists(fname) for fname in (tr_path)):
            #os.makedirs(path, exist_ok=True)
            train_archive_path = get_file_path(data_set,'archive')

            gdd.download_file_from_google_drive(file_id= dataset_url[data_set]['adult']['file_id'],
                                    dest_path=train_archive_path,
                                    unzip=True)
    
        df = pd.read_csv(tr_path)

        labels = df.pop('<=50K')

        X_train, X_test = df[:26049].copy(), df[26049:].copy()
        y_train, y_test = labels[:26049].copy(), labels[26049:].copy()

        X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                    y_train,
                                                    test_size=0.2)

        class_to_int = {c: i for i, c in enumerate(y_train.unique())}                                                                                                               
        y_train_int = [class_to_int[v] for v in y_train]                                                                                                                            
        y_val_int = [class_to_int[v] for v in y_val] 
        y_test_int = [class_to_int[v] for v in y_test]
        cat_features = ['workclass', 'education', 'marital-status',
                        'occupation', 'relationship', 'race', 'sex',
                        'native-country']
    
        cat_encoder = LeaveOneOutEncoder()
        cat_encoder.fit(X_train[cat_features], y_train_int)
        X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
        X_val[cat_features] = cat_encoder.transform(X_val[cat_features])
        X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

        # Node is going to want to have the values as float32 at some points
        X_train = X_train.values.astype('float32')
        X_val = X_val.values.astype('float32')
        X_test = X_test.values.astype('float32')
        y_train = np.array(y_train_int).astype('float32')
        y_val = np.array(y_val_int).astype('float32')
        y_test = np.array(y_test_int).astype('float32')
        
        return dict(
            X_train=X_train, y_train=y_train,
            X_valid=X_val, y_valid=y_val,
            X_test=X_test, y_test=y_test,
        )

    f_map = {'A9A': A9A,
            'EPSILON': EPSILON,
            'PROTEIN': PROTEIN,
            'YEAR': YEAR,
            'HIGGS': HIGGS,
            'MICROSOFT': MICROSOFT,
            'YAHOO': YAHOO,
            'CLICK': CLICK,
            'SHELTER':SHELTER,
            'ADULT': ADULT
        }

    return f_map[data_set]()