import numpy as np
import pandas as pd
import torch
import os
import random
import numpy as np
import requests
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file

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
                    'extension':'.csv'}}
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


def data_processing(data_set,random_state, train_size=None, valid_size=None, test_size=None):
    tr_size,val_size,t_size = train_size,valid_size,test_size
    if data_set not in dataset_url:
        return print('Unkown file!')
   
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    random.seed(random_state)

    def process_A9A():
        X_train, y_train = load_svmlight_file(get_file_path(data_set,'train'), dtype=np.float32, n_features=123)
        X_test, y_test = load_svmlight_file(get_file_path(data_set,'test'), dtype=np.float32, n_features=123)
        X_train, X_test = X_train.toarray(), X_test.toarray()
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0
        y_train, y_test = y_train.astype(np.int), y_test.astype(np.int)

        if all(sizes is None for sizes in (train_size, valid_size, test_size)):
            train_idx = pd.read_csv(get_file_path(data_set,'train_idx'), header=None)[0].values
            valid_idx = pd.read_csv(get_file_path(data_set,'valid_idx'), header=None)[0].values
        else:
            assert tr_size, "please provide either train_size or none of sizes"
            if val_size is None:
                val_size = len(X_train) - tr_size
                assert val_size > 0
            if tr_size + val_size > len(X_train):
                print('train_size + valid_size = {} > dataset size: {}.'.format(
                    tr_size + val_size, len(X_train)))
                    
            shuffled_indices = np.random.permutation(np.arange(len(X_train)))
            train_idx = shuffled_indices[:tr_size]
            valid_idx = shuffled_indices[tr_size: tr_size + val_size]
        
        return dict(
        X_train=X_train[train_idx], y_train=y_train[train_idx],
        X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test)



    f_map = {'A9A': process_A9A,
        # 'EPSILON': process_EPSILON,
        # 'PROTEIN': process_PROTEIN,
        # 'YEAR': process_YEAR,
        # 'HIGGS': process_HIGGS,
        # 'MICROSOFT': process_MICROSOFT,
        # 'YAHOO': process_YAHOO,
        # 'CLICK': process_CLICK,}
        }

    return f_map[data_set]()