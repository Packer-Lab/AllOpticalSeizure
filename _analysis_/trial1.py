import pickle

import numpy as np
import os
from sqlitedict import SqliteDict
from funcsforprajay import funcs as pj

class A:
    data = []
    string = 'asdfa'

    @classmethod
    def append_data(cls):
        for i in range(10):
            cls.data.append(i)

    def __init__(self):
        pass

def save_pkl(obj, pkl_path: str):
    if os.path.exists(pj.return_parent_dir(pkl_path)):
        os.makedirs(pj.return_parent_dir(pkl_path), exist_ok=True)
        with open(pkl_path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"\- saved to {pkl_path} -- ")
    else:
        raise NotADirectoryError(f'parent directory of {pkl_path} cannot be reached.')


# %% sqlite3
class MyClass():
    data = np.random.random(100)
    string = 'asdkasdfad'


def save(key, value, cache_file="/home/pshah/Documents/temp.sqlite3"):
    try:
        with SqliteDict(cache_file) as mydict:
            mydict[key] = value  # Using dict[key] to store
            mydict.commit()  # Need to commit() to actually flush the data
    except Exception as ex:
        print("Error during storing data (Possibly unsupported):", ex)


def load(key, cache_file="/home/pshah/Documents/temp.sqlite3"):
    try:
        with SqliteDict(cache_file) as mydict:
            value = mydict[key]  # No need to use commit(), since we are only loading data!
        return value
    except Exception as ex:
        print("Error during loading data:", ex)

# %%



if __name__ == '__main__':
    a = A()
    print(a.data)

    a.append_data()

    print(a.data)

    save_pkl(obj=a, pkl_path='/home/pshah/Documents/code/temp_A.pkl')

    # import h5py
    # hf = h5py.File('/home/pshah/Documents/code/temp.h5', 'w')
    # hf.create_dataset('data', data=a.data)
    # hf.create_dataset('string', data=a.string)
    # hf.close()

    # with open('/home/pshah/Documents/code/temp.pkl', 'wb') as f:
    #     pickle.dump(a, f)

    # import json
    # with open('/home/pshah/Documents/code/temp.pkl', 'w') as f:
    #     json.dump(a.__dict__, f)


    # f = h5py.File('/home/pshah/Documents/code/temp.h5', 'r')
    #
    #
    # # %%
    # obj1 = MyClass
    # save("MyClass_key", obj1)
    #
    # obj2 = load("MyClass_key")

    # print(obj1.data, obj2.data)
    # print(isinstance(obj1, MyClass), isinstance(obj2, MyClass))




