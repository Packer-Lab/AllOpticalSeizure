import json
import pickle
from funcsforprajay import funcs as pj

A = pj.load_pkl('home/pshah/Documents/code/temp_A.pkl')

print(A.data)

# Read JSON file
with open('data.json') as data_file:
    data_loaded = json.load(data_file)

# %% sqlite3
from _analysis_.trial1 import load


obj2 = load("MyClass_key")
