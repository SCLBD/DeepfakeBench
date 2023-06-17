import json

base_dir = './'

# celeb-df-v1
with open('dataset_json/Celeb-DF-v1.json') as f:
    data = json.load(f)

# celeb-df-v2
with open('dataset_json/Celeb-DF-v2.json') as f:
    data = json.load(f)

# df1.0
with open('dataset_json/DeeperForensics-1.0.json') as f:
    data = json.load(f)

# dfdc
with open('dataset_json/DFDC.json') as f:
    data = json.load(f)
    
# dfdcp
with open('dataset_json/DFDCP.json') as f:
    data = json.load(f)
    
# uadfv
with open('dataset_json/UADFV.json') as f:
    data = json.load(f)
    
# ff++, including df,f2f,fs,nt. Also include faceshifter.
with open('dataset_json/FaceForensics++.json') as f:
    data = json.load(f)
