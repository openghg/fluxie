import json

data_dir = '/project/paris/inverse_modelling/'

fileList = []

with open(filepath, "r") as f:
    json_data = json.load(f)
    
for model,file_dict in json_data.items():
    m0 = model.split("_")[0]
    for species, exp_name in file_dict.items():
        search_str = f"{data_dir}/{m0}/{species}/{m0}_{exp_name}_*.nc"
        filepaths = glob(search_str)
        if len(filepaths)==2:
            fileList += filepaths
        else:
            raise ValueError(f"{len(filepaths)} files found for {search_str}")
print(filepaths)