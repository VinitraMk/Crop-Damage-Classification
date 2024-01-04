from torchvision.io import read_image
from PIL import Image
import torch
import os
import yaml

root_dir = ''
config_params = {}

def save_df_to_csv(df, filename, columns = []):
    if len(columns) == 0:
        columns = df.columns.tolist()
    df.to_csv(filename, index = False)
    
def read_imgtensor(imgpath):
    return read_image(imgpath)

def read_imgnp(imgpath):
    img = Image.open(imgpath)
    return img

def img2tensor(imgnparr):
    return torch.from_numpy(imgnparr)
   
def join_path(path_a, path_b):
    return os.path.join(path_a, path_b)

def read_yaml(ypath):
    yml_params = {}
    with open(ypath, "r") as stream:
        try:
            yml_params = yaml.safe_load(stream)
        except yaml.YAMLError as err:
            print(err)
    return yml_params

def dump_yaml(ypath, datadict):
    with open(ypath, 'w') as outfile:
        yaml.dump(datadict, outfile, default_flow_style=False)

def init_config():
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, 'data')
    config_path = os.path.join(root_dir, 'config.yaml')
    config_params = read_yaml(config_path)
    config_params['root_dir'] = root_dir
    config_params['data_dir'] = data_dir
    dump_yaml(config_path, config_params)

def get_exp_params():
    yaml_fp = os.path.join(root_dir, 'run.yaml')
    exp_params = read_yaml(yaml_fp)
    return exp_params

def get_config():
    config_path = os.path.join(root_dir, 'config.yaml')
    config_params = read_yaml(config_path)
    return config_params

def save2config(key, val):
    config_params[key] = val
    config_path = os.path.join(root_dir, 'config.yaml')
    dump_yaml(config_path, config_params)

def get_accuracy(y_pred, y_true):
    c = torch.sum(y_pred == y_true)
    return c / len(y_true)

def get_error(y_pred, y_true):
    c = torch.sum(y_pred != y_true)
    return c / len(y_true)

def save_model_chkpt(chkpt_info, chkpt_filename, best_model = False):
    if best_model:
        fpath = os.path.join(config_params['root_dir'], 'models/best-models')
    else:
        fpath = os.path.join(config_params['root_dir'], 'models/chkpoints')
    
    fpath = os.path.join(fpath, chkpt_filename)
    with open(fpath, 'w') as fp:
        torch.save(
            chkpt_info,
            fp
        )
        
def load_model(model_path):
    return torch.load(model_path)