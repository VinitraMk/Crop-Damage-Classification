from torchvision.io import read_image
from PIL import Image
import torch
import os
import yaml
from datetime import datetime
import json

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

def read_json(json_path):
    json_data = {}
    with open(json_path, 'r') as fp:
        json_data = json.load(fp)
    return json_data

def dump_json(data, json_path):
    with open(json_path, 'w') as fp:
        json.dump(data, fp)

def init_config():
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, 'data')
    config_path = os.path.join(root_dir, 'config.yaml')
    config_params = read_yaml(config_path)
    config_params['root_dir'] = root_dir
    config_params['data_dir'] = data_dir
    config_params['output_dir'] = os.path.join(root_dir, "output")
    config_params['img_dir'] = os.path.join(root_dir, 'data/input/images')
    config_params['use_gpu'] = torch.cuda.is_available()
    config_params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    config_params = get_config()
    config_params[key] = val
    config_path = os.path.join(root_dir, 'config.yaml')
    dump_yaml(config_path, config_params)

def get_accuracy(y_pred, y_true):
    c = torch.sum(y_pred == y_true)
    return c / len(y_true)

def get_error(y_pred, y_true):
    c = torch.sum(y_pred != y_true)
    return c / len(y_true)

def save_model(model_state, chkpt_info, chkpt_filename = 'last_model', is_checkpoint = True, best_model = False):
    config_params = get_config()
    if is_checkpoint:
        fpath = os.path.join(config_params['root_dir'], 'models/checkpoints')
    else:
        if best_model:
            fpath = os.path.join(config_params['output_dir'], 'experiment_results/best_experiments')
        else:
            fpath = os.path.join(config_params['output_dir'], 'experiment_results/experiments')
    
    mpath = os.path.join(fpath, f'{chkpt_filename}.pt')
    jpath = os.path.join(fpath, f'{chkpt_filename}.json')
    torch.save(
        model_state,
        mpath
    )
    with open(jpath, 'w') as fp:
        json.dump(chkpt_info, fp)
        
def load_modelpt(model_path):
    config_params = get_config()
    return torch.load(model_path, map_location = torch.device(config_params["device"]))

def get_modelinfo(json_filename, is_chkpt = True, is_best = False):
    model_info = {}
    cfg = get_config()
    if is_chkpt:
        json_path = os.path.join(cfg["root_dir"], "models/checkpoints/last_model.json")
    else:
        if is_best:
            json_path = os.path.join(cfg["output"], f"experiment_results/best_experiments/{json_filename}.json")
        else:
            json_path = os.path.join(cfg["output"], f"experiment_results/experiments/{json_filename}.json")
    with open(json_path, 'r') as fp:
        model_info = json.load(fp)
    return model_info

def get_model_filename(model_name):
    now = datetime.now()
    nowstr = now.strftime("%d%m%Y%H%M%S")
    fname = f'{model_name}_{nowstr}'
    return fname

def save_experiment_output(model_state, chkpt_info, exp_params, is_chkpoint = True, save_as_best = False):
    model_info = {
        'experiment_params': exp_params,
        'results': {
            'valloss': chkpt_info['valloss'],
            'trloss': chkpt_info['trloss'],
            'valacc': chkpt_info['valacc'].item(),
            'trlosshistory': chkpt_info['trlosshistory'].tolist(),
            'vallosshistory': chkpt_info['vallosshistory'].tolist(),
            'valacchistory': chkpt_info['valacchistory'].tolist(),
            'epoch': -1,
            'fold': chkpt_info['fold']
        }
    }
    save_model(model_state, model_info,
        f'last_model', is_chkpoint, save_as_best)

def save_experiment_chkpt(model_state, optimizer_state, chkpt_info, model_history, chkpt_type = "last_state"):
    cfg = get_config()
    mpath = os.path.join(cfg["root_dir"], "models/checkpoints/current_model.pt")
    if os.path.exists(mpath):
        saved_model = load_modelpt(mpath)
    else:
        saved_model = {
            "model_complete": False,
            "model_history": model_history,
            "best_state": None,
            "last_state": None
        }
    saved_model[chkpt_type] = chkpt_info
    saved_model["model_state"] = model_state #always the last state
    saved_model["optimizer_state"] = optimizer_state #always the last state
    torch.save(saved_model, mpath)

def get_saved_model(model, model_filename = '', is_chkpt = True, is_best = False):
    cfg = get_config()
    if is_chkpt:
        model_dict = load_modelpt(os.path.join(cfg["root_dir"], "models/checkpoints/last_model.pt"))
    else:
        if is_best:
            model_dict = load_modelpt(os.path.join(cfg["output_dir"], f"experiment_results/best_experiments/{model_filename}.pt"))
        else:
            model_dict = load_modelpt(os.path.join(cfg["output_dir"], f"experiment_results/experiments/{model_filename}.pt"))
    model.load_state_dict(model_dict)
    return model

def save_model_helpers(model_history, optimizer_state, model_filename = '', is_chkpt = True, is_best = False):
    cfg = get_config()
    if is_chkpt:
        mhpath = os.path.join(cfg["root_dir"], "models/checkpoints/last_model_history.pt")
        opath = os.path.join(cfg["root_dir"], "models/checkpoints/last_model_optimizer.pt")
    else:
        if is_best:
            mhpath = os.path.join(cfg["root_dir"], f"experiment_results/best_experiments/{model_filename}_history.pt")
            opath = os.path.join(cfg["root_dir"], f"experiment_results/best_experiments/{model_filename}_optimizer.pt")
        else:
            mhpath = os.path.join(cfg["root_dir"], f"experiment_results/experiments/{model_filename}_history.pt")
            opath = os.path.join(cfg["root_dir"], f"experiment_results/experiments/{model_filename}_optimizer.pt")

    torch.save(model_history, mhpath)
    torch.save(optimizer_state, opath)

def get_model_data(mh_filename, is_chkpt = True, is_best = False):
    cfg = get_config()
    if is_chkpt:
        mpath = os.path.join(cfg["root_dir"], "models/checkpoints/last_model.pt")
        mhpath = os.path.join(cfg["root_dir"], "models/checkpoints/last_model_history.pt")
        opath = os.path.join(cfg["root_dir"], "models/checkpoints/last_model_optimizer.pt")
    else:
        if is_best:
            mpath = os.path.join(cfg["root_dir"], f"experiment_results/best_experiments/{mh_filename}.pt")
            mhpath = os.path.join(cfg["root_dir"], f"experiment_results/best_experiments/{mh_filename}_history.pt")
            opath = os.path.join(cfg["root_dir"], f"experiment_results/best_experiments/{mh_filename}_optimizer.pt")
        else:
            mpath = os.path.join(cfg["root_dir"], f"experiment_results/experiments/{mh_filename}.pt")
            mhpath = os.path.join(cfg["root_dir"], f"experiment_results/experiments/{mh_filename}_history.pt")
            opath = os.path.join(cfg["root_dir"], f"experiment_results/experiments/{mh_filename}_optimizer.pt")

    return torch.load(mpath, map_location = torch.device(cfg["device"])),
    torch.load(mhpath, map_location = torch.device(cfg["device"])),
    torch.load(opath, map_location = torch.device(cfg["device"]))

def convert_model2current(model, model_filename, is_chkpt = True, is_best = False):
    saved_model_state, saved_model_history, saved_model_optimizer = get_model_data(model_filename, is_chkpt, is_best)
    model_info = get_modelinfo(model_filename, is_chkpt, is_best)
    prev_ep = model_info['experiment_params']
    num_epochs = prev_ep['train']['num_epochs']

    cfg = get_config()
    mpath = os.path.join(cfg["root_dir"], "models/checkpoints/current_model.pt")
    model_state = {
        'trloss': model_info["results"]["trloss"],
        'valloss':  model_info["results"]["valloss"],
        'valacc': model_info["results"]["valacc"],
        'trlosshistory': model_info['trlosshistory'].tolist(),
        'vallosshistory': model_info['vallosshistory'].tolist(),
        'valacchistory': model_info['valacchistory'].tolist(),
        'epoch': num_epochs,
        'fold': 0
    }
    new_curr = {
        "model_complete": False,
        "model_history": saved_model_history,
        "best_state": model_state,
        "last_state": model_state
    }
    new_curr['model_state'] = saved_model_state
    new_curr['optimizer_state'] = saved_model_optimizer

    torch.save(new_curr, "current_model.pt")
    return new_curr


