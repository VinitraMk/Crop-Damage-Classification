from common.utils import get_exp_params
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import os
from random import shuffle
import torch.nn.functional as F
import pandas as pd
from torchvision.transforms import Normalize, Compose
import time
from tqdm import tqdm
import warnings

from common.utils import get_accuracy, get_config, save_experiment_output, save_experiment_chkpt, load_modelpt, image_collate
from models.custom_models import get_model
from preprocess.preprocessor import Preprocessor

class Experiment:

    def __get_optimizer(self, model, model_params, optimizer_name = 'Adam'):
        if optimizer_name == 'Adam':
            return torch.optim.Adam(model.parameters(), lr = model_params['lr'], weight_decay = model_params['weight_decay'], amsgrad = model_params['amsgrad'])
        elif optimizer_name == 'SGD':
            return torch.optim.SGD(model.parameters(), lr = model_params['lr'], weight_decay = model_params['weight_decay'], momentum = model_params['momentum'], nesterov= True)
        else:
            raise SystemExit("Error: no valid optimizer name passed! Check run.yaml file")


    def __init__(self, model_name, ftr_dataset, data_transforms, all_folds_metrics):
        self.exp_params = get_exp_params()
        self.model_name = model_name
        self.ftr_dataset = ftr_dataset
        cfg = get_config()
        self.root_dir = cfg["root_dir"]
        self.device = cfg['device']
        self.all_folds_res = {}
        self.all_folds_metrics = all_folds_metrics
        self.metrics = {}
        self.data_transforms = data_transforms

    def __loss_fn(self, loss_name = 'cross-entropy'):
        if loss_name == 'cross-entropy':
            return torch.nn.CrossEntropyLoss()
        elif loss_name == 'mse':
            return torch.nn.MSELoss()
        else:
            raise SystemExit("Error: no valid loss function name passed! Check run.yaml")

    def __conduct_training(self, model, fold_idx, fold_si, epoch_index,
                           train_loader, val_loader,
                           train_len, val_len,
                           trlosshistory = [], vallosshistory = [], valacchistory = []):
        loss_fn = self.__loss_fn()
        num_epochs = self.exp_params['train']['num_epochs']
        epoch_ivl = self.exp_params['train']['epoch_interval']
        tr_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0
        model_info = {}
        epoch_arr = list(range(epoch_index, num_epochs))
        data_transforms = Compose(self.data_transforms)
        normalize = Normalize(self.metrics['mean'], self.metrics['std0'])
        warnings.filterwarnings("ignore")
        
        for i in epoch_arr:
            print(f'\tRunning Epoch {i}')
            model.train()
            tr_loss = 0.0
            #print(f'\t\tRunning through training dataset')
            sf = time.time()
            for batch_idx, batch in enumerate(tqdm(train_loader, desc = '\t\tRunning through training set', position = 0, leave = True)):
                self.optimizer.zero_grad()
                img_batch = list(map(data_transforms, batch[1]))
                img_batch = np.stack(img_batch, 0)
                img_batch = normalize(torch.from_numpy(img_batch)).to(self.device)
                img_target = torch.from_numpy(batch[2]).to(self.device)
                op = model(img_batch)
                loss = loss_fn(op, img_target)
                loss.backward()
                self.optimizer.step()
                tr_loss += (loss.item() * img_batch.size()[0])
                torch.cuda.empty_cache()
                del batch

            tr_loss /= train_len
            trlosshistory.append(tr_loss)

            #print('\t\tRunning through validation set')
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc = '\t\tRunning through validation set', position = 0, leave = True)):
                    img_batch = list(map(data_transforms, batch[1]))
                    img_batch = np.stack(img_batch, 0)
                    img_batch = normalize(torch.from_numpy(img_batch)).to(self.device)
                    img_target = torch.from_numpy(batch[2]).to(self.device)
                    lop = model(img_batch)
                    loss = loss_fn(lop, img_target)
                    val_loss += (loss.item() * img_target.size()[0])
                    lop_lbls = torch.argmax(lop, 1)
                    val_acc += (get_accuracy(lop_lbls, img_target) * img_target.size()[0])
                    torch.cuda.empty_cache()
                    del batch

            val_loss /= val_len
            val_acc /= val_len
            vallosshistory.append(val_loss)
            valacchistory.append(val_acc)
            if (i+1) % epoch_ivl == 0:
                print(f'\tEpoch {i} Training Loss: {tr_loss}')
                print(f"\tEpoch {i} Validation Loss: {val_loss}")
                print(f"\tEpoch {i} Validation Accuracy: {val_acc}\n")
            model_info = {
                'valloss': val_loss,
                'valacc': val_acc,
                'trloss': tr_loss,
                'trlosshistory': torch.tensor(trlosshistory),
                'vallosshistory': torch.tensor(vallosshistory),
                'valacchistory': torch.tensor(valacchistory),
                'fold': fold_si,
                'epoch': i
            }
            self.save_model_checkpoint(model.state_dict(), self.optimizer.state_dict(),
            model_info, self.all_folds_res, 'last_state')


        model_info = {
            'valloss': val_loss,
            'valacc': val_acc,
            'trloss': tr_loss,
            'trlosshistory': torch.tensor(trlosshistory),
            'vallosshistory': torch.tensor(vallosshistory),
            'valacchistory': torch.tensor(valacchistory),
            'fold': fold_si,
            'epoch': -1
        }
        self.all_folds_res[fold_idx] = model_info
        self.save_model_checkpoint(model.state_dict(), self.optimizer.state_dict(),
        model_info, self.all_folds_res, 'last_state')
        return model, model_info

    def __get_experiment_chkpt(self, model):
        mpath = os.path.join(self.root_dir, "models/checkpoints/current_model.pt")
        if os.path.exists(mpath):
            print("Loading saved model")
            saved_model = load_modelpt(mpath)
            model_dict = saved_model["model_state"]
            model.load_state_dict(model_dict) 
            self.all_folds_res = saved_model["model_history"]
            return model, saved_model["last_state"], saved_model["best_state"], saved_model["optimizer_state"]
        else:
            return model, None, None, None

    def train(self, model_type = "best_model"):
        train_loader = {}
        val_loader = {}
        
        if self.exp_params['train']['val_split_method'] == 'k-fold':
            model = get_model(self.model_name)
            model = model.to(self.device)
            model, ls, bs, ops = self.__get_experiment_chkpt(model)

            k = self.exp_params['train']['k']
            fl = len(self.ftr_dataset)
            fr = list(range(fl))
            vlen = fl // k

            #get last model state if it exists
            if ls == None:
                epoch_index = 0
                val_eei = list(range(0, fl, vlen))
                trlosshistory = []
                vallosshistory = []
                valacchistory = []
                self.optimizer = self.__get_optimizer(model, self.exp_params['model'], self.exp_params['model']['optimizer'])
            elif ls['epoch'] == -1:
                si = ls['fold'] + vlen
                epoch_index = 0
                val_eei = list(range(si, fl, vlen))
                trlosshistory = []
                vallosshistory = []
                valacchistory = []
                model = get_model(self.model_name)
                model = model.to(self.device)
                self.optimizer = self.__get_optimizer(model, self.exp_params['model'], self.exp_params['model']['optimizer'])
            else:
                si = ls['fold'] + vlen
                epoch_index = ls['epoch'] + 1
                val_eei = list(range(si, fl, vlen))
                trlosshistory = ls['trlosshistory'].tolist()
                vallosshistory = ls['vallosshistory'].tolist()
                valacchistory = ls['valacchistory'].tolist()
                self.optimizer = self.__get_optimizer(model, self.exp_params['model'], self.exp_params['model']['optimizer'])
                self.optimizer.load_state_dict(ops)
            #get best model state if it exists
            bestm_valacc = 0.0 if bs == None else bs['valacc']
            bestm_valloss = 99999 if bs == None else bs['valloss']
            bestm_trloss = 0.0 if bs == None else 0.0
            bestm_tlh = torch.zeros(self.exp_params['train']['num_epochs']) if bs == None else bs['trlosshistory']
            bestm_vlh = torch.zeros(self.exp_params['train']['num_epochs']) if bs == None else bs['vallosshistory']
            bestm_vah = torch.zeros(self.exp_params['train']['num_epochs']) if bs == None else bs['valacchistory']
            best_model = {}
            if bs != None:
                best_model = get_model()
                bmd = bs['model_state']
                bms = best_model.state_dict()
                for key in bmd:
                    bms[key] = bmd[key]
            best_fold = 0

            for vi, si in enumerate(val_eei):
                ei = si + vlen
                print(f"Running split {vi} starting at {si} and ending with {ei}")
                val_idxs = fr[si:ei]
                tr_idxs = [fi for fi in fr if fi not in val_idxs]
                train_dataset = Subset(self.ftr_dataset, tr_idxs)
                val_dataset = Subset(self.ftr_dataset, val_idxs)
                tr_len = len(tr_idxs)
                val_len = len(val_idxs)
                self.metrics = self.all_folds_metrics[vi]

                train_loader = DataLoader(train_dataset,
                    batch_size = self.exp_params['train']['batch_size'],
                    shuffle = False,
                    collate_fn = image_collate
                )
                val_loader = DataLoader(val_dataset,
                    batch_size = self.exp_params['train']['batch_size'],
                    shuffle = False,
                    collate_fn = image_collate
                )

                if ls != None:
                    model, model_info = self.__conduct_training(model, vi, si, epoch_index,
                        train_loader, val_loader,
                        tr_len, val_len,
                        trlosshistory, vallosshistory, valacchistory)
                else:
                    model, model_info = self.__conduct_training(model, vi, si, epoch_index,
                        train_loader, val_loader, tr_len, val_len,
                        trlosshistory, vallosshistory, valacchistory)
                
                self.all_folds_res[vi] = model_info
                if model_info["valloss"] < bestm_valloss:
                    best_model = model
                    bestm_valloss = model_info["valloss"]
                    bestm_valacc = model_info["valacc"]
                    bestm_trloss = model_info["trloss"]
                    bestm_vlh = model_info["vallosshistory"]
                    bestm_tlh = model_info["trlosshistory"]
                    bestm_vah = model_info["valacchistory"]
                    best_fold = vi
                trlosshistory = []
                valacchistory = []
                vallosshistory = []

                model_info = {
                    'model_state': best_model.state_dict(),
                    'valloss': bestm_valloss,
                    'trloss': bestm_trloss,
                    'valacc': bestm_valacc,
                    'trlosshistory': bestm_tlh,
                    'vallosshistory': bestm_vlh,
                    'valacchistory': bestm_vah,
                    'fold': best_fold,
                    'epoch': -1,
                }
                self.save_model_checkpoint(model.state_dict(), self.optimizer.state_dict(), model_info,
                self.all_folds_res, 'best_state')
                model = get_model(self.model_name)
                model = model.to(self.device)
                self.optimizer = self.__get_optimizer(model, self.exp_params['model'], self.exp_params['model']['optimizer'])


            self.save_model_checkpoint(best_model.state_dict(), None, model_info, None)
            return self.all_folds_res
        elif self.exp_params['train']['val_split_method'] == 'fixed-split':
            model = get_model(self.model_name)
            model = model.to(self.device)
            model, ls, bs = self.__get_experiment_chkpt(model)
            self.optimizer = self.__get_optimizer(model, self.exp_params['model'], self.exp_params['model']['optimizer'])
            preop = Preprocessor()
            
            print("Running straight split")
            epoch_index = 0 if ls == None else ls['epoch'] + 1
            trlosshistory, vallosshistory, valacchistory = [] if ls == None else ls['trlosshistory'].tolist(), ls['vallosshistory'].tolist(), ls['valacchistory'].tolist()
            vp = self.exp_params['train']['val_percentage'] / 100
            fl = len(self.ftr_dataset)
            vlen = int(vp * fl)
            fr = list(range(fl))
            if self.exp_params['shuffle_data']:
                shuffle(fr)
            val_idxs = fr[:vlen]
            tr_idxs = fr[vlen:]
            train_dataset = Subset(self.ftr_dataset, tr_idxs)
            val_dataset = Subset(self.ftr_dataset, val_idxs)
            tr_len = len(tr_idxs)
            val_len = len(val_idxs)
            self.metrics = all_folds_metrics = preop.get_dataset_metrics(train_dataset, 'fixed-split')

            train_loader = DataLoader(train_dataset,
                batch_size = self.exp_params['train']['batch_size'],
                shuffle = False,
                collate_fn = image_collate
            )
            val_loader = DataLoader(val_dataset,
                batch_size = self.exp_params['train']['batch_size'],
                shuffle = False,
                collate_fn = image_collate
            )

            if ls != None:
                model, model_info = self.__conduct_training(model, -1, -1, epoch_index,
                    train_loader, val_loader,
                    tr_len, val_len,
                    trlosshistory, vallosshistory, valacchistory)
            else:
                model, model_info = self.__conduct_training(model, -1, -1, epoch_index,
                    train_loader, val_loader, tr_len, val_len,
                    trlosshistory, vallosshistory, valacchistory)
            model_info['fold'] = 0
            self.save_model_checkpoint(model.state_dict(), None, model_info, None)
            return {}
        else:
            raise SystemExit("Error: no valid split method passed! Check run.yaml")

    def save_model_checkpoint(self, model_state, optimizer_state, chkpt_info,
    model_history = None, chkpt_type = 'last_state'):
        if model_history == None:
            save_experiment_output(model_state, chkpt_info, self.exp_params,
                True, False)
            os.remove(os.path.join(self.root_dir, "models/checkpoints/current_model.pt"))
        else:
            save_experiment_chkpt(model_state, optimizer_state, chkpt_info, model_history, chkpt_type)


