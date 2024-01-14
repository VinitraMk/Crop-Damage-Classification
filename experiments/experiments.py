from common.utils import get_exp_params
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import os
from random import shuffle
import torch.nn.functional as F
import pandas as pd

from common.utils import get_accuracy, get_config, save_experiment_output, save_experiment_chkpt, load_modelpt
from models.custom_models import get_model


class Experiment:

    def __get_optimizer(self, model, model_params, optimizer_name = 'Adam'):
        if optimizer_name == 'Adam':
            return torch.optim.Adam(model.parameters(), lr = model_params['lr'], weight_decay = model_params['weight_decay'], amsgrad = model_params['amsgrad'])
        elif optimizer_name == 'SGD':
            return torch.optim.SGD(model.parameters(), lr = model_params['lr'], weight_decay = model_params['weight_decay'], momentum = model_params['momentum'], nesterov= True)
        else:
            raise SystemExit("Error: no valid optimizer name passed! Check run.yaml file")


    def __init__(self, model_name, ftr_dataset):
        self.exp_params = get_exp_params()
        self.model_name = model_name
        self.ftr_dataset = ftr_dataset
        cfg = get_config()
        self.X_key = cfg['X_key']
        self.y_key = cfg['y_key']
        self.root_dir = cfg["root_dir"]
        self.output_dir = cfg['output_dir']
        self.device = 'cuda' if cfg['use_gpu'] else 'cpu'
        self.all_folds_res = {}

    def __loss_fn(self, loss_name = 'cross-entropy'):
        if loss_name == 'cross-entropy':
            return torch.nn.CrossEntropyLoss()
        elif loss_name == 'mse':
            return torch.nn.MSELoss()
        else:
            raise SystemExit("Error: no valid loss function name passed! Check run.yaml")

    def __conduct_training(self, model, fold_idx, epoch_index,
                           train_loader, val_loader,
                           train_len, val_len,
                           trlosshistory = [], vallosshistory = [], valacchistory = []):
        loss_fn = self.__loss_fn()
        num_epochs = self.exp_params['train']['num_epochs']
        epoch_ivl = self.exp_params['train']['epoch_interval']
        batch_ivl = self.exp_params['train']['batch_interval']
        tr_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0
        model_info = {}
        epoch_arr = list(range(epoch_index, num_epochs))

        for i in epoch_arr:
            print(f'\tRunning Epoch {i}')
            model.train()
            tr_loss = 0.0
            print(f'\t\tRunning through training dataset')
            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                batch[self.X_key] = batch[self.X_key].float().to(self.device)
                batch[self.y_key] = batch[self.y_key].to(self.device)
                op = model(batch[self.X_key])
                loss = loss_fn(op, batch[self.y_key])
                loss.backward()
                self.optimizer.step()
                tr_loss += (loss.item() * batch[self.X_key].size()[0])
                torch.cuda.empty_cache() 
            tr_loss /= train_len
            trlosshistory.append(tr_loss)

            print('\t\tRunning through validation set')
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    batch[self.X_key] = batch[self.X_key].float().to(self.device)
                    batch[self.y_key] = batch[self.y_key].to(self.device)
                    lop = model(batch[self.X_key])
                    loss = loss_fn(lop, batch[self.y_key])
                    val_loss += (loss.item() * batch[self.y_key].size()[0])
                    lop_lbls = torch.argmax(lop, 1)
                    val_acc += (get_accuracy(lop_lbls, batch[self.y_key]) * batch[self.y_key].size()[0])
                    torch.cuda.empty_cache()
            val_loss /= val_len
            val_acc /= val_len
            vallosshistory.append(val_loss)
            valacchistory.append(val_acc)
            if (i+1) % epoch_ivl == 0:
                print(f'Epoch {i} Training Loss: {tr_loss}')
                print(f"Epoch {i} Validation Loss: {val_loss}")
                print(f"Epoch {i} Validation Accuracy: {val_acc}\n")
            model_info = {
                'valloss': val_loss,
                'valacc': val_acc,
                'trloss': tr_loss,
                'trlosshistory': torch.tensor(trlosshistory),
                'vallosshistory': torch.tensor(vallosshistory),
                'valacchistory': torch.tensor(valacchistory),
                'fold': fold_idx,
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
            'fold': fold_idx,
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
            self.optimizer = self.__get_optimizer(model, self.exp_params['model'], self.exp_params['model']['optimizer'])
            ops = saved_model['optimizer_state']
            self.optimizer.load_state_dict(ops)
            return model, saved_model["last_state"], saved_model["best_state"]
        else:
            self.optimizer = self.__get_optimizer(model, self.exp_params['model'], self.exp_params['model']['optimizer'])
            return model, None, None

    def train(self):
        train_loader = {}
        val_loader = {}
        model = get_model(self.model_name)
        model = model.to(self.device)
        model, ls, bs = self.__get_experiment_chkpt(model)

        if self.exp_params['train']['val_split_method'] == 'k-fold':
            k = self.exp_params['train']['k']
            fl = len(self.ftr_dataset)
            fr = list(range(fl))
            shuffle(fr)
            vlen = fl // k

            #get last model state if it exists
            if ls == None:
                vset_ei = fl // k
                epoch_index = 0
                val_eei = list(range(vset_ei, fl, vlen))
                si = 0
            elif ls['epoch'] == -1:
                si = ls['fold'] + vlen
                vset_ei = ls['fold'] + (2 * vlen)
                epoch_index = 0
                val_eei = list(range(vset_ei, fl, vlen))
                if val_eei[-1] == None or val_eei[-1] + vlen < fl:
                    val_eei.append(val_eei[-1] + vlen)
            else:
                si = ls['fold']
                vset_ei = ls['fold'] + vlen
                epoch_index = ls['epoch'] + 1
                val_eei = list(range(vset_ei, fl, vlen))
                if val_eei[-1] == None or val_eei[-1] + vlen >= fl:
                    val_eei.append(val_eei[-1] + vlen)

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
            best_fold = vset_ei

            for vi, ei in enumerate(val_eei):
                print(f"Running split {vi}")
                val_idxs = fr[si:ei]
                tr_idxs = [fi for fi in fr if fi not in val_idxs]
                train_dataset = Subset(self.ftr_dataset, tr_idxs)
                val_dataset = Subset(self.ftr_dataset, val_idxs)
                tr_len = len(tr_idxs)
                val_len = len(tr_idxs)

                train_loader = DataLoader(train_dataset,
                    batch_size = self.exp_params['train']['batch_size'],
                    shuffle = False
                )
                val_loader = DataLoader(val_dataset,
                    batch_size = self.exp_params['train']['batch_size'],
                    shuffle = False
                )
                if ls != None:
                    model, model_info = self.__conduct_training(model, si, epoch_index,
                        train_loader, val_loader,
                        tr_len, val_len,
                        ls['trlosshistory'].tolist(), ls['vallosshistory'].tolist(), ls['valacchistory'].tolist())
                else:
                    model, model_info = self.__conduct_training(model, si, epoch_index,
                        train_loader, val_loader, tr_len, val_len)
                self.all_folds_res[si] = model_info
                si = ei
                if model_info["valloss"] < bestm_valloss:
                    best_model = model
                    bestm_valloss = model_info["valloss"]
                    bestm_valacc = model_info["valacc"]
                    bestm_trloss = model_info["trloss"]
                    bestm_vlh = model_info["vallosshistory"]
                    bestm_tlh = model_info["trlosshistory"]
                    bestm_vah = model_info["valacchistory"]
                    best_fold = model_info["fold"]
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

            del model_info['fold']
            del model_info['epoch']
            self.save_model_checkpoint(best_model.state_dict(), None, model_info, None)
            return self.all_folds_res
        elif self.exp_params['train']['val_split_method'] == 'fix-split':
            print("Running straight split")
            epoch_index = 0 if ls == None else ls['epoch'] + 1
            vp = self.exp_params['train']['val_percentage'] / 100
            fl = len(self.ftr_dataset)
            vlen = int(vp * fl)
            fr = list(range(fl))
            shuffle(fr)
            val_idxs = fr[:vlen]
            tr_idxs = fr[vlen:]
            train_dataset = Subset(self.ftr_dataset, tr_idxs)
            val_dataset = Subset(self.ftr_dataset, val_idxs)
            tr_len = len(tr_idxs)
            val_len = len(val_idxs)
            train_loader = DataLoader(train_dataset,
                batch_size = self.exp_params['train']['batch_size'],
                shuffle = self.exp_params['train']['shuffle_data']
            )
            val_loader = DataLoader(val_dataset,
                batch_size = self.exp_params['train']['batch_size'],
                shuffle = self.exp_params['train']['shuffle_data']
            )
            if ls != None:
                model, model_info = self.__conduct_training(model, -1, epoch_index,
                    train_loader, val_loader,
                    tr_len, val_len,
                    ls['trlosshistory'].tolist(), ls['vallosshistory'].tolist(), ls['valacchistory'].tolist())
            else:
                model, model_info = self.__conduct_training(model, -1, epoch_index,
                    train_loader, val_loader, tr_len, val_len)
            del model_info['fold']
            del model_info['epoch']
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

