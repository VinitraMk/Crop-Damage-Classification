from common.utils import get_exp_params
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from common.utils import get_accuracy, get_config
import pandas as pd
import os
import torch.nn.functional as F

class Experiment:

    def __get_optimizer(self, model, model_params, optimizer_name = 'Adam'):
        if optimizer_name == 'Adam':
            return torch.optim.Adam(model.parameters(), lr=model_params['lr'], weight_decay = model_params['weight_decay'], amsgrad = model_params['amsgrad'])
        elif optimizer_name == 'SGD':
            return torch.optim.SGD(model.parameters(), lr=model_params['lr'], weight_decay = model_params['weight_decay'], momentum = model_params['momentum'], nesterov= True)
        else:
            raise SystemExit("Error: no valid optimizer name passed! Check run.yaml file")


    def __init__(self, model, fr_train_dataset):
        self.exp_params = get_exp_params()
        self.model = model
        self.optimizer = self.__get_optimizer(self.model, self.exp_params['model'], self.exp_params['model']['optimizer'])
        self.fr_train_dataset = fr_train_dataset
        cfg = get_config()
        self.X_key = cfg['X_key']
        self.y_key = cfg['y_key']
        self.device = "cuda" if cfg['use_gpu'] else "cpu"

    def __loss_fn(self, loss_name = 'cross-entropy'):
        if loss_name == 'cross-entropy':
            return torch.nn.CrossEntropyLoss()
        else:
            raise SystemExit("Error: no valid loss function name passed! Check run.yaml")

    def __conduct_training(self, train_loader, val_loader):
        loss_fn = self.__loss_fn()
        tr_batch_num = len(train_loader)
        val_batch_num = len(val_loader)
        num_epochs = self.exp_params['train']['num_epochs']
        epoch_ivl = self.exp_params['train']['epoch_interval']
        batch_ivl = self.exp_params['train']['batch_interval']
        best_loss = 99999
        best_model = {}
        best_model_trlosshistory = []
        best_model_vallosshistory = []
        tr_loss_history = []
        val_loss_history = []
        tr_loss = 0.0
        for i in range(num_epochs):
            print(f'\tRunning Epoch {i}')
            self.model.train()
            running_loss = 0.0
            print(f'\t\tRunning through training dataset')
            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                batch[self.X_key] = batch[self.X_key].float().to(self.device)
                batch[self.y_key] = batch[self.y_key].to(self.device)
                op = self.model(batch[self.X_key])
                loss = loss_fn(op, batch[self.y_key])
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if (batch_idx + 1) % batch_ivl == 0:
                    print(f'\t\tBatch {batch_idx + 1} Loss: {running_loss / (batch_idx + 1)}')
            tr_loss = running_loss / tr_batch_num
            tr_loss_history.append(tr_loss)

            print('\t\tRunning through validation set')
            self.model.eval()
            val_loss = 0.0
            val_acc = 0
            for batch_idx, batch in enumerate(val_loader):
                batch[self.X_key] = batch[self.X_key].float().to(self.device)
                batch[self.y_key] = batch[self.y_key].to(self.device)
                lop = self.model(batch[self.X_key])
                loss = loss_fn(lop, batch[self.y_key])
                lop_lbls = torch.argmax(lop, 1)
                loss.backward()
                val_loss += loss.item()
                val_acc += get_accuracy(lop_lbls, batch[self.y_key])

                if (batch_idx + 1) % batch_ivl == 0:
                    print(f'\t\tBatch {batch_idx + 1} Last Model Loss: {val_loss / (batch_idx + 1)}')
                    print(f'\t\tBatch {batch_idx + 1} Best Model Loss: {val_loss / (batch_idx + 1)}')
            val_loss /= val_batch_num
            val_acc /= val_batch_num
            val_loss_history.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = self.model
                best_acc = val_acc
                best_model_trlosshistory = tr_loss_history
                best_model_vallosshistory = val_loss_history
            if (i+1) % epoch_ivl == 0:
                print(f'Epoch {i} Training Loss: {tr_loss}')
                print(f"Epoch {i} Validation Loss: {val_loss}")
                print(f"Epoch {i} Validation Accuracy: {val_acc}\n")


        model_info = {
            'best_model': best_model if best_model != {} else {},
            'best_model_valloss': best_loss,
            'best_model_valacc': best_acc,
            'best_model_trlosshistory': torch.tensor(best_model_trlosshistory),
            'best_model_vallosshistory': torch.tensor(best_model_vallosshistory),
            'last_model': self.model,
            'last_model_valloss': val_loss,
            'last_model_valacc': val_acc,
            'last_model_trlosshistory': torch.tensor(tr_loss_history),
            'last_model_vallosshistory': torch.tensor(val_loss_history)
        }

        return model_info


    def train(self):
        train_loader = {}
        val_loader = {}
        self.model = self.model.to(self.device)
        if self.exp_params['train']['val_split_method'] == 'k-fold':
            k = self.exp_params['train']['k']
            vp = self.exp_params['train']['val_percentage']
            fl = len(self.fr_train_dataset)
            fr = list(range(fl))
            vlen = int(vp * fl)
            vset_len = fl // k
            val_eei = list(range(vset_len, fl, vset_len))
            si = 0
            bestm_acc = 0.0
            lastm_acc = 0.0
            lastm_loss = 0.0
            bestm_loss = 0.0
            bestm_tlh = torch.zeros(self.exp_params['train']['num_epochs'])
            bestm_vlh = torch.zeros(self.exp_params['train']['num_epochs'])
            lastm_tlh = torch.zeros(self.exp_params['train']['num_epochs'])
            lastm_vlh = torch.zeros(self.exp_params['train']['num_epochs'])

            for vi, ei in enumerate(val_eei):
                print(f"Running split {vi}")
                val_idxs = fr[si:ei]
                tr_idxs = fr[ei:]
                si = ei
                train_dataset = Subset(self.fr_train_dataset, tr_idxs)
                val_dataset = Subset(self.fr_train_dataset, val_idxs)

                train_loader = DataLoader(train_dataset,
                batch_size = self.exp_params['train']['batch_size'],
                shuffle = self.exp_params['train']['shuffle_data']
                )
                val_loader = DataLoader(val_dataset,
                    batch_size = self.exp_params['train']['batch_size'],
                    shuffle = self.exp_params['train']['shuffle_data']
                )
                model_info = self.__conduct_training(train_loader, val_loader)
                bestm_acc += model_info['best_model_valacc']
                bestm_loss += model_info['best_model_valloss']
                bestm_tlh += model_info['best_model_trlosshistory']
                bestm_vlh += model_info['best_model_vallosshistory']
                lastm_acc += model_info['last_model_valacc']
                lastm_loss += model_info['last_model_valloss']
                lastm_tlh += model_info['last_model_trlosshistory']
                lastm_vlh += model_info['last_model_vallosshistory']
            bestm_loss/=k
            bestm_acc/=k
            bestm_tlh/=k
            bestm_vlh/=k
            lastm_loss/=k
            lastm_acc/=k
            lastm_tlh/=k
            lastm_vlh/=k
            model_info['best_model_valacc'] = bestm_acc
            model_info['best_model_valloss'] = bestm_loss
            model_info['best_model_trlosshistory'] = bestm_tlh
            model_info['best_model_vallosshistory'] = bestm_vlh
            model_info['last_model_valacc'] = lastm_acc
            model_info['last_model_valloss'] = lastm_loss
            model_info['last_model_trlosshistory'] = lastm_tlh
            model_info['last_model_vallosshistory'] = lastm_vlh
            return model_info
        elif self.exp_params['train']['val_split_method'] == 'fix-split':
            print("Running straight split")
            vp = self.exp_params['train']['val_percentage']
            vlen = int(vp * len(self.fr_train_dataset))
            val_idxs = np.random.randint(0, len(self.fr_train_dataset), vlen).tolist()
            tr_idxs = [idx not in val_idxs for idx in range(len(self.fr_train_dataset))]
            train_dataset = Subset(self.fr_train_dataset, tr_idxs)
            val_dataset = Subset(self.fr_train_dataset, val_idxs)

            train_loader = DataLoader(train_dataset,
                batch_size = self.exp_params['train']['batch_size'],
                shuffle = self.exp_params['train']['shuffle_data']
            )
            val_loader = DataLoader(val_dataset,
                batch_size = self.exp_params['train']['batch_size'],
                shuffle = self.exp_params['train']['shuffle_data']
            )
            model_info = self.__conduct_training(train_loader, val_loader)
            return model_info
        else:
            raise SystemExit("Error: no valid split method passed! Check run.yaml")

    def test(self, model, test_dataset, lbl_dict):
        model = model.cpu()
        test_loader = DataLoader(test_dataset, batch_size = self.exp_params["train"]["batch_size"], shuffle = True)
        model.eval()
        loss_fn = self.__loss_fn(self.exp_params["train"]["loss"])
        running_loss = 0.0
        acc = 0
        num2class = lambda x: lbl_dict[x.item()]
        sub_lbls = ['ID', 'DR', 'G', 'ND', 'WD', 'other']
        results_df = pd.DataFrame([], columns = sub_lbls)
        print("Running through test dataset")
        with torch.no_grad():
            for bi, batch in enumerate(test_loader):
                op = F.softmax(model(batch[self.X_key].float()), 1)
                #get labels
                '''
                oplbls = torch.argmax(op, 1)
                classlbls = list(map(num2class, oplbls))
                '''
                res = [[id] + preds for id,preds in zip(batch['id'], op.tolist())]
                batch_df = pd.DataFrame(res, columns = sub_lbls)
                results_df = pd.concat([results_df, batch_df], 0)
                del batch[self.X_key]
                del batch[self.y_key]
        results_df.to_csv(os.path.join(self.output_dir, "results.csv"), index = False)

