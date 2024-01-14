#script for testing the model

from common.utils import get_exp_params, get_accuracy, get_config, save_model_chkpt, get_model_filename, save_experiment_output
from torch.utils.data import DataLoader, Subset
import torch
from matplotlib import pyplot as plt
from common import colorspaces
import os
import pandas as pd
import torch.nn.functional as F

class ModelTester:

    def __init__(self, model, te_dataset):
        cfg = get_config()
        self.te_dataset = te_dataset
        self.model = model.cpu()
        self.model.eval()
        self.exp_params = get_exp_params()
        self.te_loader = DataLoader(self.te_dataset,
            batch_size = self.exp_params['train']['batch_size'],
            shuffle = False
        )
        self.output_dir = cfg['output_dir']
        self.X_key = cfg['X_key']
        self.y_key = cfg['y_key']

    def __plot_results(self, predicted_labels, subset_len = 10):
        fr = list(range(subset_len))
        subset_dataset = Subset(self.te_dataset, fr)
        subset_loader = DataLoader(subset_dataset, batch_size = 1, shuffle = False)
        fl = len(subset_dataset)
        plt.clf()
        plt.figure(figsize = (subset_len, 1))
        for bi, batch in enumerate(subset_loader):
            img = batch[self.X_key][0]
            plt.subplot(1,10,bi+1).set_title(predicted_labels[bi])
            plt.imshow(batch[bi,:,:,:])
            plt.axis(False)
        plt.show()
        plt.savefig(os.path.join(self.output_dir, "sample_test_results.png"))

    def test_and_save_csv(self, model, lbl_dict, plot_sample_results = False):
        model = model.to(self.device)
        model.eval()
        loss_fn = self.__loss_fn(self.exp_params["train"]["loss"])
        running_loss = 0.0
        acc = 0
        num2class = lambda x: lbl_dict[x.item()]
        sub_lbls = ['ID', 'DR', 'G', 'ND', 'WD', 'other']
        res_file = os.pathth.join(self.output_dir, "results.csv")
        if not(os.path.exists(res_file)):
            results_df = pd.DataFrame([], columns = sub_lbls)
        else:
            results_df = pd.read_csv(res_file)
        classlbls = []
        print("Running through test dataset")
        with torch.no_grad():
            for bi, batch in enumerate(self.te_loader):
                print(f"\n\tRunning through batch {bi}")
                batch[self.X_key] = batch[self.X_key].float().to(self.device)
                op = F.softmax(model(batch[self.X_key].float()), 1)
                if self.device == "cuda":
                    batch[self.X_key] = batch[self.X_key].to("cpu")
                else:
                    del batch[self.X_key]
                # predicted labels
                oplbls = torch.argmax(op, 1)
                classlbls = list(map(num2class, oplbls))
                res = [[id] + preds for id,preds in zip(batch['id'], op.tolist())]
                batch_df = pd.DataFrame(res, columns = sub_lbls)
                results_df = pd.concat([results_df, batch_df], 0)
                print(f"\tFinished collecting results for batch {bi}")
                if plot_sample_results:
                    self.__plot_results(classlbls) 
                results_df.to_csv(os.path.join(self.output_dir, "results.csv"), index = False)
    