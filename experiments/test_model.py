#script for testing the model

from common.utils import get_exp_params, get_accuracy, get_config, get_model_filename, save_experiment_output
from torch.utils.data import DataLoader, Subset
import torch
from matplotlib import pyplot as plt
import os
import pandas as pd
import torch.nn.functional as F
from torchvision.transforms import Normalize

class ModelTester:

    def __init__(self, model, te_dataset, metrics):
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
        self.device = cfg['device']
        self.metrics = metrics

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

    def test_and_save_csv(self, lbl_dict, plot_sample_results = False):
        self.model = self.model.to(self.device)
        test_loader = DataLoader(self.te_dataset,
            batch_size = self.exp_params["train"]["batch_size"], shuffle = False)
        self.model.eval()
        running_loss = 0.0
        acc = 0.0
        num2class = lambda x: lbl_dict[x.item()]
        sub_lbls = ['ID', 'DR', 'G', 'ND', 'WD', 'other']
        rpath = os.path.join(self.output_dir, "results.csv")
        cbi = 0
        bsize = self.exp_params['train']['batch_size']
        if os.path.exists(rpath):
            results_df = pd.read_csv(rpath)
            no_rows = len(results_df)
            if no_rows % bsize == 0:
                cbi = (no_rows // bsize) - 1
                cbin = cbi * bsize
            else:
                cbi = no_rows // bsize
                cbin = cbi * bsize
            cbir = list(range(cbin, no_rows))
            results_df = results_df.drop(labels = cbir, axis = 0)
        else:
            results_df = pd.DataFrame([], columns = sub_lbls)
            cbi = 0

        normalize = Normalize(self.metrics['mean'], self.metrics['std0'])
        
        print("Running through test dataset")
        with torch.no_grad():
            for bi, batch in enumerate(test_loader):
                print(f"\tRunning through batch {bi}")
                if bi >= cbi:
                    batch[self.X_key] = batch[self.X_key].float().to(self.device)
                    op = F.softmax(self.model(normalize(batch[self.X_key].float())))
                    oplbls = torch.argmax(op, 1)
                    classlbls = list(map(num2class, oplbls))
                    res = [[id] + preds for id,preds in zip(batch['id'], op.tolist())]
                    batch_df = pd.DataFrame(res, columns = sub_lbls)
                    results_df = pd.concat([results_df, batch_df], 0)
                    results_df.to_csv(rpath, index = False)
                    del batch[self.X_key]
                    del batch[self.y_key]
                else:
                    pass
        
        