import matplotlib.pyplot as plt
from common.utils import get_exp_params

class Visualization:
    
    def __init__(self, model_info):
        self.bestm_acc = model_info['best_model_acc']
        self.bestm_loss = model_info['best_model_loss']
        self.bestm_tlh = model_info['best_model_trlosshistory']
        self.bestm_vlh = model_info['best_model_vallosshistory']
        self.lastm_acc = model_info['last_model_acc']
        self.lastm_loss = model_info['last_model_loss']
        self.lastm_tlh = model_info['last_model_trlosshistory']
        self.lastm_vlh = model_info['last_model_vallosshistory']
        self.exp_params = get_exp_params()
        
    def __plot_loss_history(self):
        num_epochs = self.exp_params["train"]["num_epochs"]
        plt.clf()
        print("\nBest model results\n\n")
        plt.plot(list(range(num_epochs)), self.bestm_tlh, color="red", label="Best model training loss history")
        plt.plot(list(range(num_epochs)), self.bestm_vlh, color="orange", label="Best model valid loss history")
        plt.title("Best model loss history")
        plt.legend()
        plt.show()
        print("\nBest loss:", self.bestm_loss)
        print(f"Best Accuracy: {self.bestm_acc}\n")
        print("\nLast model results\n\n")
        plt.plot(list(range(num_epochs)), self.lastm_tlh, color="red", label="Last model training loss history")
        plt.plot(list(range(num_epochs)), self.lastm_vlh, color="orange", label="Last model valid loss history")
        plt.title("Last model loss history")
        plt.legend()
        plt.show()
        print("\nLast loss:", self.bestm_loss)
        print(f"Last Accuracy: {self.bestm_acc}\n\n")
        
    def get_results(self):
        self.__plot_loss_history()

        