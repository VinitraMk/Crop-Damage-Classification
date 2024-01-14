import matplotlib.pyplot as plt
from common.utils import get_exp_params

class Visualization:

    def __init__(self, model_info, model_history = {}):
        self.bestm_valacc = model_info['results']['valacc']
        self.bestm_valloss = model_info['results']['valloss']
        self.bestm_trloss = model_info['results']['trloss']
        self.bestm_tlh = model_info['results']['trlosshistory']
        self.bestm_vlh = model_info['results']['vallosshistory']
        self.bestm_vah = model_info['results']['valacchistory']
        self.exp_params = get_exp_params()
        self.model_history = model_history if model_history != {} else None

    def __plot_loss_history(self):
        num_epochs = self.exp_params["train"]["num_epochs"]
        plt.clf()
        print("\nBest model results\n\n")
        plt.plot(list(range(num_epochs)), self.bestm_tlh, color="red", label="Best model training loss history")
        plt.plot(list(range(num_epochs)), self.bestm_vlh, color="orange", label="Best model validation loss history")
        plt.title("Best model loss history")
        plt.legend()
        plt.show()
        print('\n\n\n')
        plt.plot(list(range(num_epochs)), self.bestm_vah, label = "Best model validation accuracy history")
        plt.title("Best model validation accuracy history")
        plt.legend()
        plt.show()
        print("\nBest Model Training Loss:", self.bestm_trloss)
        print("Best Model Validation Loss:", self.bestm_valloss)
        print(f"Best Model Accuracy: {self.bestm_valacc}\n")

    def __get_performance_metrics(self):
        k = self.exp_params['train']['k']
        if self.model_history != None:
            avg_valloss = 0.0
            avg_valacc = 0.0
            avg_trloss = 0.0
            print("\nFold\tTraining Loss\tValidation Loss\tValidation Accuracy")
            for fi in self.model_history:
                print(f"{fi}\t{self.model_history[fi]['trloss']}\t{self.model_history[fi]['valloss']}\t{self.model_history[fi]['valacc']}")
                avg_valacc += self.model_history[fi]['valacc'].item()
                avg_valloss += self.model_history[fi]['valloss']
                avg_trloss += self.model_history[fi]['trloss']
            avg_valacc/=k
            avg_valloss/=k
            avg_trloss/=k
            print("\nAverage Performance Metrics of Model")
            print("Average Training Loss", avg_trloss)
            print("Average Validation Loss", avg_valloss)
            print("Average Validation Accuracy", avg_valacc)

    def get_results(self):
        self.__plot_loss_history()
        self.__get_performance_metrics()

