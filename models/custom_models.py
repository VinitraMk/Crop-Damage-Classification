from models.resnet18 import Resnet18
from models.alexnet import Alexnet

def get_model(model_name = 'resnet18'):
    model = {}
    if model_name == "resnet18":
        model = Resnet18(5)
    elif model_name == "alexnet":
        model = Alexnet(5)
    else:
        raise SystemExit("Error: no valid model name passed! Check run.yaml")
    return model
