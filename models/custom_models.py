from models.resnet18 import Resnet18

def get_model(model_name = 'resnet18'):
    model = {}
    if model_name == "resnet18":
        model = Resnet18(5)
    else:
        raise SystemExit("Error: no valid model name passed! Check run.yaml")
    return model
