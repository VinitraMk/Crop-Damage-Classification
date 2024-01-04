import torchvision

def get_model(model_name = "resnet18", get_weights = False):
    model = {}
    if model_name.lower() == "resnet50":
        if get_weights:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            model = torchvision.models.resnet50(weights=weights)
        else:
            model = torchvision.models.resnet50()
        model_id2name = weights.meta["categories"]
    elif model_name.lower() == "resnet18":
        if get_weights:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V2
            model = torchvision.models.resnet18(weights=weights)
        else:
            model = torchvision.models.resnet18()
        model_id2name = weights.meta["categories"]
    elif model_name.lower() == "inception":
        if get_weights:
            weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V2
            model = torchvision.models.inception_v3(weights=weights)
        else:
            model = torchvision.models.inception_v3()
        model_id2name = weights.meta["categories"]
    
    return model, model_id2name