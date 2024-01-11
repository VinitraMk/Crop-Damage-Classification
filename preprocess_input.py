from argparse import ArgumentParser
from torchvision.transforms import Compose
from transforms.transforms import Resize, CenterCrop
from preprocess.preprocessor import Preprocessor
from common.utils import init_config, get_config, save2config

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--resize_dim", type = int, required = True)
    parser.add_argument("--crop_dim", type = int, required = True)
    args = parser.parse_args()
    
    init_config()
    cfg = get_config()
    save2config('X_key', 'image')
    save2config('y_key', 'label')
    print(cfg)
    
    data_transform = Compose([Resize(args.resize_dim), CenterCrop(args.crop_dim)])
    preop = Preprocessor()
    preop.transform_input(data_transform)    
    