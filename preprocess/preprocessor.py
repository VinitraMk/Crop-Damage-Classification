import os

class Preprocessor:
    image_dir = ''

    def __init__(self, root_dir, image_dir):
        self.root_dir = root_dir
        self.image_dir = image_dir

    def make_label_csv(self):
        ## This function is for implementing code that constructs a csv file
        ## listing labels of all images. The csv file will have 4 columsn - image file name, label (encoded),
        ## original label and full path
        pass

