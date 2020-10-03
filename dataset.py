import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import glob

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img
    
class ImageDataset(Dataset):
    def __init__(self, cfg):

        if cfg["datasets"]["nested_dir"] : 
            self.dataset = [] 
            for fdir in os.listdir(cfg["datasets"]["dir_path"]):
                self.dataset +=  glob.glob(os.path.join(cfg["datasets"]["dir_path"],fdir,'*.jpg'))
        else: self.dataset = glob.glob(os.path.join(cfg["datasets"]["dir_path"], '*.jpg'))
        self.transform =  T.Compose([T.Resize(cfg["input"]["img_size"]),T.ToTensor(),T.Normalize(mean=cfg["input"]["pixel_mean"], std=cfg["input"]["pixel_std"])])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, img_path  