import torch
from modeling import build_model
from dataset import ImageDataset
from torch.utils.data import DataLoader

from distance import Mahalanobis_distance
from uploader import upload_multiple_files,upload_to_db

def dataloader_collate_fn(batch):
    imgs, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), img_paths

def top_k_similarity(k,distance_mat):
    H, W = distance_mat.shape
    oneD_distance_mat = distance_mat.flatten()  # if we are sure then use 1d_distance_mat.view(-1) : kind of inplace
    top_k_value, oneD_indices = oneD_distance_mat.topk(k,largest=False) # largest = False ---> Top min : "distance" ---> "similarity"
    twoD_indices = torch.cat(((oneD_indices // W).unsqueeze(1), (oneD_indices % W).unsqueeze(1)), dim=1)
    print(f"twoD_indices.shape : {twoD_indices.shape} ")
    return   top_k_value,twoD_indices


def main():

    # load cfg
    import yaml
    with open("config.yaml") as cfg_file:
        cfg =  yaml.load(cfg_file, Loader=yaml.FullLoader)

    # data loader 
    print("start DataLoader")
    print(f"dataset dir: {cfg['datasets']['dir_path']}")
    dataset = ImageDataset(cfg)
    data_loader = DataLoader(
        dataset, batch_size=cfg["dataloader"]["batch_size"], shuffle=False, 
        num_workers=cfg["dataloader"]["num_workers"],collate_fn=dataloader_collate_fn)
    print(dataset.dataset)
    
    # feature extractor model
    print("start initialize model")
    model = build_model(cfg) 
    model.load_param(cfg["model"]["weight"])
    model = model.cuda(0) if torch.cuda.device_count() >= 1 else img_batch
    model.eval()
    print("start processing feature bank")
    feature_bank = {} # {img_path:feature}
    with torch.no_grad():
        for img_batch,img_path_batch  in data_loader:
            img_batch = img_batch.cuda(0) if torch.cuda.device_count() >= 1 else img_batch
            feature_batch = model(img_batch)
            feature_bank.update({img_path:feature for img_path,feature in zip(img_path_batch,feature_batch)})
    print("complete processing feature bank")
    # backup "feature_bank"
    torch.save(feature_bank, f"{cfg['log']['log_dir']}/{cfg['log']['feature_bank']}")
    print(f"feature bank has been saved to {cfg['log']['log_dir']}/{cfg['log']['feature_bank']}")

    print("start calculating distace/simlarity")
    distance_mat = Mahalanobis_distance(torch.stack(list(feature_bank.values())))
    torch.save(distance_mat,f"{cfg['log']['log_dir']}/{cfg['log']['distance_mat']}")
    print(f"distance_mat has been saved to {cfg['log']['log_dir']}/{cfg['log']['distance_mat']}")
    
    # db:
    print("start uploading process")
    img_paths =  list(feature_bank.keys())
    if not cfg["storage"]["local"]:  upload_multiple_files(img_paths,cfg["storage"]["bucket"])
    initial_rank = torch.argsort(distance_mat,dim=1)[:,:cfg['budget']['k']+1].cpu().numpy()
    storage =  'local'  if cfg["storage"]["local"] else fg["storage"]["bucket"]
    upload_to_db(cfg["db"],storage,cfg["budget"]["k"],initial_rank,img_paths)
    print("complete uploading process")



if __name__ == '__main__':
    main()


