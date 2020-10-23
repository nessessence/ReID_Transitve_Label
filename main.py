import torch
from modeling import build_model
from dataset import ImageDataset
from torch.utils.data import DataLoader

from distance import Mahalanobis_distance
from uploader import upload_multiple_files,upload_to_db

def dataloader_collate_fn(batch):
    imgs, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), img_paths

def process_featerbank(data_loader,dataset,model,cfg):
    if cfg["pre_computed"]["feature_bank"]:
        print(f"loading pre-computed feature_bank from {cfg['pre_computed']['feature_bank']} ")
        feature_bank = torch.load(cfg["pre_computed"]["feature_bank"])
        print("loading feature_bank complete")
        return feature_bank
    print("start processing feature bank")
    feature_bank = {} # {img_path:feature}
    batch_size = cfg["dataloader"]["batch_size"]
    print(batch_size,len(dataset))
    percent = 0
    with torch.no_grad():
        for i,(img_batch,img_path_batch)  in enumerate(data_loader):
            img_batch = img_batch.cuda(0) if torch.cuda.device_count() >= 1 else img_batch
            feature_batch = model(img_batch).half()
            if i % 10 == 0: 
                percent += 100*(10*batch_size)/(len(dataset))
                print(f" processing : {percent}% ")
            feature_bank.update({img_path:feature.half() for img_path,feature in zip(img_path_batch,feature_batch)})
            del feature_batch

    print("complete processing feature bank")
    # backup "feature_bank"
    torch.save(feature_bank, f"{cfg['log']['log_dir']}/{cfg['log']['feature_bank']}")
    return feature_bank 

def process_distancing(feature_bank,cfg):
    if cfg["pre_computed"]["distance_mat"]:
        print(f"loading pre-computed distance_mat from {cfg['pre_computed']['distance_mat']} ")
        distance_mat = torch.load(cfg["pre_computed"]["distance_mat"])
        print("loading distance_mat complete")
        return distance_mat
    print("start calculating distace/simlarity")
    distance_mat = Mahalanobis_distance(torch.stack(list(feature_bank.values())))
    torch.save(distance_mat,f"{cfg['log']['log_dir']}/{cfg['log']['distance_mat']}")
    print(f"distance_mat has been saved to {cfg['log']['log_dir']}/{cfg['log']['distance_mat']}")
    return distance_mat

def upload_to_storage(img_paths,cfg):
    print("start uploading to storage")
    if not cfg["storage"]["local"]:  upload_multiple_files(img_paths,cfg["storage"]["bucket"])

def process_argsort(distance_mat,cfg):
    import numpy as np
    if cfg["pre_computed"]["argsorted_distance"]:
        print(f"loading pre-computed argsorted_distance from {cfg['pre_computed']['argsorted_distance']} ")
        argsorted_distance = np.load(cfg["pre_computed"]["argsorted_distance"])
        print("loading argsorted_distance complete")
        return argsorted_distance
    print("start calculating argsort distance")
    argsorted_distance = np.argsort(distance_mat.cpu().numpy(),axis=1)[:,:cfg['budget']['k']+1]
    np.save(f"{cfg['log']['log_dir']}/{cfg['log']['argsorted_distance']}",argsorted_distance )
    print(f"argsorted_distance has been saved to {cfg['log']['log_dir']}/{cfg['log']['argsorted_distance']}")
    return argsorted_distance

def upload_to_database(argsorted_distance,img_paths,cfg):
    initial_rank = argsorted_distance[:,:cfg['budget']['k']+1]
    storage =  'local'  if cfg["storage"]["local"] else fg["storage"]["bucket"]
    upload_to_db(cfg["db"],storage,cfg["budget"]["k"],initial_rank,img_paths)
    print("complete updating db")

def main(cfg):

    # data loader 
    print("start DataLoader")
    print(f"dataset dir: {cfg['datasets']['dir_path']}")
    dataset = ImageDataset(cfg)
    data_loader = DataLoader(
        dataset, batch_size=cfg["dataloader"]["batch_size"], shuffle=False, 
        num_workers=cfg["dataloader"]["num_workers"],collate_fn=dataloader_collate_fn)
    print(f"There are {len(dataset)} images to be processed")
    
    # feature extractor model
    print("start initialize model")
    model = build_model(cfg) 
    model.load_param(cfg["model"]["weight"])
    model = model.cuda(0) if torch.cuda.device_count() >= 1 else img_batch
    model.eval()

    feature_bank = process_featerbank(data_loader,dataset,model,cfg)
    img_paths = list(feature_bank.keys())
    distance_mat = process_distancing(feature_bank,cfg)
    del feature_bank
    upload_to_storage(img_paths,cfg)
    argsorted_distance = process_argsort(distance_mat,cfg)
    del distance_mat
    upload_to_database(argsorted_distance,img_paths,cfg)
    del argsorted_distance
    print("processing has been completed")





    # if cfg["pre_computed"]["feature_bank"]:
    #     print(f"loading pre-computed feature_bank from {cfg['pre_computed']['feature_bank']} ")
    #     feature_bank = torch.load(cfg["pre_computed"]["feature_bank"])
    #     print("loading feature_bank complete")
    # else:
    #     print("start processing feature bank")
    #     feature_bank = {} # {img_path:feature}

    #     batch_size = cfg["dataloader"]["batch_size"]
    #     percent = 0
    #     with torch.no_grad():
    #         for i,(img_batch,img_path_batch)  in enumerate(data_loader):
    #             img_batch = img_batch.cuda(0) if torch.cuda.device_count() >= 1 else img_batch
    #             #print(img_batch.type())
    #             feature_batch = model(img_batch).half()
    #             if i % 10 == 0: 
    #                 percent += 100*(10*batch_size)/(len(dataset))
    #                 print(f" processing : {percent}% ")

    #             feature_bank.update({img_path:feature.half() for img_path,feature in zip(img_path_batch,feature_batch)})
    #             del feature_batch

    #     print("complete processing feature bank")
    #     # backup "feature_bank"
    #     torch.save(feature_bank, f"{cfg['log']['log_dir']}/{cfg['log']['feature_bank']}")
    #     print(f"feature bank has been saved to {cfg['log']['log_dir']}/{cfg['log']['feature_bank']}")

    # print("start calculating distace/simlarity")
    # distance_mat = Mahalanobis_distance(torch.stack(list(feature_bank.values())))
    # torch.save(distance_mat,f"{cfg['log']['log_dir']}/{cfg['log']['distance_mat']}")
    # print(f"distance_mat has been saved to {cfg['log']['log_dir']}/{cfg['log']['distance_mat']}")
    
    # # db:
    # print("start uploading process")
    # img_paths =  list(feature_bank.keys())
    # del feature_bank
    # if not cfg["storage"]["local"]:  upload_multiple_files(img_paths,cfg["storage"]["bucket"])
    # # initial_rank = torch.argsort(distance_mat,dim=1)[:,:cfg['budget']['k']+1].cpu().numpy()
    # import numpy as np
    # argsorted_distance = np.argsort(distance_mat.cpu().numpy(),axis=1)[:,:cfg['budget']['k']+1]
    # np.save(f"{cfg['log']['log_dir']}/{cfg['log']['argsorted_distance']}",argsorted_distance )
    # print(f"argsorted_distance has been saved to {cfg['log']['log_dir']}/{cfg['log']['argsorted_distance']}")
    # initial_rank = argsorted_distance[:,:cfg['budget']['k']+1]
    # del distance_mat
    # storage =  'local'  if cfg["storage"]["local"] else fg["storage"]["bucket"]
    # upload_to_db(cfg["db"],storage,cfg["budget"]["k"],initial_rank,img_paths)
    # print("complete uploading process")



if __name__ == '__main__':
    # import argparse 
    # parser = argparse.ArgumentParser(description="Transitive Rebeler Platform")
    # parser.add_argument("-fb","--feature_bank", default="", help="path to feature_bank", type=str)
    # args = parser.parse_args()
    # load cfg
    import yaml
    with open("config.yaml") as cfg_file:
        cfg =  yaml.load(cfg_file, Loader=yaml.FullLoader)
    # if not args.feature_bank: cfg['pre_computed']['feature_bank'] : cfg['pre_computed']['feature_bank']  = parser.feature_bank

    main(cfg)


