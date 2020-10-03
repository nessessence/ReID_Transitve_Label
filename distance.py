import torch
def Mahalanobis_distance(Features):
    n_Feature = Features.shape[0]
    distmat = torch.pow(Features,2).sum(dim=1, keepdim=True).expand(n_Feature,n_Feature) + \
                      torch.pow(Features, 2).sum(dim=1, keepdim=True).expand(n_Feature, n_Feature).t()
    distmat.addmm_(1,-2,Features,Features.t())
    distmat = torch.div( distmat,torch.max(distmat, axis=0)[0])
    return distmat.T 
    #initial_rank = np.argsort(distmat).astype(np.int32)
    #return initial_rank