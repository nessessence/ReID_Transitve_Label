def top_k_similarity(k,distance_mat):
    H, W = distance_mat.shape
    oneD_distance_mat = distance_mat.flatten()  # if we are sure then use 1d_distance_mat.view(-1) : kind of inplace
    top_k_value, oneD_indices = oneD_distance_mat.topk(k,largest=False) # largest = False ---> Top min : "distance" ---> "similarity"
    twoD_indices = torch.cat(((oneD_indices // W).unsqueeze(1), (oneD_indices % W).unsqueeze(1)), dim=1)
    print(f"twoD_indices.shape : {twoD_indices.shape} ")
    return   top_k_value,twoD_indices

def transitive(feature_bank,dataset)
    print("start Greedy Max Cut")
    from collections import defaultdict 
    G = defaultdict(set) # Graph 
    img_paths = list(feature_bank.keys())
    To_be_label_img = [] ; To_be_Transitive_img = defaultdict(list) 
    # k here is the "Bank_Budget" limited by memory
    top_k_similar,top_k_indices = top_k_similarity(min(len(dataset)+cfg["budget"]["bank"],len(dataset)*len(dataset)) ,distance_mat)
    del distance_mat,feature_bank,top_k_similar
    top_k_indices = top_k_indices.cpu().numpy()
    percent=0
    for a,b in top_k_indices[len(dataset):]:
        if len(To_be_label_img)*100.0/cfg["budget"]["label"] >= percent+1:
            percent+= 1
            print(f"{percent}% has completed") 
        if a == b: continue
        a,b = min(a,b),max(a,b)
        imgA = img_paths[a]; imgB = img_paths[b]
        if len(To_be_label_img) >= cfg["budget"]["label"]: break
        # Triangle check 
        a_X_b = G[a].intersection(G[b])
        if len(a_X_b) > 0:
            #print(f"{(a,b)} is to be transitived! by {}")
            To_be_Transitive_img[(imgA,imgB)] +=  list(a_X_b)
            if len(To_be_Transitive_img) % 1000 == 0: print(f"Now : {len(To_be_label_img) } pairs to be labeled, {len(To_be_Transitive_img)} pairs to be transitived ")
            continue
        G[a].add(b); G[b].add(a)
        To_be_label_img.append((imgA,imgB))
return  To_be_label_img,To_be_Transitive_img