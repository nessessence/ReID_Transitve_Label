storage:
  local: True
  bucket: ''
db:
  URI:  ''
  name: ''
  collection: ''

budget:
  k : 10
  bank: 100000000
  label: 200000

model:
  num_class: 1
  backbone_choice: 'imagenet'
  backbone_weight:  './modeling/pretrained/resnet50-19c8e357.pth'
  backbone_name: 'resnet50_nl'
  weight:  './modeling/pretrained/market1501_AGW.pth'
  center_loss: 'on'
  center_feat_dim: 2048
  weight_regularized_triplet: 'off'
  generalized_mean_pool: 'on'
  last_stride: 1

input:
  img_size: [256, 128]
  pixel_mean: [0.485, 0.456, 0.406]
  pixel_std:  [0.229, 0.224, 0.225]

datasets:
  nested_dir: True
  dir_path: '.'

dataloader:
  batch_size: 128
  num_workers: 8

log: 
  log_dir: "./log"
  feature_bank:  feature_bank.tensor
  distance_mat: distance_mat.tensor
  labeled_graph: graph.tensor
  To_be_label_img:  toLB.tensor
  To_be_Transitive_img:  toTrans.tensor
  
# './data/market1501/bounding_box_test/'

