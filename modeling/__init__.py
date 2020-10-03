# encoding: utf-8

from .baseline import Baseline

def build_model(cfg):
    model = Baseline(cfg["model"]["num_class"], cfg["model"]["last_stride"], cfg["model"]["backbone_weight"], cfg["model"]["backbone_name"],
                     cfg["model"]["generalized_mean_pool"], cfg["model"]["backbone_choice"])
    return model