import sys

sys.path.append("gaussian-splatting")

import argparse
import math
import cv2
import torchvision
import torch
import torch.nn as nn
import os
import numpy as np
import json
import copy
from tqdm import tqdm

class Force2bTrained:
    
    # init with 1 bc dict
    def __init__(self, bc, time_params):
        if bc["type"] == "cuboid":
            self.type = "cuboid"
            assert (
                "point" in bc.keys() and "size" in bc.keys() and "velocity" in bc.keys()
            )
            self.point = nn.Parameter(torch.Tensor(bc['point']))
            self.size = nn.Parameter(torch.Tensor(bc['size']))
            if bc["fix_only"]:
                self.velocity = torch.Tensor([0., 0., 0.])
                if "start_time" in bc.keys():
                    self.start_time = bc["start_time"]
                if "end_time" in bc.keys():
                    self.end_time = bc["end_time"]
                if "reset" in bc.keys():
                    self.reset = bc["reset"]
            else:
                self.velocity = nn.Parameter(torch.Tensor([0., 0., 0.]))
                self.start_time = nn.Parameter(torch.Tensor(0.))
                self.end_time = nn.Parameter(torch.Tensor(1000.))
                self.reset = bc["reset"]
        elif bc["type"] == "particle_impulse":
            self.type = "particle_impulse"
            self.force = nn.Parameter(torch.Tensor([0., 0., 0.]))
            self.start_time = nn.Parameter(torch.Tensor(0.))
            self.num_dt = 1
            self.point = nn.Parameter(torch.Tensor([1., 1., 1.]))
            self.size = nn.Parameter(torch.Tensor([1., 1., 1.]))
            self.dt = time_params["substep_dt"]
    
    def forward(self, mpm_solver):
        if self.type == "cuboid":
            pass
            
            
                