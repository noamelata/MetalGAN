import os
import glob
import torch
from Params import *
import re

def get_version(path):
    return int(re.findall(re.compile(r".*_(\d*)_.*"), path)[0])

class Checkpoint:
    def __init__(self, models, optimizers, epoch, loss, script_name, load=False):
        self.models = models
        self.epoch = epoch
        self.optimizers = optimizers
        self.loss = loss
        self.version = 1
        self.script_name = script_name
        self.filename = script_name + "_" + str(self.version) + "_checkpoint.pth"
        self.path = os.path.join(os.getcwd(), "checkpoints", self.filename)
        if load:
            self.load()
        else:
            self.delete_old(backups=1)

    def evaluate_file_name(self):
        self.filename = self.script_name + "_" + str(self.version) + "_checkpoint.pth"
        self.path = os.path.join(os.getcwd(), "checkpoints", self.filename)

    def to_dict(self):
        dictionary = {'epoch': self.epoch,
                      'loss': self.loss}
        for model_name in self.models.keys():
            dictionary[model_name] = self.models[model_name].state_dict()
        for optimizer_name in self.optimizers.keys():
            dictionary[optimizer_name] = self.optimizers[optimizer_name].state_dict()
        return dictionary

    def delete_old(self, backups=3):
        checkpoints = glob.glob(os.path.join(os.getcwd(), "checkpoints", "*.pth"))
        checkpoints.sort(key=get_version)
        while len(checkpoints) > backups:
            os.remove(checkpoints.pop(0))

    def save(self):
        self.evaluate_file_name()
        torch.save(self.to_dict(), self.path)
        print("saved checkpoint at", self.path)
        self.version += 1
        self.delete_old()

    def load(self):
        paths = glob.glob(os.path.join(os.getcwd(), "checkpoints", "*.pth"))
        paths.sort()
        if len(paths) == 0:
            return 0
        checkpoint = torch.load(paths[-1])
        for model_name in self.models.keys():
            self.models[model_name].load_state_dict(checkpoint[model_name])
        for optimizer_name in self.optimizers.keys():
            self.optimizers[optimizer_name].load_state_dict(checkpoint[optimizer_name])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        self.version = get_version(paths[-1]) + 1
        self.evaluate_file_name()
        print("loaded checkpoint version", self.version - 1)

