# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 22:54:08 2021

@author: MaNaM
"""


import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

class Data_with_labels(Dataset):
	def __init__(self, csv_file, root_dir, transforms=None):
		self.annotation = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transforms = transforms

	def __len__(self):
		return len(self.annotation)

	def __getitem__(self, index):
		img_path = os.path.join(self.root_dir, self.annotation.iloc[index, 0])
		image = io.imread(img_path)
		y_label = torch.tensor(int(self.annotation.iloc[index, 1]))

		if self.transforms:
			image = self.transforms(image)
		return (image, y_label)


class Data_without_labels(Dataset):
	def __init__(self, csv_file, root_dir, transforms=None):
		self.annotation = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transforms = transforms

	def __len__(self):
		return len(self.annotation)

	def __getitem__(self, index):
		img_path = os.path.join(self.root_dir, self.annotation['id'][index])
		image = io.imread(img_path)

		if self.transforms:
			image = self.transforms(image)
		return (image)

