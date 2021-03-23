# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 02:25:22 2021

@author: MaNaM
"""

import torch
import torchvision
from skimage import io
from torchvision import transforms as T
from torch.nn import  functional as F
import argparse

parser = argparse.ArgumentParser(description='Dogs vs Cats')
parser.add_argument('--checkpoint', type=str, help='location of the model file', default=None)
parser.add_argument('--image', type=str, help='the log dir of the image')
parser.add_argument('--labels', type=bool, help='location of the label file', default=True)

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(path_to_model):
	model = torchvision.models.resnet18(pretrained=True)
	check = torch.load(path_to_model, map_location=torch.device('cpu'))
	model.load_state_dict(check['state_dict'])
	for parameter in model.parameters():
		parameter.requires_grad = False
	model.to(device)
	model.eval()
	return model


def load_picture(path_to_image):
	image = io.imread(path_to_image)

	#data transforms
	transform = T.Compose([
		T.ToPILImage(),
		T.Resize(256),
		T.CenterCrop(224),
		T.ToTensor()])
	image = transform(image)
	image = image.to(device)
	image = torch.unsqueeze(image, 0)
	return image

def model_prediction(path_to_model , path_to_image, path_to_labels = r'labels_name.txt'):
	model = load_model(path_to_model)
	image = load_picture(path_to_image)
	with open(path_to_labels) as f:
		labels = [line.strip() for line in f.readlines()]

	out = model(image)
	_, predictions = torch.max(out,1)
	_, index = torch.max(out, 1)
	percentage = F.softmax(out, dim=1)[0] * 100
	print(labels[index[0]], percentage[index[0]].item())


if __name__ == '__main__':
	args = parser.parse_args()
	model_prediction(args.checkpoint,args.image)