# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 22:54:08 2021

@author: MaNaM
"""

import os
import torch
import torch.optim as optim
import torchvision.models as models
from torchvision.transforms import transforms as T
from torch.utils.data.dataloader import DataLoader
from DogData import Data_with_labels
from skimage import io
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser(description='Dogs vs Cats')
parser.add_argument('--root_dir', type=str, help='location of the data file', default=None)
parser.add_argument('--csv_file', type=str, help='the log dir of the csv')


#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#paths
#root_dir = r'C:\Users\MaNaM\Desktop\tuna\dogs-vs-cats\all_data'
#csv_file = r'C:\Users\MaNaM\Desktop\tuna\labels.csv'
writer = SummaryWriter(f'runs')


def check_accuracy(loader, model, criterion):
	num_correct = 0
	num_samples = 0
	model.eval()

	with torch.no_grad():
		for x, y in loader:
			x = x.to(device=device)
			y = y.to(device=device)


			scores = model(x)
			loss = criterion(scores, y)
			_, predictions = scores.max(1)
			num_correct += (predictions == y).sum()
			num_samples += predictions.size(0)

	model.train()
	return float(num_correct)/float(num_samples), loss


def save_ckp(state, checkpoint_path, ):
	f_path = checkpoint_path
	torch.save(state, f_path)


def train(root_dir, csv_file, is_tensorboard = None):
	#data transforms
	transform = T.Compose([
		T.ToPILImage(),
		T.Resize(256),
		T.CenterCrop(224),
		T.ToTensor()])

	# Hyperparameters
	number_of_classes = 2
	learning_rate = 1e-3
	batch_size = 64
	num_epochs = 100


	#Load Data and split
	dataset = Data_with_labels(csv_file = csv_file, root_dir = root_dir, transforms = transform)
	train_set, test_set = torch.utils.data.random_split(dataset, [4800,1200])
	train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

	# Model
	model = torchvision.models.resnet18(pretrained=True)
	model.fc.out_features = number_of_classes
	for param in model.parameters():
		param.requires_grad = False
	model.fc.weight.requires_grad = True
	model.fc.bias.requires_grad = True
	model.to(device)

	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)



	is_best_acc = 0
	curr_test_acc = 0
	# Train Network

	step = 0
	for epoch in range(num_epochs):
		losses = []

		for batch_idx, (data, targets) in enumerate(train_loader):
			# Get data to cuda if possible
			data = data.to(device=device)
			targets = targets.to(device=device)

			# forward
			scores = model(data)
			loss = criterion(scores, targets)

			losses.append(loss.item())

			# backward
			optimizer.zero_grad()
			loss.backward()

			# gradient descent or adam step
			optimizer.step()

		curr_test_acc, test_loss = check_accuracy(test_loader, model, criterion)
		curr_train_acc, train_loss = check_accuracy(train_loader, model, criterion)

		writer.add_scalar('Training_loss', train_loss, global_step=step)
		writer.add_scalar('Testing_loss', test_loss, global_step=step)
		step += 1
		writer.add_scalar('Train_Accuracy', curr_train_acc, global_step=step)
		writer.add_scalar('Test_Accuracy', curr_test_acc, global_step=step)

		checkpoint = {
			'epoch': epoch + 1,
			'best_acc': is_best_acc,
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			}
		if curr_test_acc > is_best_acc:
			is_best_acc = curr_test_acc
			file_path = os.path.join(r'C:\Users\MaNaM\Desktop\tuna\checkpoints',str(is_best_acc)+'model-{:04d}.pth.tar'.format(int(epoch)))
			save_ckp(checkpoint, file_path)

		print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)} is_best_acc {is_best_acc}")


if __name__ == '__main__':
	#args = parser.parse_args()
	train(r'dogs-vs-cats/all_data',r'labels.csv')