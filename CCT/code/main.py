### YOUR CODE HERE
import torch
import os, argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import configs
from ImageUtils import visualize


parser = argparse.ArgumentParser()
parser.add_argument("mode", help="train, test or predict")
parser.add_argument("data_dir", help="path to the data")
parser.add_argument("--save_dir", default = configs["save_dir"], help="path to save the results")
args = parser.parse_args()

if __name__ == '__main__':
	model = MyModel(configs)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	if args.mode == 'train':
		train_loader, _ = load_data(args.data_dir)
		train_loader, valid_loader = train_valid_split(train_loader, train_ratio=0.8)

		model.train(train_loader, valid_loader)

	elif args.mode == 'test':
		# Testing on public testing dataset
		_, test_loader = load_data(args.data_dir)
		accuracy, cm, precision, recall, f1 = model.evaluate(test_loader, "test")
		print(f"Test Accuracy: {accuracy:.2f}%")
		print(f"Confusion Matrix:\n{cm}")
		print(f"Precision: {precision:.2f}")
		print(f"Recall: {recall:.2f}")
		print(f"F1 Score: {f1:.2f}")


		class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

		# Plotting the confusion matrix
		plt.figure(figsize=(10, 8))
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

		# Adding title and labels
		plt.title('Confusion Matrix (CCT)', fontsize=16)
		plt.xlabel('Predicted', fontsize=14)
		plt.ylabel('Actual', fontsize=14)

		# Rotate the tick labels for better readability
		plt.xticks(rotation=45, ha="right", fontsize=12)
		plt.yticks(rotation=45, va="center", fontsize=12)

		# Show the plot
		plt.tight_layout()  # Ensure the labels fit within the figure
		plt.savefig('confusion_matrix_cifar10.png', format='png')
