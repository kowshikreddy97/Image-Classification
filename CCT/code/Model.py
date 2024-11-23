### YOUR CODE HERE
import torch
import os, math
from time import time
import numpy as np
from Network import CCT
from ImageUtils import parse_record
from Configure import configs
from loss import LabelSmoothingCrossEntropy
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs):
        self.configs = configs
        self.model = CCT(num_layers=self.configs["num_layers"],
                           num_heads=self.configs["num_heads"],
                           mlp_ratio=self.configs["mlp_ratio"],
                           embedding_dim=self.configs["embedding_dim"],
                           kernel_size=self.configs["kernel_size"],
                           stride=self.configs["stride"],
                           padding=self.configs["padding"])
        
        self.criterion = LabelSmoothingCrossEntropy()
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.model.cuda(0)
        self.criterion = self.criterion.cuda(0)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.configs["lr"],
                                  weight_decay=self.configs["weight_decay"])

    def train(self, train_loader, valid_loader):
        print("Beginning training")
        time_begin = time()
        best_acc1 = 0
        for epoch in range(self.configs["epochs"]):
            self.adjust_learning_rate( epoch, lr=self.configs["lr"])
            self.model.train()
            loss_val, acc1_val = 0, 0
            n = 0
            print(len(train_loader), len(valid_loader))
            for i, (images, target) in enumerate(train_loader):
                images = images.cuda(0, non_blocking=True)
                target = target.cuda(0, non_blocking=True)
                output = self.model(images)

                loss = self.criterion(output, target)
                acc1 = self.accuracy(output, target)
                n += images.size(0)
                loss_val += float(loss.item() * images.size(0))
                acc1_val += float(acc1 * images.size(0))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 20 == 0:
                    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
                    print(f'[Epoch {epoch + 1}][Train][{i}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')
                if i % 10 == 0:
                    torch.save(self.model.state_dict(), self.configs["save_dir"] + 'checkpoint.pth')

            acc1 = self.evaluate(valid_loader, "train")
            best_acc1 = max(acc1, best_acc1)

        total_mins = (time() - time_begin) / 60
        print(f'Script finished in {total_mins:.2f} minutes, '
            f'best top-1: {best_acc1:.2f}, '
            f'final top-1: {acc1:.2f}')
        # torch.save(self.model.state_dict(), 'checkpoint_7.pth')

    def evaluate(self, test_loader, mode="test"):
        """
        Evaluate the model on the test dataset and calculate accuracy, confusion matrix,
        precision, recall, and F1 score.
        """
        if mode != "train":
            state_dict = torch.load(self.configs["save_dir"] + 'checkpoint.pth')
            self.model.load_state_dict(state_dict)

        self.model.eval()
        loss_val, acc1_val = 0, 0
        n = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i, (images, target) in enumerate(test_loader):
                images = images.cuda(0, non_blocking=True)
                target = target.cuda(0, non_blocking=True)

                output = self.model(images)
                loss = self.criterion(output, target)

                acc1 = self.accuracy(output, target)
                n += images.size(0)
                loss_val += float(loss.item() * images.size(0))
                acc1_val += float(acc1 * images.size(0))

                # Store predictions and targets for metrics calculation
                all_preds.append(output)
                all_targets.append(target)

                if i % 20 == 0:
                    avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
                    print(f'[Eval][{i}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')

        # Flatten all predictions and targets
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Calculate metrics
        avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
        cm, precision, recall, f1 = self.evaluate_metrics(all_preds, all_targets)

        # # Plot confusion matrix
        # class_names = [str(i) for i in range(all_targets.max() + 1)]  # Adjust class names as needed
        # self.plot_confusion_matrix(cm, class_names)

        # Return metrics
        return avg_acc1, cm, precision, recall, f1

    def predict_prob(self, x):
        state_dict = torch.load(self.configs["save_dir"] + 'checkpoint.pth')
        self.model.load_state_dict(state_dict)
        with torch.no_grad():
            self.model.eval()
            x = np.array(list(map(lambda l: parse_record(l, False), x)))
            x = torch.from_numpy(x).cuda()
            pred = self.model(x).cpu()
            return pred
    
    def accuracy(self, output, target):
        with torch.no_grad():
            pred = output.argmax(dim=1)
            correct = pred.eq(target)
            accuracy = correct.float().mean() * 100.0
            return accuracy.item()

    def evaluate_metrics(self, output, target):
        """
        Compute confusion matrix, precision, recall, F1 score, etc.
        """
        with torch.no_grad():
            pred = output.argmax(dim=1)
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()

            cm = confusion_matrix(target_np, pred_np)
            precision = precision_score(target_np, pred_np, average='macro', zero_division=0)
            recall = recall_score(target_np, pred_np, average='macro', zero_division=0)
            f1 = f1_score(target_np, pred_np, average='macro', zero_division=0)

            return cm, precision, recall, f1

    def plot_confusion_matrix(self, cm, class_names):
        """
        Plot the confusion matrix using seaborn heatmap.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()


    def adjust_learning_rate(self, epoch, lr):
        warmup = 5
        if epoch < warmup:
            lr = lr / (warmup - epoch)
        else:
            lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup) / (self.configs["epochs"] - warmup)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

### END CODE HERE
