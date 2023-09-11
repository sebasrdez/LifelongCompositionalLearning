import torch
import torch.nn as nn
import os
from itertools import zip_longest
import numpy as np

class Learner():
    def __init__(self, net, results_dir='./tmp/results/'):
        self.net = net  # Neural network model
        self.loss = []  # List to store different loss functions
        
        # Initialize the appropriate loss function for each output
        for i in range(self.net.num_outputs):
            if self.net.regression[i]:
                self.loss.append(nn.MSELoss())
            else:
                self.loss.append(nn.BCEWithLogitsLoss()) 
        # Use Adam optimizer for training
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        
        self.T = 0
        self.observed_tasks = set()  # Set to store observed tasks
        self.results_dir = results_dir  # Directory to store results
        self.init_trainloaders = None  # Store initial training data loaders

    def train(self, *args, **kwargs):
        # Placeholder for the main training loop
        raise NotImplementedError('Training loop is algorithm specific')

    def init_train(self, trainloader, task_id, num_epochs, save_freq=1, testloaders=None):
        # This function initializes training for a given task
        if self.init_trainloaders is None:
            self.init_trainloaders = {}
        self.init_trainloaders[task_id] = trainloader
        
        eval_bool = testloaders is not None  # Boolean indicating if there are test loaders

        if len(self.init_trainloaders) == self.net.num_init_tasks:
            for i in range(num_epochs): 
                # Iterates over batches from different tasks in parallel
                for XY_all in zip_longest(*self.init_trainloaders.values()):
                    for task, XY in zip(self.init_trainloaders.keys(), XY_all):
                        if XY is not None:
                            X, Y = XY
                            X = X.to(self.net.device, non_blocking=True)
                            Y = Y.to(self.net.device, non_blocking=True)
                            self.gradient_step(X, Y, task)
                # Evaluate and save data periodically
                if i % save_freq == 0 or i == num_epochs - 1:
                    if eval_bool:
                        self.evaluate(testloaders)
                    self.save_data(i + 1, task_id, save_eval=eval_bool)
            for task, loader in self.init_trainloaders.items():
                self.update_multitask_cost(loader, task)

    def evaluate(self, testloaders):
        # This function evaluates the model using test data
        was_training = self.net.training 
        prev_reduction = []

        # Store current loss reduction method and set to sum for evaluation
        for i in range(self.net.num_outputs):
            prev_reduction.append(self.loss[i].reduction)
            self.loss[i].reduction = 'sum'
        
        self.net.eval()  # Set the model to evaluation mode
        
        with torch.no_grad():  # No gradient computation during evaluation
            self.test_loss = {}
            self.test_acc = {}
            for task, loader in testloaders.items():
                # Variables to store cumulative loss and accuracy
                l = [0.] * self.net.num_outputs
                acc = [0.] * self.net.regression.count(False)
                n = len(loader.dataset)  # Number of samples
                cnt_not_regression = 0
                for X, Y in loader:
                    X = X.to(self.net.device, non_blocking=True)
                    Y = Y.to(self.net.device, non_blocking=True)
                    Y_hat = self.net(X, task).unsqueeze(1)  # Model predictions

                    # Calculate and accumulate loss and accuracy for each output
                    for i in range(self.net.num_outputs):
                        l[i] += self.loss[i](Y_hat[:, :, i], Y[:, :, i])
                        if not self.net.regression[i]:
                            acc[cnt_not_regression] += ((Y_hat[:, :, i] > 0.5) == (Y[:, :, i] == 1)).sum().item()
                            cnt_not_regression += 1
                
                # Store computed test loss and accuracy for each task
                cnt_not_regression = 0
                for i in range(self.net.num_outputs):
                    if self.net.regression[i]:
                        self.test_loss.setdefault(task, []).append(np.sqrt(l[i] / n))
                    else:
                        self.test_loss.setdefault(task, []).append(l[i] / n)
                        self.test_acc.setdefault(task, []).append(acc[cnt_not_regression] / n)
                        cnt_not_regression += 1
        
        # Restore the previous loss reduction method
        for i in range(self.net.num_outputs):
            self.loss[i].reduction = prev_reduction[i] 
        if was_training:
            self.net.train()  # Return model to training mode

    def gradient_step(self, X, Y, task_id):
        # This function performs a single optimization step using the provided batch
        Y_hat = self.net(X, task_id=task_id).unsqueeze(1)
        total_loss = 0.
        
        # Calculate total loss across all outputs
        for i in range(self.net.num_outputs):
            total_loss += self.loss[i](Y_hat[:, :, i], Y[:, :, i]) 
        mean_loss = total_loss / self.net.num_outputs
        
        # Backpropagate the error and update model parameters
        self.optimizer.zero_grad()
        mean_loss.backward()
        self.optimizer.step()

    def save_data(self, epoch, task_id, save_eval=False):
        # This function saves model checkpoints and evaluation logs
        task_results_dir = os.path.join(self.results_dir, 'task_{}'.format(task_id))
        os.makedirs(task_results_dir, exist_ok=True)

        # Save model checkpoints periodically
        if epoch == 0 or epoch % 10 == 0:
            path = os.path.join(task_results_dir, 'checkpoint.pt'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'observed_tasks': self.observed_tasks,
                }, path)
        
        # Log evaluation results if required
        if save_eval:
            log_open_mode = 'a' if epoch > 0 else 'w'
            with open(os.path.join(task_results_dir, 'log.txt'), log_open_mode) as f:
                line = 'epochs: {}, training task: {}\n'.format(epoch, task_id)
                f.write(line)
                print(line, end='')
                for task in self.test_loss:
                    line = '\ttask: {}'.format(task)
                    for i, loss in enumerate(self.test_loss[task]):
                        if type(loss) == tuple:
                            line += '\tloss{}: ({:.4f}, {:.4f})'.format(i+1, loss[0], loss[1])
                        else:
                            line += '\tloss{}: {:.4f}'.format(i+1, loss)
                    line += '\n'
                    f.write(line)
                    print(line, end='')
    
    def update_multitask_cost(self, loader, task_id):
        # Placeholder for updating multitask cost
        raise NotImplementedError('Update update_multitask is algorithm specific')


class CompositionalLearner(Learner):
    """Define a compositional learning approach."""

    def train(self, trainloader, task_id, component_update_freq=100, num_epochs=100, save_freq=1, testloaders=None):

        # Check if the current task_id has been observed before
        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1

        # Check if testloaders are provided and evaluate the network.
        eval_bool = testloaders is not None
        if eval_bool:
            self.evaluate(testloaders)
        self.save_data(0, task_id, save_eval=eval_bool)

         # If number of observed tasks is within the limit of initial tasks, use the initial training method.
        if self.T <= self.net.num_init_tasks:
            self.net.freeze_structure()
            self.init_train(trainloader, task_id, num_epochs, save_freq, testloaders)
        else:
            # Else, freeze modules and adapt structure.
            self.net.freeze_modules()
            self.net.freeze_structure()     # freeze structure for all tasks
            self.net.freeze_structure(freeze=False, task_id=task_id)    # except current one
            for i in range(num_epochs):
                if (i + 1) % component_update_freq == 0:
                    self.update_modules(trainloader, task_id)   # replace one structure epoch with one module epoch
                else:
                    for X, Y in trainloader:
                        X = X.to(self.net.device, non_blocking=True)
                        Y = Y.to(self.net.device, non_blocking=True)
                        self.update_structure(X, Y, task_id)
                if i % save_freq == 0 or i == num_epochs - 1:
                    if eval_bool:
                        self.evaluate(testloaders)
                    self.save_data(i + 1, task_id, save_eval=eval_bool)
            self.update_multitask_cost(trainloader, task_id)

    # Update the network structure using gradients.
    def update_structure(self, X, Y, task_id):
        self.gradient_step(X, Y, task_id)    # assume shared parameters are frozen and just take a gradient step on the structure

    def update_modules(self, *args, **kwargs):
        raise NotImplementedError('Update modules is algorithm specific')

class JointLearner(Learner):
    pass

class NoComponentsLearner(Learner):
    pass

class CompositionalDynamicLearner(CompositionalLearner):
    """This class extends the functionality of CompositionalLearner by allowing dynamic changes to the network components."""

    def train(self, trainloader, task_id, valloader, component_update_freq=100, num_epochs=100, save_freq=1, testloaders=None):

        # Initial training steps are similar to CompositionalLearner.
        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1
        eval_bool = testloaders is not None
        if eval_bool:
            self.evaluate(testloaders)
        self.save_data(0, task_id, save_eval=eval_bool)
        if self.T <= self.net.num_init_tasks:
            self.net.freeze_structure()
            self.init_train(trainloader, task_id, num_epochs, save_freq, testloaders)
        else:
            # Add dynamic handling of network components.
            self.net.freeze_modules()
            self.net.freeze_structure()     # freeze structure for all tasks
            self.net.add_tmp_module(task_id)   # freeze original modules and structure
            self.optimizer.add_param_group({'params': self.net.components[-1].parameters()})
            if hasattr(self, 'preconditioner'):
                self.preconditioner.add_module(self.net.components[-1])

            self.net.freeze_structure(freeze=False, task_id=task_id)    # unfreeze (new) structure for current task

            for i in range(num_epochs):
                if (i + 1) % component_update_freq == 0:
                    self.update_modules(trainloader, task_id)
                else:
                    for X, Y in trainloader:
                        X = X.to(self.net.device, non_blocking=True)
                        Y = Y.to(self.net.device, non_blocking=True)
                        self.update_structure(X, Y, task_id)
                        self.net.hide_tmp_module()
                        self.update_structure(X, Y, task_id)
                        self.net.recover_hidden_module()
                if i % save_freq == 0 or i == num_epochs - 1:
                    if eval_bool:
                        self.evaluate(testloaders)
                    self.save_data(i + 1, task_id, save_eval=eval_bool)

            # Decide whether to keep the temporary module or not
            self.conditionally_add_module(valloader, task_id)
            self.save_data(num_epochs + 1, task_id, save_eval=False, final_save=True)
            self.update_multitask_cost(trainloader, task_id)

    # Method to evaluate whether to keep a temporary module or not
    def conditionally_add_module(self, valloader, task_id):
        test_loss = self.test_loss
        test_acc = self.test_acc
        self.evaluate({task_id: valloader})
        W_losses, WO_losses = [], []
        for output in self.test_loss[task_id]:
            W_losses.append(output[0])
            WO_losses.append(output[1])
        update_loss = np.mean(W_losses)
        no_update_loss = np.mean(WO_losses)
        print('W/update: {}, WO/update: {}'.format(update_loss, no_update_loss))
        if (no_update_loss - update_loss) / update_loss > .05:
            print('Keeping new module. Total: {}\n'.format(self.net.num_components))
        else:
            self.net.remove_tmp_module()
            print('Not keeping new module. Total: {}\n'.format(self.net.num_components))

        self.test_loss = test_loss
        self.test_acc = test_acc

    # Evaluate network performance on multiple tasks
    def evaluate(self, testloaders, eval_no_update=True):
        was_training = self.net.training 
        prev_reduction = []
        for i in range(self.net.num_outputs):
            prev_reduction.append(self.loss[i].reduction)
            self.loss[i].reduction = 'sum'     # make sure the loss is summed over instances
        self.net.eval()
        with torch.no_grad():
            self.test_loss = {}
            self.test_acc = {}
            for task, loader in testloaders.items():
                l = [0.] * self.net.num_outputs
                acc = [0.] * self.net.regression.count(False)
                n = len(loader.dataset)
                cnt_not_regression = 0
                for X, Y in loader:
                    X = X.to(self.net.device, non_blocking=True)
                    Y = Y.to(self.net.device, non_blocking=True)
                    Y_hat = self.net(X, task).unsqueeze(1)
                    for i in range(self.net.num_outputs):
                        l[i] += self.loss[i](Y_hat[:, :, i], Y[:, :, i])
                        if not self.net.regression[i]:
                            acc[cnt_not_regression] += ((Y_hat[:, :, i] > 0.5) == (Y[:, :, i] == 1)).sum().item()
                            cnt_not_regression += 1
                cnt_not_regression = 0
                if eval_no_update and task == self.T - 1 and self.T > self.net.num_init_tasks:
                    self.net.hide_tmp_module()
                    l1 = [0.] * self.net.num_outputs
                    acc1 = [0.] * self.net.regression.count(False)
                    for X, Y in loader:
                        X = X.to(self.net.device, non_blocking=True)
                        Y = Y.to(self.net.device, non_blocking=True)
                        Y_hat = self.net(X, task).unsqueeze(1)
                        for i in range(self.net.num_outputs):
                            l1[i] += self.loss[i](Y_hat[:, :, i], Y[:, :, i])
                            if not self.net.regression[i]:
                                acc1[cnt_not_regression] += ((Y_hat[:, :, i] > 0.5) == (Y[:, :, i] == 1)).sum().item()
                                cnt_not_regression += 1
                    cnt_not_regression = 0
                    for i in range(self.net.num_outputs):
                        if self.net.regression[i]:
                            if task not in self.test_loss.keys():
                                self.test_loss[task] = [(np.sqrt(l[i] / n), np.sqrt(l1[i] / n))]
                            else:
                                self.test_loss[task].append((np.sqrt(l[i] / n), np.sqrt(l1[i] / n)))
                        else:
                            if task not in self.test_loss.keys():
                                self.test_loss[task] = [(l[i] / n, l1[i] / n)]
                            else:
                                self.test_loss[task].append((l[i] / n, l1[i] / n))
                            if task not in self.test_acc.keys():
                                self.test_acc[task] = [(acc[cnt_not_regression] / n, acc1[cnt_not_regression] / n)]
                            else:
                                self.test_acc[task] = [(acc[cnt_not_regression] / n, acc1[cnt_not_regression] / n)]
                            cnt_not_regression += 1
                    self.net.recover_hidden_module()
                else: 
                    for i in range(self.net.num_outputs):
                        if self.net.regression[i]:
                            if task not in self.test_loss.keys():
                                self.test_loss[task] = [np.sqrt(l[i] / n)]
                            else:
                                self.test_loss[task].append(np.sqrt(l[i] / n))
                        else:
                            if task not in self.test_loss.keys():
                                self.test_loss[task] = [l[i] / n]
                            else:
                                self.test_loss[task].append(l[i] / n)
                            if task not in self.test_acc.keys():
                                self.test_acc[task] = [acc[cnt_not_regression] / n]
                            else:
                                self.test_acc[task] = [acc[cnt_not_regression] / n]
                            cnt_not_regression += 1

        for i in range(self.net.num_outputs):
            self.loss[i].reduction = prev_reduction[i] 
        if was_training:
            self.net.train()

    # Save training data
    def save_data(self, epoch, task_id, save_eval=False, final_save=False):
        super().save_data(epoch, task_id, save_eval)
        if final_save:
            task_results_dir = os.path.join(self.results_dir, 'task_{}'.format(task_id))
            with open(os.path.join(task_results_dir, 'num_components.txt'), 'w') as f:
                line = 'final components: {}'.format(self.net.num_components)
                f.write(line)

