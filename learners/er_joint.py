import torch
from torch.utils.data.dataset import ConcatDataset
import copy
from utils.replay_buffers import ReplayBufferReservoir
from learners.base_learning_classes import JointLearner

class JointER(JointLearner):
    def __init__(self, net, memory_size, results_dir='./tmp/results/'):
        super().__init__(net, results_dir)

        # Initialize replay buffers and data loaders
        self.replay_buffers = {}
        self.memory_loaders = {}
        self.memory_size = memory_size

    def train(self, trainloader, task_id, component_update_freq=100, num_epochs=100, save_freq=1, testloaders=None):
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
            self.net.freeze_structure()     # freeze structure for all tasks
            self.net.freeze_structure(freeze=False, task_id=task_id)    # except current one
            prev_reduction = []
            for i in range(self.net.num_outputs):
                prev_reduction.append(self.loss[i].reduction)
                self.loss[i].reduction = 'sum'     # make sure the loss is summed over instances
            
            tmp_dataset = copy.copy(trainloader.dataset)
            tmp_dataset.tensors = tmp_dataset.tensors + (torch.full((len(tmp_dataset),), task_id, dtype=int),)
            mega_dataset = ConcatDataset([loader.dataset for loader in self.memory_loaders.values()] + [tmp_dataset])
            mega_loader = torch.utils.data.DataLoader(mega_dataset,
                batch_size=trainloader.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
                )
            for i in range(num_epochs):
                for X, Y, t in mega_loader:
                    X = X.to(self.net.device, non_blocking=True)
                    Y = Y.to(self.net.device, non_blocking=True)
                    l = [0.] * self.net.num_outputs
                    n = 0
                    all_t = torch.unique(t)
                    for task_id_tmp in all_t:
                        Y_hat = self.net(X[t == task_id_tmp], task_id=task_id_tmp).unsqueeze(1)
                        for j in range(self.net.num_outputs):
                            l[j] += self.loss[j](Y_hat[:, :, j], Y[t == task_id_tmp, :, j])
                        n += X.shape[0]
                    total_loss = 0.
                    for k in range(self.net.num_outputs):
                        total_loss += l[k] / n
                    mean_loss = total_loss / self.net.num_outputs
                    self.optimizer.zero_grad()
                    mean_loss.backward()
                    self.optimizer.step()

                if i % save_freq == 0 or i == num_epochs - 1:
                    if eval_bool:
                        self.evaluate(testloaders)
                    self.save_data(i + 1, task_id, save_eval=eval_bool)
            for i in range(self.net.num_outputs):
                self.loss[i].reduction = prev_reduction[i] 
            self.update_multitask_cost(trainloader, task_id)

    def update_multitask_cost(self, trainloader, task_id):
        self.replay_buffers[task_id] = ReplayBufferReservoir(self.memory_size, task_id)
        for X, Y in trainloader:
            self.replay_buffers[task_id].push(X, Y)
        self.memory_loaders[task_id] =  (
            torch.utils.data.DataLoader(self.replay_buffers[task_id],
                batch_size=trainloader.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
                ))
