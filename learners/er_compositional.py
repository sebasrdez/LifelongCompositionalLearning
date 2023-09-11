import torch
from torch.utils.data.dataset import ConcatDataset
import copy
from utils.replay_buffers import ReplayBufferReservoir
from learners.base_learning_classes import CompositionalLearner


class CompositionalER(CompositionalLearner):
    def __init__(self, net, memory_size, results_dir='./tmp/results/'):
        super().__init__(net, results_dir)

        # Initialize replay buffers and data loaders
        self.replay_buffers = {}
        self.memory_loaders = {}
        self.memory_size = memory_size

    def update_modules(self, trainloader, task_id):
        self.net.freeze_modules(freeze=False)
        self.net.freeze_structure(freeze=True)

        # Store previous loss reduction mode
        prev_reduction = []
        for i in range(self.net.num_outputs):
            prev_reduction.append(self.loss[i].reduction)
            self.loss[i].reduction = 'sum'     # make sure the loss is summed over instances

        # Append task_id to dataset and merge with previous data
        tmp_dataset = copy.copy(trainloader.dataset)
        tmp_dataset.tensors = tmp_dataset.tensors + (torch.full((len(tmp_dataset),), task_id, dtype=int),)
        mega_dataset = ConcatDataset([loader.dataset for loader in self.memory_loaders.values()] + [tmp_dataset])
        batch_size = trainloader.batch_size
        mega_loader = torch.utils.data.DataLoader(mega_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True)

        # Training loop using combined dataset
        for X, Y, t in mega_loader:
            X = X.to(self.net.device, non_blocking=True)
            Y = Y.to(self.net.device, non_blocking=True)
            l = [0.] * self.net.num_outputs
            n = 0
            all_t = torch.unique(t)
            for task_id_tmp in all_t:
                Y_hat = self.net(X[t == task_id_tmp], task_id=task_id_tmp).unsqueeze(1)
                for i in range(self.net.num_outputs):
                    l[i] += self.loss[i](Y_hat[:, :, i], Y[t == task_id_tmp, :, i])
                n += X.shape[0]
            total_loss = 0.
            for i in range(self.net.num_outputs):
                total_loss += l[i] / n
            mean_loss = total_loss / self.net.num_outputs
            self.optimizer.zero_grad()
            mean_loss.backward()
            self.optimizer.step()

        for i in range(self.net.num_outputs):
            self.loss[i].reduction = prev_reduction[i] 
        self.net.freeze_modules(freeze=True)
        self.net.freeze_structure(freeze=False, task_id=task_id)    # unfreeze only current task's structure

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
