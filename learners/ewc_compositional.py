from utils.kfac_ewc import KFAC_EWC
from learners.base_learning_classes import CompositionalLearner

class CompositionalEWC(CompositionalLearner):
    def __init__(self, net, ewc_lambda=1e-3, results_dir='./tmp/results/'):
        super().__init__(net, results_dir)
        
        # Initialize K-FAC-based Elastic Weight Consolidation (EWC) preconditioner
        self.preconditioner = KFAC_EWC(self.net.components, ewc_lambda=ewc_lambda)

    def update_modules(self, trainloader, task_id):
        # Prepare the network for module updates: unfreeze modules and freeze structure
        self.net.freeze_modules(freeze=False)
        self.net.freeze_structure(freeze=True)

        # Training loop
        for X, Y in trainloader:
            X, Y = X.to(self.net.device), Y.to(self.net.device)
            total_loss = 0.
            Y_hat = self.net(X, task_id=task_id).unsqueeze(1)
            
            # Compute the total loss across all outputs
            for i in range(self.net.num_outputs):
                total_loss += self.loss[i](Y_hat[:, :, i], Y[:, :, i])
            
            # Backward pass and optimizer steps
            mean_loss = total_loss / self.net.num_outputs
            self.optimizer.zero_grad()
            self.preconditioner.zero_grad()
            mean_loss.backward()
            self.preconditioner.step(task_id, update_stats=False, update_params=True)
            self.optimizer.step()

        self.net.freeze_modules(freeze=True)
        self.net.freeze_structure(freeze=False, task_id=task_id)

    def update_multitask_cost(self, loader, task_id):
        self.net.freeze_modules(freeze=False)
        self.net.freeze_structure(freeze=True)
        for X, Y in loader:
            X = X.to(self.net.device, non_blocking=True)
            Y = Y.to(self.net.device, non_blocking=True)
            total_loss = 0.
            Y_hat = self.net(X, task_id=task_id).unsqueeze(1)
            for i in range(self.net.num_outputs):
                total_loss += self.loss[i](Y_hat[:, :, i], Y[:, :, i])
            mean_loss = total_loss / self.net.num_outputs
            self.preconditioner.zero_grad()
            mean_loss.backward()
            self.preconditioner.step(task_id, update_stats=True, update_params=False)
            break

        self.net.freeze_modules(freeze=True)
        self.net.freeze_structure(freeze=False, task_id=task_id)
