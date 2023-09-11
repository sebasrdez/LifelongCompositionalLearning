from utils.kfac_ewc import KFAC_EWC
from learners.base_learning_classes import CompositionalDynamicLearner

class CompositionalDynamicEWC(CompositionalDynamicLearner):
    def __init__(self, net, ewc_lambda=1e-3, results_dir='./tmp/results/'):
        super().__init__(net, results_dir)
        
        # Initialize K-FAC-based Elastic Weight Consolidation (EWC) preconditioner
        self.preconditioner = KFAC_EWC(self.net.components, ewc_lambda=ewc_lambda)

    def update_modules(self, trainloader, task_id):
        # Configure the network for updates
        self.net.freeze_modules(freeze=False)
        self.net.freeze_structure(freeze=True)

        for X, Y in trainloader:
            X, Y = X.to(self.net.device), Y.to(self.net.device)
            
            # Compute and optimize loss (two passes: one before hiding a module, one after)
            for _ in range(2):
                total_loss = 0.
                Y_hat = self.net(X, task_id=task_id).unsqueeze(1)
                for i in range(self.net.num_outputs):
                    total_loss += self.loss[i](Y_hat[:, :, i], Y[:, :, i])
                mean_loss = total_loss / self.net.num_outputs
                self.optimizer.zero_grad()
                self.preconditioner.zero_grad()
                mean_loss.backward()
                self.preconditioner.step(task_id, update_stats=False, update_params=True)
                self.optimizer.step()
                
                # Hide temporary module for second pass (only if in first iteration)
                if _ == 0:
                    self.net.hide_tmp_module()
                else:
                    self.net.recover_hidden_module()

        # Post-update configuration
        self.net.freeze_modules(freeze=True)
        self.net.freeze_structure(freeze=False, task_id=task_id)

    def update_multitask_cost(self, loader, task_id):
        # Configure the network for multitask cost computation
        self.net.freeze_modules(freeze=False)
        self.net.freeze_structure(freeze=True)
        
        for X, Y in loader:
            X, Y = X.to(self.net.device), Y.to(self.net.device)
            
            # Compute loss and update stats of the preconditioner (no optimizer step)
            total_loss = 0.
            Y_hat = self.net(X, task_id=task_id).unsqueeze(1)
            for i in range(self.net.num_outputs):
                total_loss += self.loss[i](Y_hat[:, :, i], Y[:, :, i])
            mean_loss = total_loss / self.net.num_outputs
            self.preconditioner.zero_grad()
            mean_loss.backward()
            self.preconditioner.step(task_id, update_stats=True, update_params=False)
            break

        # Post-update configuration
        self.net.freeze_modules(freeze=True)
        self.net.freeze_structure(freeze=False, task_id=task_id)

