from utils.kfac_ewc import KFAC_EWC
from learners.base_learning_classes import JointLearner

class JointEWC(JointLearner):
    def __init__(self, net, ewc_lambda=1e-3, results_dir='./tmp/results/'):
        super().__init__(net, results_dir)
        
        # Initialize K-FAC-based Elastic Weight Consolidation (EWC) preconditioner
        self.preconditioner = KFAC_EWC(self.net.components, ewc_lambda=ewc_lambda)

    def train(self, trainloader, task_id, component_update_freq=100, num_epochs=100, save_freq=1, testloaders=None):
        # Track and evaluate tasks
        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1
        if testloaders:
            self.evaluate(testloaders)
        self.save_data(0, task_id, save_eval=testloaders is not None)
        
        # Distinct training procedures based on initialization phase or later tasks
        if self.T <= self.net.num_init_tasks:
            self.net.freeze_structure()
            self.init_train(trainloader, task_id, num_epochs, save_freq, testloaders)
        else:
            self.net.freeze_structure()
            self.net.freeze_structure(freeze=False, task_id=task_id)
            
            # Training over epochs
            for i in range(num_epochs):
                for X, Y in trainloader:
                    X, Y = X.to(self.net.device), Y.to(self.net.device)
                    
                    # Compute and optimize loss
                    total_loss = sum(self.loss[j](self.net(X, task_id=task_id).unsqueeze(1)[:, :, j], Y[:, :, j]) for j in range(self.net.num_outputs))
                    mean_loss = total_loss / self.net.num_outputs
                    self.optimizer.zero_grad()
                    self.preconditioner.zero_grad()
                    mean_loss.backward()
                    self.preconditioner.step(task_id, update_stats=False, update_params=True)
                    self.optimizer.step()
                
                # Evaluate and save periodically
                if i % save_freq == 0 or i == num_epochs - 1:
                    if testloaders:
                        self.evaluate(testloaders)
                    self.save_data(i + 1, task_id, save_eval=testloaders is not None)
            
            self.update_multitask_cost(trainloader, task_id)

    def update_multitask_cost(self, loader, task_id):
        # Compute the multitask cost to adjust for task interference
        self.net.freeze_structure(freeze=True)
        X, Y = next(iter(loader))
        X, Y = X.to(self.net.device), Y.to(self.net.device)
        
        total_loss = sum(self.loss[i](self.net(X, task_id=task_id).unsqueeze(1)[:, :, i], Y[:, :, i]) for i in range(self.net.num_outputs))
        mean_loss = total_loss / self.net.num_outputs
        self.preconditioner.zero_grad()
        mean_loss.backward()
        self.preconditioner.step(task_id, update_stats=True, update_params=False)
        
        self.net.freeze_structure(freeze=False, task_id=task_id)
