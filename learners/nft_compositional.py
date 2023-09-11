from learners.base_learning_classes import CompositionalLearner

class CompositionalNFT(CompositionalLearner):
    def __init__(self, net, results_dir='./tmp/results/'):
        super().__init__(net, results_dir)

    def update_modules(self, trainloader, task_id):
        self.net.freeze_modules(freeze=False)
        self.net.freeze_structure(freeze=True)
        for X, Y in trainloader:
            X = X.to(self.net.device, non_blocking=True)
            Y = Y.to(self.net.device, non_blocking=True)
            Y_hat = self.net(X, task_id=task_id).unsqueeze(1)
            total_loss = 0.
            for i in range(self.net.num_outputs):
                total_loss += self.loss[i](Y_hat[:, :, i], Y[:, :, i]) 
            mean_loss = total_loss / self.net.num_outputs
            self.optimizer.zero_grad()
            mean_loss.backward()
            self.optimizer.step()
        self.net.freeze_modules(freeze=True)
        self.net.freeze_structure(freeze=False, task_id=task_id)

    def update_multitask_cost(self, trainloader, task_id):
        pass