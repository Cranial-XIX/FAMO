import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union


class FAMO:
    """
    Fast Adaptive Multitask Optimization.
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 0.01,   # the regularization coefficient
        w_lr: float = 0.025,   # the learning rate of the task logits
        max_norm: float = 1.0, # the maximum gradient norm
    ):
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.n_tasks = n_tasks
        self.device = device
    
    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses):
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss

    def update(self, curr_loss):
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss      - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
    ) -> Union[torch.Tensor, None]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        Returns
        -------
        Loss, extra outputs
        """
        loss = self.get_weighted_loss(losses=losses)
        loss.backward()
        if self.max_norm > 0 and shared_parameters is not None:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return loss


if __name__ == "__main__":

    n   = 1000 # number of datapoints
    dim = 20   # dimension of data
    K   = 100  # number of tasks
    X = torch.randn(n, dim)
    Y = torch.randn(n, K)

    model = torch.nn.Linear(dim, K)
    weight_opt = FAMO(n_tasks=K, device="cpu")
    opt = torch.optim.Adam(model.parameters())

    for it in range(100):
        loss = (Y - model(X)).pow(2).mean(0) # (K,)
        opt.zero_grad()
        weight_opt.backward(loss)
        opt.step()
        # update the task weighting
        with torch.no_grad():
            new_loss = (Y - model(X)).pow(2).mean(0) # (K,)
            weight_opt.update(new_loss)
        print(f"[info] iter {it:3d} | avg loss {loss.mean().item():.4f}")
