from argparse import ArgumentParser

import numpy as np
import time
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.celeba.data import CelebaDataset
from experiments.celeba.models import Network
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)


class CelebaMetrics():
    """
    CelebA metric accumulator.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        
    def incr(self, y_, y_task):
        y_task = y_task.cpu().numpy()
        y_pred = (y_.detach().cpu().gt(0.5)).numpy()
        self.tp += (y_pred * y_task).sum()
        self.fp += (y_pred * (1 - y_task)).sum()
        self.fn += ((1 - y_pred) * y_task).sum()
                
    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall    = self.tp / (self.tp + self.fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        return f1.item()


def main(path, lr, bs, device, task, seed):
    # we only train for specific task
    model = Network().to(device)
    
    train_set = CelebaDataset(data_dir=path, split='train')
    val_set = CelebaDataset(data_dir=path, split='val')
    test_set = CelebaDataset(data_dir=path, split='test')
    train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=bs, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=bs, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs    = args.n_epochs
    metrics   = np.zeros([epochs], dtype=np.float32) # test_f1
    metric    = CelebaMetrics()
    loss_fn   = torch.nn.BCELoss()

    best_val_f1 = 0.0
    best_epoch = None
    for epoch in range(epochs):
        # training
        t0 = time.time()
        for x, y in tqdm.tqdm(train_loader):
            x = x.to(device)
            y_task = y[task].to(device)
            y_ = model(x, task=task)
            loss = loss_fn(y_, y_task)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t1 = time.time()

        # validation
        metric.reset()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y_task = y[task].to(device)
                y_ = model(x, task=task)
                loss = loss_fn(y_, y_task)
                metric.incr(y_, y_task)
        val_f1 = metric.result()
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch

        # testing
        metric.reset()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y_task = y[task].to(device)
                y_ = model(x, task=task)
                loss = loss_fn(y_, y_task)
                metric.incr(y_, y_task)
        test_f1 = metric.result()
        metrics[epoch] = test_f1
        t2 = time.time()
        print(f"[info] epoch {epoch+1} | train takes {(t1-t0)/60:.1f} min | test takes {(t2-t1)/60:.1f} min")
        torch.save({"metric": metrics, "best_epoch": best_epoch}, f"./save/task{task}_{seed}.stats")


if __name__ == "__main__":
    parser = ArgumentParser("Celeba", parents=[common_parser])
    parser.set_defaults(
        data_path="/scratch/cluster/bliu_new/nash-mtl/experiments/celeba/dataset",
        lr=3e-4,
        n_epochs=15,
        batch_size=256,
        task=0,
    )
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)
    main(path=args.data_path,
         lr=args.lr,
         bs=args.batch_size,
         device=device,
         task=args.task,
         seed=args.seed)
