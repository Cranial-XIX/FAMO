import os
import time
import wandb
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from experiments.toy.problem import Toy
from experiments.toy.utils import plot_2d_pareto
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    set_logger,
)
from methods.weight_methods import WeightMethods

set_logger()


def main(method_type, device, n_iter, scale):
    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    n_tasks = 2

    F = Toy(scale=scale)

    all_traj = dict()
    all_time = dict()

    # the initial positions
    inits = [
        torch.Tensor([-8.5, 7.5]),
        torch.Tensor([0.0, 0.0]),
        torch.Tensor([9.0, 9.0]),
        torch.Tensor([-7.5, -0.5]),
        torch.Tensor([9, -1.0]),
    ]

    for i, init in enumerate(inits):
        traj = []
        x = init.clone()
        x.requires_grad = True
        x = x.to(device)

        #if "famo" in method_type:
        #    weight_methods_parameters["min_gamma"] = 0.0

        method = WeightMethods(
            method=method_type,
            device=device,
            n_tasks=n_tasks,
            **weight_methods_parameters[method_type],
        )
        if "famo" in method_type:
            method.method.set_min_losses(torch.Tensor([-21 * scale, -21]))

        optimizer = torch.optim.Adam(
            [
                dict(params=[x], lr=1e-3),
                dict(params=method.parameters(), lr=args.method_params_lr),
            ],
        )

        t0 = time.time()
        for it in tqdm(range(n_iter)):
            traj.append(x.cpu().detach().numpy().copy())

            optimizer.zero_grad()
            f = F(x, False)
            if "famo" in args.method and it > 0:
                with torch.no_grad():
                    method.method.update(f)

            _, extra_outputs = method.backward(
                losses=f,
                shared_parameters=(x,),
                task_specific_parameters=None,
                last_shared_parameters=None,
                representation=None,
            )
            optimizer.step()
        t1 = time.time()
        all_time[i] = t1-t0
        all_traj[i] = dict(init=init.cpu().detach().numpy().copy(), traj=np.array(traj))

    return all_traj, all_time


if __name__ == "__main__":
    parser = ArgumentParser(
        "Toy example (modification of the one in CAGrad)", parents=[common_parser]
    )
    parser.set_defaults(n_epochs=50000, method="nashmtl", data_path=None)
    parser.add_argument(
        "--scale", default=1e-1, type=float, help="scale for first loss"
    )
    parser.add_argument("--out-path", default="outputs", type=Path, help="output path")
    parser.add_argument("--wandb_project", type=str, default=None, help="Name of Weights & Biases Project.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Name of Weights & Biases Entity.")
    args = parser.parse_args()

    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    out_path = args.out_path
    out_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Logs and plots are saved in: {out_path.as_posix()}")

    device = torch.device("cpu")
    all_traj, all_time = main(
        method_type=args.method, device=device, n_iter=args.n_epochs, scale=args.scale
    )
    os.system("mkdir -p ./time")
    torch.save(all_time, f"./time/{args.method}.time")

    # plot
    ax, fig, legend = plot_2d_pareto(trajectories=all_traj, scale=args.scale)

    title_map = {
        "nashmtl": "Nash-MTL",
        "cagrad": "CAGrad",
        "mgda": "MGDA",
        "pcgrad": "PCGrad",
        "ls": "LS",
    }
    if "famo" in args.method:
        ax.set_title(f"FAMO (γ={args.gamma})", fontsize=25)
    else:
        ax.set_title(title_map[args.method], fontsize=25)

    save_path = out_path / f"{args.method}.png" if "famo" not in args.method else out_path / f"{args.method}_gamma{args.gamma}.png"
    plt.savefig(
        save_path,
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    if wandb.run is not None:
        wandb.log({"Pareto Front": wandb.Image((out_path / f"{args.method}.png").as_posix())})
        wandb.finish()
