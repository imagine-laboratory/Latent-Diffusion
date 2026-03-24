import os
import wandb
import argparse
import numpy as np




def create_directory(directory):
    os.makedirs(directory, exist_ok=True)

def log_bar(name, loss_dict, step):
    """Helper to build a W&B bar plot from timestep→[losses]."""
    epoch = step
    table = wandb.Table(columns=["epoch", "timestep", "loss"])
    for t, vals in sorted(loss_dict.items()):
        table.add_data(epoch, t, float(np.mean(vals)))
    wandb.log({
        name: wandb.plot.bar(
            table, "timestep", "loss", title=name.replace("_", " ").title()
        )
    }, step=epoch)

def setup_wandb(lr, epochs, batch_size, run_name, run_id=None):
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    init_args = {
        "entity": "imagine-laboratory-conare",
        "project": "SD_training_exp1",
        "name": run_name,
        "config": {
            "learning_rate": lr,
            "architecture": "stable_diffusion",
            "dataset": "Pineapples",
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "AdamW"
        }
    }
    if run_id is not None:
        # Resume into an existing run
        init_args["id"] = run_id
        init_args["resume"] = "allow"

    run = wandb.init(**init_args)
    return run