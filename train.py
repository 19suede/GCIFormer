import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import os
import argparse
from datetime import datetime
from dataset import PansharpeningDataset
import random
import numpy as np

from utils import select_model
from config import select_config
from loss_fn import Criterion
import wandb # Import wandb

torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
os.environ["WANDB_START_METHOD"] = "thread"


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def set_args():
    parser = argparse.ArgumentParser(description="Pansharpening implementation")

    parser.add_argument("--seed", default=1234, type=int, required=False)
    parser.add_argument("--dataset", default="WorldView3", required=False, help="QuickBird, WorldView3, GaoFen2")
    parser.add_argument("--model", default="Base1", required=False)
    parser.add_argument("--cuda_id", default=0, type=int, required=False)

    parser.add_argument("--save_checkpoint_path", default="saved_models", type=str, required=False)
    parser.add_argument("--num_workers", default=16, type=int, required=False)
    parser.add_argument("--batch_size", default=32, type=int, required=False)
    parser.add_argument("--epochs", default=100, type=int, required=False)
    parser.add_argument("--device", default="cuda", type=str, required=False)
    parser.add_argument("--lr", default=1e-3, type=float, required=False)
    parser.add_argument("--gamma", default=0.5, type=float, required=False, help="学习率衰减比率")
    parser.add_argument("--weight_decay", default=1e-5, type=float, required=False)
    parser.add_argument("--lr_decay_steps", default=25, type=int, required=False)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, required=False)
    parser.add_argument("--prenorm", default=False, type=bool, required=False)
    parser.add_argument("--beta", default=0.5, type=float, required=False)
    parser.add_argument("--save_checkpoint_steps", default=10, type=int, required=False)

    parser.add_argument("--resume", default=False, type=bool, required=False)

    args = parser.parse_args()

    return args


def train_one_epoch(model, train_dataloader,
                    criterion, optimizer, scheduler,
                    epoch, args=None): # Removed logger
    model.train()
    epoch_start_time = datetime.now()
    total_loss = 0

    for batch_idx, batch in enumerate(train_dataloader):
        gt = batch["gt"].to(args.device)
        ms = batch["ms"].to(args.device)
        lms = batch["lms"].to(args.device)
        pan = batch["pan"].to(args.device)

        fused, spa_rec = model(pan, lms, ms)
        loss = criterion(fused, gt, spa_rec, pan)
        total_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_finish_time = datetime.now()
    # Log training loss to wandb
    wandb.log({"train/loss": epoch_mean_loss, "epoch": epoch})
    print(f"Epoch {epoch}: training loss {epoch_mean_loss:.4f} || Training time for this epoch: {epoch_finish_time - epoch_start_time}")


def validate_one_epoch(model, val_dataloader,
                       criterion, epoch, # Removed logger
                       args=None):
    model.eval()
    epoch_start_time = datetime.now()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            gt = batch["gt"].to(args.device)
            ms = batch["ms"].to(args.device)
            lms = batch["lms"].to(args.device)
            pan = batch["pan"].to(args.device)

            fused, spa_rec = model(pan, lms, ms)
            loss = criterion(fused, gt, spa_rec, pan)

            total_loss += loss.item()

        epoch_mean_loss = total_loss / len(val_dataloader)

        epoch_finish_time = datetime.now()
        # Log validation loss to wandb
        wandb.log({"val/loss": epoch_mean_loss, "epoch": epoch})
        print(f"Epoch {epoch}: validating loss {epoch_mean_loss:.4f} || Validating time for this epoch: {epoch_finish_time - epoch_start_time}")


        return epoch_mean_loss

def train(model, config, train_dataloader, val_dataloader, args, criterion, optimizer, scheduler): # Removed logger
    print('Start training...')

    best_val_loss = 10000000
    start_epoch = 1

    for epoch in range(start_epoch, args.epochs+1):
        # ========== train ========== #
        train_one_epoch(
            model=model, train_dataloader=train_dataloader,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, args=args
        )

        # ========== validate ========== #
        validate_loss = validate_one_epoch(
            model=model, val_dataloader=val_dataloader,
            criterion=criterion,
            epoch=epoch, args=args
        )

        # save model
        if epoch % args.save_checkpoint_steps == 0:
            model_info = f"{args.model}" + \
                         f"_{config.num_blocks}_{config.patch_size}_{config.window_size}_{config.model_dim}_{config.hidden_ch}" + \
                         f"_cudaid{args.cuda_id}"
            model_path = os.path.join(args.save_checkpoint_path, args.dataset, model_info)
            save_checkpoint(model, optimizer, scheduler, epoch, validate_loss, model_path, args, config, epoch)
            # Log checkpoint to wandb as an artifact
            wandb.save(os.path.join(model_path, f"epoch{epoch}.pth"))


        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            print('Saving current best model for epoch {}'.format(epoch))
            model_info = f"{args.model}" + \
                         f"_{config.num_blocks}_{config.patch_size}_{config.window_size}_{config.model_dim}_{config.hidden_ch}" + \
                         f"_cudaid{args.cuda_id}"
            model_path = os.path.join(args.save_checkpoint_path, args.dataset, model_info)

            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, model_path, args, config, "best")
            # Log best checkpoint to wandb as an artifact
            wandb.save(os.path.join(model_path, f"epochbest.pth"))

    print('Training finished!')

def save_checkpoint(model, optimizer, scheduler, epoch, loss, path, args, config, *save_type):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }, os.path.join(path, f"epoch{save_type[0]}.pth"))

def main():
    args = set_args()  # args for other hyperparams of training procedure
    config = select_config(args.dataset)  # dataset specific config for model

    wandb.init(
        project="pansharpening", # Your project name
        config={
            **vars(args), # Log all argparse arguments
            "num_blocks": config.num_blocks,
            "patch_size": config.patch_size,
            "window_size": config.window_size,
            "model_dim": config.model_dim,
            "hidden_ch": config.hidden_ch,
            "lms_ch": config.lms_ch,
            "dr": config.dr,
        },
        name=f"{args.dataset}_{args.model}_beta{args.beta}_cuda{args.cuda_id}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    args.logger_path = args.logger_path + f"_{args.dataset}_{args.model}" + \
                       f"_{config.num_blocks}_{config.patch_size}_{config.window_size}_{config.model_dim}_{config.hidden_ch}" + \
                       f"_cudaid{args.cuda_id}"
    seed_all(args.seed)
    args.device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")

    print("args:{}".format(args))

    train_dataset = PansharpeningDataset(h5_file_path=config.trainset_path, norm=args.prenorm, dr=config.dr)
    val_dataset = PansharpeningDataset(h5_file_path=config.validset_path, norm=args.prenorm, dr=config.dr)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = select_model(args.model, config)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.gamma)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Number of trainable params: %.2f million." % (n_parameters / 1.e6))
    print("Train on %s." % (args.device))

    criterion = Criterion(config.lms_ch, config.dr, args.beta)

    train(model, config, train_dataloader, val_dataloader, args, criterion, optimizer, scheduler)

    wandb.finish() # Call wandb.finish() at the end of the script


if __name__ == "__main__":
    main()