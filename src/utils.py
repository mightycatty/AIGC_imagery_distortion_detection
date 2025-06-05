import time
import torch
from functools import wraps
import torch.distributed as dist
import wandb


def only_rank_0(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            if dist.get_rank() == 0:
                return func(*args, **kwargs)
        except:
            return func(*args, **kwargs)

    return wrapper


def ddp_print(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


import yaml


def read_defaults(yaml_path=None):
    if yaml_path is None:
        yaml_path = '../config.yaml'
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


@only_rank_0
def print_log_iter(optimizer, iter_counter, iter_loss, logger):
    current_lr = optimizer.param_groups[0]['lr']
    aver_loss_im_heatmap = sum(iter_loss[0]) / len(iter_loss[0])
    aver_loss_im_score = sum(iter_loss[1]) / len(iter_loss[1])
    aver_loss_sematic_heatmap = sum(iter_loss[2]) / len(iter_loss[2])
    # aver_loss_mis_score = sum(iter_loss[3]) / len(iter_loss[3])
    aver_loss_mis_heatmap, aver_loss_mis_score = 0, 0
    logger.info(f"Iteration {iter_counter}: Learning Rate = {current_lr}\n"
                f"Implausibility Heatmap Loss = {aver_loss_im_heatmap}, Implausibility Score Loss = {aver_loss_im_score}\n"
                f"Sematic Heatmap Loss = {aver_loss_sematic_heatmap}, Misalignment Score Loss = {aver_loss_mis_score}")
    print(f"Iteration {iter_counter}: Learning Rate = {current_lr}\n"
          f"Implausibility Heatmap Loss = {aver_loss_im_heatmap}, Implausibility Score Loss = {aver_loss_im_score}\n"
          f"Sematic Heatmap Loss = {aver_loss_sematic_heatmap}, Misalignment Score Loss = {aver_loss_mis_score}")
    wandb.log({'lr': current_lr, 'heatmap_loss': aver_loss_im_heatmap, 'score_loss': aver_loss_im_score},
              step=iter_counter)


def print_trainable_params(model):
    print('#' * 10 + 'Trainable parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


@only_rank_0
def save_in_training(model, optimizer, scheduler, iter_counter, save_path):
    checkpoint = {
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(checkpoint, f'{save_path}/{iter_counter}.pth')
    torch.save(checkpoint, f'{save_path}/last.pth')
    print(f"Model weights saved to {save_path}/{iter_counter}.pth")


@only_rank_0
def cprint(*args, **kwargs):
    print(*args, **kwargs)


@only_rank_0
def final_save(model, optimizer, scheduler, start_time, save_path):
    end_time = time.time()
    checkpoint = {
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    total_minutes = (end_time - start_time) / 60
    torch.save(checkpoint, f'{save_path}/last.pth')
    print(f"Model weights saved to {save_path}/last.pth. Total training time: {total_minutes:.2f} minutes")
