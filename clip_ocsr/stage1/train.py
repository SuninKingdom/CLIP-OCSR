"""Stage 1 training: Domain-adaptive pretraining of CLIP-RN50 on chemical structures.

Usage:
    torchrun --nproc_per_node=1 -m clip_ocsr.stage1.train --config configs/stage1_pretrain.yaml
    torchrun --nproc_per_node=2 -m clip_ocsr.stage1.train --config configs/stage1_pretrain.yaml
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

import clip
from PIL import Image

from clip_ocsr.stage1.config import Stage1Config
from clip_ocsr.stage1.dataset import CLIPFinetuneDataset, get_tokenizer
from clip_ocsr.utils.seed import set_seed


def setup_distributed():
    """Set up distributed training environment (compatible with torchrun).

    Returns:
        Tuple of (rank, world_size, local_rank, device).
    """
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    device = torch.device('cpu')

    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        if rank == 0:
            print(f"Distributed training initialized: world_size={world_size}, rank={rank}, local_rank={local_rank}")
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        if rank == 0:
            print("Single GPU training mode")
    else:
        if rank == 0:
            print("CPU training mode")

    return rank, world_size, local_rank, device


@torch.no_grad()
def all_gather(tensor, world_size):
    """Gather tensors from all GPUs (for contrastive loss in DDP)."""
    if world_size == 1:
        return tensor

    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)


def compute_loss(image_features, text_features, logit_scale):
    """Compute symmetric cross-entropy contrastive loss.

    Args:
        image_features: Image feature vectors (global_batch_size, dim).
        text_features: Text feature vectors (global_batch_size, dim).
        logit_scale: Learnable temperature parameter.

    Returns:
        Scalar loss value.
    """
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    scale = logit_scale.exp()

    logits_per_image = scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device, dtype=torch.long)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t) / 2.0

    return loss


def get_loaders(df, preprocess, tokenizer, rank, world_size, cfg):
    """Create training and validation DataLoaders.

    Args:
        df: DataFrame with 'file_path' and 'caption_nl' columns.
        preprocess: CLIP image preprocess transform.
        tokenizer: clip.tokenize function.
        rank: Current process rank.
        world_size: Total number of processes.
        cfg: Stage1Config instance.

    Returns:
        Tuple of (train_loader, val_loader, train_sampler).
    """
    train_df, val_df = train_test_split(
        df,
        train_size=cfg.train_split_ratio,
        random_state=cfg.seed,
        shuffle=True
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_dataset = CLIPFinetuneDataset(train_df, cfg.data_dir, preprocess, tokenizer)
    val_dataset = CLIPFinetuneDataset(val_df, cfg.data_dir, preprocess, tokenizer)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=cfg.seed
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, train_sampler


def save_checkpoint(model, optimizer, scheduler, epoch, loss, rank, cfg):
    """Save model checkpoint (only on rank 0)."""
    if rank != 0:
        return

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model

    filename = os.path.join(cfg.checkpoint_dir, f"{cfg.model_basename}_epoch_{epoch+1}.pt")

    checkpoint_state = {
        'epoch': epoch + 1,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }

    torch.save(checkpoint_state, filename)
    print(f"Checkpoint saved to {filename}")


def train_one_epoch(model, loader, optimizer, scheduler, device, world_size, epoch, cfg, rank):
    """Train for one epoch.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} [Train]", disable=(rank != 0))

    model_core = model.module if world_size > 1 and hasattr(model, "module") else model

    current_lr = optimizer.param_groups[0]['lr']

    for batch in pbar:
        images = batch['image'].to(device, non_blocking=True)
        texts = batch['text'].to(device, non_blocking=True)

        # Forward pass
        image_features = model_core.encode_image(images)
        text_features = model_core.encode_text(texts)

        # Gather features across GPUs for global contrastive loss
        gathered_image_features = all_gather(image_features, world_size)
        gathered_text_features = all_gather(text_features, world_size)

        # Compute global loss
        loss = compute_loss(gathered_image_features, gathered_text_features, model_core.logit_scale)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if rank == 0:
            pbar.set_postfix({"loss": loss.item(), "lr": f"{current_lr:.9f}"})

    scheduler.step()

    return total_loss / len(loader)


def validate_one_epoch(model, loader, device, world_size, epoch, cfg, rank):
    """Validate for one epoch.

    Returns:
        Average validation loss for the epoch.
    """
    model.eval()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} [Valid]", disable=(rank != 0))

    model_core = model.module if world_size > 1 and hasattr(model, "module") else model

    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device, non_blocking=True)
            texts = batch['text'].to(device, non_blocking=True)

            image_features = model_core.encode_image(images)
            text_features = model_core.encode_text(texts)

            gathered_image_features = all_gather(image_features, world_size)
            gathered_text_features = all_gather(text_features, world_size)

            loss = compute_loss(gathered_image_features, gathered_text_features, model_core.logit_scale)

            total_loss += loss.item()

            if rank == 0:
                pbar.set_postfix({"loss": loss.item()})

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Stage 1: CLIP domain-adaptive pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Load config
    cfg = Stage1Config.from_yaml(args.config)

    # Set seed
    set_seed(cfg.seed)

    # Setup distributed training
    rank, world_size, local_rank, device = setup_distributed()

    # Load CLIP model
    if rank == 0:
        print(f"Loading CLIP model '{cfg.model_name}' from: {cfg.pretrained_path}")

    try:
        model, preprocess = clip.load(
            cfg.pretrained_path,
            device='cpu',
            jit=False
        )
        if rank == 0:
            print("Successfully loaded model from local .pt file.")
    except Exception as e:
        if rank == 0:
            print(f"Error loading local .pt file ({e}). Falling back to default download/cached model.")
        model, preprocess = clip.load(
            cfg.model_name,
            device='cpu',
            jit=False
        )
        if rank == 0:
            print(f"Warning: Using default or cached CLIP {cfg.model_name} weights.")

    model.to(device)

    # Get tokenizer
    tokenizer = get_tokenizer()

    # Load data
    if rank == 0:
        print(f"Loading CSV data from {cfg.csv_path}")
    df = pd.read_csv(cfg.csv_path)
    train_loader, val_loader, train_sampler = get_loaders(df, preprocess, tokenizer, rank, world_size, cfg)

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.scheduler_T_max
    )

    # DDP wrap
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # TensorBoard
    writer = None
    if rank == 0:
        writer = SummaryWriter(cfg.tensorboard_dir)
        print("TensorBoard logging enabled. Run: tensorboard --logdir=./runs")

    # Training loop
    if rank == 0:
        print("Starting training...")

    for epoch in range(cfg.num_epochs):
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, world_size, epoch, cfg, rank
        )

        avg_valid_loss = validate_one_epoch(
            model, val_loader, device, world_size, epoch, cfg, rank
        )

        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{cfg.num_epochs} - Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, LR: {current_lr:.9f}")

            if writer:
                writer.add_scalar("Loss/train", avg_train_loss, epoch)
                writer.add_scalar("Loss/valid", avg_valid_loss, epoch)
                writer.add_scalar("LearningRate", current_lr, epoch)

            if (epoch + 1) % cfg.save_every_epoch == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, avg_valid_loss,
                    rank, cfg
                )

    if writer and rank == 0:
        writer.close()
        print("Training complete.")


if __name__ == "__main__":
    main()
