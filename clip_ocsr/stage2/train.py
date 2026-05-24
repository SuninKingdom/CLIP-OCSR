"""Stage 2 training: Supervised fine-tuning of CLIP-OCSR for image-to-SMILES translation.

Usage:
    torchrun --nproc_per_node=1 -m clip_ocsr.stage2.train --config configs/stage2_finetune.yaml
    torchrun --nproc_per_node=2 -m clip_ocsr.stage2.train --config configs/stage2_finetune.yaml
"""

import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, random_split
import torch.optim.lr_scheduler as lr_scheduler

import pandas as pd
from tokenizers import Tokenizer, pre_tokenizers, trainers
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from rdkit import Chem
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from clip_ocsr.stage2.model import build_smilesmodel
from clip_ocsr.stage2.dataset import SmilesDataset, causal_mask
from clip_ocsr.stage2.config import Stage2Config, get_weights_file_path, latest_weights_file_path
from clip_ocsr.evaluation.metrics import calculate_acc, calculate_tanimoto_similarity
from clip_ocsr.utils.abbrev_group import abbrevgroup2smiles
from clip_ocsr.utils.seed import set_seed


# Global GPU index for DDP
gpu = 0


def init_distributed_mode():
    """Initialize distributed training environment."""
    global gpu
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = -1
        world_size = -1
        gpu = 0
    torch.cuda.set_device(gpu)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    dist.barrier()
    setup_for_distributed(rank == 0)


def setup_for_distributed(is_master):
    """Override print to only output on the master process."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print


def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    """Batch-compatible greedy decoding for image-to-SMILES generation.

    Args:
        model: DDP-wrapped SmilesModel.
        source: Input image tensor (batch, 3, H, W).
        source_mask: Source mask (unused, kept for API compatibility).
        tokenizer_tgt: SMILES tokenizer.
        max_len: Maximum decoding length.
        device: Target device.

    Returns:
        Decoded token sequences (batch, seq_len).
    """
    batch_size = source.size(0)
    sos_idx = tokenizer_tgt.token_to_id('<sos>')
    eos_idx = tokenizer_tgt.token_to_id('<eos>')

    decoder_input = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    while True:
        if decoder_input.size(1) >= max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).to(device)

        prob = model.module(source, decoder_input, decoder_mask, return_last_token=True)

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.long().to(device)

        finished |= (next_word == eos_idx)
        next_word[finished] = eos_idx
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(1)], dim=1)

        if finished.all():
            break

    return decoder_input


def get_all_smiles_from_dataframe(dataframe, column_name='SMILES'):
    """Yield SMILES strings from a DataFrame column."""
    for smiles in dataframe[column_name]:
        yield smiles


def get_or_build_smiles_tokenizer(tokenizer_path, dataframe, column_name='SMILES'):
    """Load or build a character-level SMILES tokenizer.

    Args:
        tokenizer_path: Path to save/load the tokenizer JSON file.
        dataframe: DataFrame containing SMILES strings.
        column_name: Name of the SMILES column.

    Returns:
        HuggingFace Tokenizer object.
    """
    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token='<unk>'))
        tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern='', behavior='isolated')
        trainer = trainers.WordLevelTrainer(
            special_tokens=["<unk>", "<pad>", "<sos>", "<eos>"],
            min_frequency=2
        )
        tokenizer.train_from_iterator(
            get_all_smiles_from_dataframe(dataframe, column_name),
            trainer=trainer
        )
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(dataframe_path, img_folder, training_ratio, tokenizer_path, seq_len, batch_size):
    """Load data and create DataLoaders.

    Returns:
        Tuple of (train_loader, val_loader, tokenizer_tgt).
    """
    dataframe = pd.read_csv(dataframe_path)
    tokenizer_path = Path(tokenizer_path)

    tokenizer_tgt = get_or_build_smiles_tokenizer(tokenizer_path, dataframe)

    train_ds_size = int(training_ratio * len(dataframe))
    val_ds_size = len(dataframe) - train_ds_size
    train_dataframe, val_dataframe = random_split(dataframe, [train_ds_size, val_ds_size])

    train_dataframe = dataframe.iloc[train_dataframe.indices]
    val_dataframe = dataframe.iloc[val_dataframe.indices]

    train_ds = SmilesDataset(train_dataframe, img_folder, tokenizer_tgt, seq_len)
    valid_ds = SmilesDataset(val_dataframe, img_folder, tokenizer_tgt, seq_len)

    max_len_tgt = 0
    for item in dataframe["SMILES"]:
        tgt_ids = tokenizer_tgt.encode(item).ids
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    print(f'Max length of target sentence: {max_len_tgt}')

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    valid_sampler = DistributedSampler(valid_ds, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=32, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, sampler=valid_sampler,
                              num_workers=32, pin_memory=True)

    return train_loader, valid_loader, tokenizer_tgt


def get_model(clip_ckpt_path, trainable, tgt_vocab_size, tgt_seq_len, d_model, N, h, dropout, d_ff):
    """Build and move the SmilesModel to the appropriate GPU."""
    model = build_smilesmodel(clip_ckpt_path, trainable, tgt_vocab_size, tgt_seq_len,
                              d_model, N, h, dropout, d_ff)
    model = model.to(gpu)
    return model


def is_main_process():
    """Check if this is the main (rank 0) process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def run_validation(model, validation_ds, tokenizer_tgt, max_len, device, print_msg,
                   epoch, writer, abbrev_group_path, num_examples=2):
    """Run validation and compute accuracy metrics.

    Args:
        model: DDP-wrapped SmilesModel.
        validation_ds: Validation DataLoader.
        tokenizer_tgt: SMILES tokenizer.
        max_len: Maximum decoding length.
        device: Target device.
        print_msg: Print function (from tqdm).
        epoch: Current epoch number.
        writer: TensorBoard SummaryWriter.
        abbrev_group_path: Path to abbreviation group JSON file.
        num_examples: Number of examples to print.
    """
    model.eval()
    count = 0
    total_ide = 0
    total_smi = 0
    count_num = 0

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    torch.cuda.empty_cache()

    with torch.no_grad():
        for batch in validation_ds:
            encoder_input = batch["encoder_input"].to(device)

            model_out = greedy_decode(model, encoder_input, None, tokenizer_tgt, max_len, device)

            target_texts = batch["tgt_text"]
            model_out_texts = [tokenizer_tgt.decode(out.detach().cpu().numpy()) for out in model_out]

            print_msg('-' * console_width)
            for target_text, model_out_text in zip(target_texts, model_out_texts):
                count_num += 1
                smiles_pred = model_out_text.replace(" ", "")
                smiles_unfold = abbrevgroup2smiles(smiles_pred, abbrev_group_path)
                mol_pred = Chem.MolFromSmiles(smiles_unfold)
                target_unfold = abbrevgroup2smiles(target_text, abbrev_group_path)
                tgt_mol = Chem.MolFromSmiles(target_unfold)
                if tgt_mol is not None:
                    count += 1
                    ide = calculate_acc(mol_pred, tgt_mol)
                    smi = calculate_tanimoto_similarity(mol_pred, tgt_mol)
                    total_ide += ide
                    total_smi += smi
                print_msg(f"{'TARGET: ':>12}{target_text}")
                print_msg(f"{'PREDICTED: ':>12}{smiles_pred}")

            if count_num >= num_examples:
                print_msg('-' * console_width)
                break

        avg_acc = total_ide / (count + 1)
        avg_sim = total_smi / (count + 1)

        if is_main_process():
            writer.add_scalars('Validation Metrics',
                             {'Accuracy': avg_acc, 'Tanimoto Similarity': avg_sim}, epoch)
            writer.flush()

        print(f"Validation acc at epoch {epoch}: {avg_acc}")
        print(f"Validation tanimoto similarity at epoch {epoch}: {avg_sim}")


def train_model(cfg: Stage2Config):
    """Main training function for Stage 2.

    Args:
        cfg: Stage2Config instance with all training parameters.
    """
    global gpu

    print(f'Using GPU {gpu}')
    Path(cfg.model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_tgt = get_ds(
        cfg.dataframe_path, cfg.img_folder, cfg.training_ratio,
        cfg.tokenizer_path, cfg.seq_len, cfg.batch_size
    )

    model = get_model(
        cfg.clip_ckpt_path, cfg.trainable, tokenizer_tgt.get_vocab_size(),
        cfg.tgt_seq_len, cfg.d_model, cfg.N, cfg.h, cfg.dropout, cfg.d_ff
    )

    writer = SummaryWriter(cfg.tb_folder)
    initial_epoch = 0
    global_step = 0
    lr = cfg.lr

    model_filename = (latest_weights_file_path(cfg.model_folder, cfg.model_basename)
                      if cfg.preload == 'latest'
                      else get_weights_file_path(cfg.model_folder, cfg.model_basename, cfg.preload)
                      if cfg.preload else None)

    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
        lr = state['lr']
        model = DDP(model, device_ids=[gpu])
        optimizer = torch.optim.Adam(model.parameters(), lr, eps=1e-9)
        optimizer.load_state_dict(state['optimizer_state_dict'])

        T_max = cfg.num_epochs / 4
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        scheduler.load_state_dict(state['scheduler_state_dict'])
        scheduler.step()
    else:
        print('No model to preload, starting from scratch')
        model = DDP(model, device_ids=[gpu])
        optimizer = torch.optim.Adam(model.parameters(), lr, eps=1e-9)

        T_max = cfg.num_epochs / 4
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id('<pad>'),
        label_smoothing=0.1
    ).to(gpu)

    for epoch in range(initial_epoch, cfg.num_epochs):
        torch.cuda.empty_cache()

        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)

        model.train()
        total_train_loss = 0
        total_batches = 0

        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(gpu)
            decoder_input = batch['decoder_input'].to(gpu)
            decoder_mask = batch['decoder_mask'].to(gpu)

            proj_output = model.module.forward(encoder_input, decoder_input, decoder_mask)
            label = batch['label'].to(gpu)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            if is_main_process():
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            total_train_loss += loss.item()
            total_batches += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        avg_train_loss = total_train_loss / total_batches

        if is_main_process():
            writer.add_scalar('train_loss', avg_train_loss, epoch)
            writer.flush()

        run_validation(
            model, val_dataloader, tokenizer_tgt, cfg.seq_len, gpu,
            lambda msg: batch_iterator.write(msg), epoch, writer,
            cfg.abbrev_group_path, len(val_dataloader.dataset)
        )

        # Save checkpoint
        model_filename = cfg.model_folder + "/" + cfg.model_basename + f"{epoch:02d}" + ".pt"
        if dist.get_rank() == 0:
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
                'lr': scheduler.get_last_lr()[0]
            }, model_filename)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{cfg.num_epochs}, Current LR: {current_lr}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: CLIP-OCSR fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Load config
    cfg = Stage2Config.from_yaml(args.config)

    # Set seed
    set_seed(42)

    # Initialize distributed training
    init_distributed_mode()

    # Start training
    train_model(cfg)


if __name__ == '__main__':
    main()
