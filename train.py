import argparse
import datetime
import pathlib
import random
import time

import numpy as np
import torch
import tqdm

from torch.utils.data import DataLoader

from language_detection.data import load_wili_2018_dataset, batch_collate_function, get_mask_from_lengths
from language_detection.model import TransformerClassifier, create_datasets, evaluate_model


# todo: replace with loading from yaml/json for archiving trials
parser = argparse.ArgumentParser(description="train a language detection transformer classifier")
parser.add_argument("--max_length", type=int, default=1024, help="maximum input sequence length, default 1024")
parser.add_argument("--total_epochs", type=int, default=10, help="total number of training epochs, default 10")
parser.add_argument("--accumulate_steps", type=int, default=4, help="number of gradient accumulation steps, default 4")
parser.add_argument("--clip_grad_norm", type=int, default=4, help="gradient norm clipping value, default 5")
parser.add_argument("--warmup_pct", type=float, default=0.1, help="warmup epochs percentage, default 0.1")
parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate, default 0.0001")
parser.add_argument("--max_lr", type=float, default=0.001, help="maximum learning rate, default 0.001")
parser.add_argument("--disp_loss_win", type=int, default=5, help="running loss window average n, default 5")
parser.add_argument("--batch_size", type=int, default=32, help="batch size, default 32")
parser.add_argument(
    "--data_path", type=str, default="./datasets/WiLi_2018", help="path to data, default ./datasets/WiLi_2018"
)
parser.add_argument(
    "--save_path", type=str, default="./checkpoints", help="path to save the results, default './checkpoints'"
)
parser.add_argument("--trial_name", type=str, default="wili2018", help="name of the model prefix, default 'wili2018'")
parser.add_argument("--seed", type=int, default=1337, help="seed for random number generation, default '1337'")
args = parser.parse_args()

# todo: print args
max_length = args.max_length
total_epochs = args.total_epochs
accumulate_steps = args.accumulate_steps
clip_grad_norm = args.clip_grad_norm
warmup_pct = args.warmup_pct
init_lr = args.init_lr
max_lr = args.max_lr
disp_loss_win = args.disp_loss_win
batch_size = args.batch_size
data_path = args.data_path
save_path = args.save_path
trial_name = args.trial_name
seed = args.seed


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# setup data
pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

raw_data = load_wili_2018_dataset(data_path)

train_dataset, dev_dataset, test_dataset = create_datasets(raw_data, max_seq_len=1024, dev_pct=0.10)

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=batch_collate_function
)
dev_dataloader = DataLoader(
    dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=batch_collate_function
)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=batch_collate_function
)

num_classes = len(raw_data.idx2lang)
print(f"train: {len(train_dataset)}, dev: {len(dev_dataset)}, test: {len(test_dataset)}")

# setup model
# todo: check for cuda availability and set device properly
model = TransformerClassifier(num_classes=num_classes)
_ = model.to("cuda")
for name, param in model.named_parameters():
    if not str(param.device).startswith("cuda"):
        print(f"param '{name}' is on device '{param.device}'")
print(f"model loaded to cuda!")


# initialize other objects and setup training
global_step = 0
ignore_index_value = -100
mlm_criterion = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=ignore_index_value)
clf_criterion = torch.nn.CrossEntropyLoss(reduction="sum")
optimizer = torch.optim.AdamW(params=model.parameters(), lr=init_lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    epochs=total_epochs,
    steps_per_epoch=len(train_dataloader) // accumulate_steps,
    pct_start=warmup_pct,
)

# training loop
for epoch in range(total_epochs):
    # train one epoch
    train_iterator = iter(train_dataloader)
    epoch_losses = []
    model.train()
    print(f"[{datetime.datetime.now().isoformat()}] epoch {epoch+1} starting training")
    time.sleep(0.5)
    for batch_idx, minibatch in enumerate(pbar := tqdm.tqdm(train_iterator, total=len(train_dataloader))):
        # format data and move to gpu
        x, y, seq_lens, mask_indices, targets = minibatch
        x = x.to("cuda")
        y = y.to("cuda")
        targets = targets.to("cuda")
        pad_mask = get_mask_from_lengths(seq_lens, max_length, x.device)
        # model forward
        clf_logits, mlm_logits = model.forward(x, pad_mask)
        # calculate losses
        # for mlm, only backprop on the chosen (default 15%) indices
        masked_y = ignore_index_value * torch.ones_like(y).to("cuda")
        for i in range(y.shape[0]):
            masked_y[i, mask_indices[i].long()] = y[i, mask_indices[i].long()]
        mlm_loss = mlm_criterion(torch.transpose(mlm_logits, 1, 2), masked_y)
        clf_loss = clf_criterion(clf_logits, targets)
        global_step += 1
        # display loss running avg
        epoch_losses.append((clf_loss.item(), mlm_loss.item()))
        clf_losses = np.mean([l[0] for l in epoch_losses][-disp_loss_win:])
        mlm_losses = np.mean([l[1] for l in epoch_losses][-disp_loss_win:])
        pbar.set_description(f"step {global_step}: clf: {clf_losses:.3f}, mlm: {mlm_losses:.3f}")
        # backgroup w/gradient accumulation
        mlm_loss /= accumulate_steps
        clf_loss /= accumulate_steps
        ttl_loss = mlm_loss + clf_loss
        ttl_loss.backward()
        if (batch_idx > 0 and batch_idx % accumulate_steps == 0) or (batch_idx == len(train_dataloader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)  # type: ignore
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # evaluate on withheld dev set
    print(f"[{datetime.datetime.now().isoformat()}] epoch {epoch+1} dev eval")
    time.sleep(0.5)
    model.eval()
    epoch_dev_loss: list[float] = []
    epoch_targets = []
    epoch_predictions = []
    with torch.no_grad():
        dev_iterator = iter(dev_dataloader)
        for batch_idx, minibatch in enumerate(pbar := tqdm.tqdm(dev_iterator, total=len(dev_iterator))):
            # format data and move to gpu
            x, y, seq_lens, mask_indices, targets = minibatch
            x = x.to("cuda")
            y = y.to("cuda")
            targets = targets.to("cuda")
            pad_mask = get_mask_from_lengths(seq_lens, max_length, x.device)
            clf_logits, mlm_logits = model.forward(x, pad_mask)
            # calculate loss (clf only for eval)
            clf_loss = clf_criterion(clf_logits, targets)
            epoch_dev_loss.append(clf_loss.item())
            epoch_targets += targets.detach().cpu().numpy()
            epoch_predictions += clf_logits.max(1).indices.detach().cpu().numpy()
        time.sleep(0.1)
        print(f"[{datetime.datetime.now().isoformat()}] dev clf loss : {np.mean(epoch_dev_loss):.5f}")
        dev_results = evaluate_model(set_name="dev", targets=epoch_targets, predictions=epoch_predictions)
        time.sleep(0.1)

        # save checkpoint
        # todo: make functions for saving, loading
        checkpoint_name = str(pathlib.PurePath(save_path, f"{trial_name}-checkpoint-{global_step:08d}.pt"))
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch_losses": epoch_losses,
                "output_mapping": raw_data.idx2lang,
                "extended_labels": raw_data.labels,
                "config": {
                    "max_length": max_length,
                    "total_epochs": total_epochs,
                    "accumulate_steps": accumulate_steps,
                    "init_lr": init_lr,
                    "max_lr": max_lr,
                    "warmup_pct": warmup_pct,
                    "disp_loss_win": disp_loss_win,
                    "batch_size": batch_size,
                    "data_path": data_path,
                    "save_path": save_path,
                    "trial_name": trial_name,
                    "seed": seed,
                },
                "results": dev_results,
            },
            checkpoint_name,
        )
        print(f"[{datetime.datetime.now().isoformat()}] checkpoint saved to '{checkpoint_name}'")

# eval on test
print(f"[{datetime.datetime.now().isoformat()}] starting evaluation on test set")
model.eval()
epoch_test_loss = []
epoch_targets = []
epoch_predictions = []
with torch.no_grad():
    test_iterator = iter(test_dataloader)
    for batch_idx, minibatch in enumerate(pbar := tqdm.tqdm(test_iterator, total=len(test_iterator))):
        # format data and move to gpu
        x, y, seq_lens, mask_indices, targets = minibatch
        x = x.to("cuda")
        y = y.to("cuda")
        targets = targets.to("cuda")
        pad_mask = get_mask_from_lengths(seq_lens, max_length, x.device)
        # model forward
        clf_logits, mlm_logits = model.forward(x, pad_mask)
        # calculate loss (clf only for eval)
        clf_loss = clf_criterion(clf_logits, targets)
        epoch_test_loss.append(clf_loss.item())
        epoch_targets += targets.detach().cpu().numpy()
        epoch_predictions += clf_logits.max(1).indices.detach().cpu().numpy()
    time.sleep(0.1)
    print(f"[{datetime.datetime.now().isoformat()}] test clf loss : {np.mean(epoch_test_loss):.5f}")
    test_results = evaluate_model(set_name="test", targets=epoch_targets, predictions=epoch_predictions)
    time.sleep(0.1)
