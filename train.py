import argparse
import dataclasses
import datetime
import pathlib
import random
import time

import numpy as np
import torch
import tqdm

from torch.utils.data import DataLoader

from language_detection.data import load_wili_2018_dataset, batch_collate_function, get_mask_from_lengths
from language_detection.model import TrainingConfig, TransformerClassifier, create_datasets, evaluate_model


DEFAULT_DATASET = "./datasets/WiLi_2018"
DEFALT_SAVEPATH = "./experiments"

parser = argparse.ArgumentParser(description="train a language detection transformer classifier")
parser.add_argument("--accumulate_steps", type=int, default=4, help="number of gradient accumulation steps, default 4")
parser.add_argument("--batch_size", type=int, default=32, help="batch size, default 32")
parser.add_argument("--clip_grad_norm", type=float, default=8, help="gradient norm clipping value, default 8")
parser.add_argument("--data_path", type=str, default=DEFAULT_DATASET, help=f"path to data, default {DEFAULT_DATASET}")
parser.add_argument("--debug", action="store_true", help="run in debug mode, which only runs a few steps per epoch")
parser.add_argument("--dev_pct", type=float, default=0.1, help="percent of train data withheld as dev, default 0.1")
parser.add_argument("--disp_loss_win", type=int, default=5, help="running loss window average n, default 5")
parser.add_argument("--init_lr", type=float, default=0.00001, help="initial learning rate, default 0.00001")
parser.add_argument("--max_length", type=int, default=1024, help="maximum input sequence length, default 1024")
parser.add_argument("--max_lr", type=float, default=0.001, help="maximum learning rate, default 0.001")
parser.add_argument("--save_base", type=str, default=DEFALT_SAVEPATH, help=f"checkpoint dir, default {DEFALT_SAVEPATH}")
parser.add_argument("--seed", type=int, default=0, help="seed for random number generation, default '0'")
parser.add_argument("--total_epochs", type=int, default=20, help="total number of training epochs, default 20")
parser.add_argument("--trial_name", type=str, default="wili2018", help="name of the model prefix, default 'wili2018'")
parser.add_argument("--warmup_pct", type=float, default=0.1, help="warmup epochs percentage, default 0.1")
args = parser.parse_args()

config = TrainingConfig(**vars(args))

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

# setup data
save_path = pathlib.PurePath(config.save_base, config.trial_name)
print(f"\nsaving checkpoints to path '{save_path}'")
pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

print(f"\nloading data from '{config.data_path}'")
raw_data = load_wili_2018_dataset(config.data_path)
if config.debug:
    print(f"debug mode is true, so truncate train, test to few elements only")
    raw_data.x_train = raw_data.x_train[:2048]
    raw_data.y_train = raw_data.y_train[:2048]
    raw_data.x_test = raw_data.x_test[:1024]
    raw_data.y_test = raw_data.y_test[:1024]

train_dataset, dev_dataset, test_dataset = create_datasets(
    raw_data, max_seq_len=config.max_length, dev_pct=config.dev_pct
)
train_dataloader = DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, collate_fn=batch_collate_function
)
dev_dataloader = DataLoader(
    dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=batch_collate_function
)
test_dataloader = DataLoader(
    test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=batch_collate_function
)
num_classes = len(raw_data.idx2lang)
print(f"sample count: train: {len(train_dataset)}, dev: {len(dev_dataset)}, test: {len(test_dataset)}")
print(f"total classes: {num_classes}")

# setup model
print(f"\nloading model with num_classes {num_classes}")
model = TransformerClassifier(num_classes=num_classes)
if torch.cuda.is_available():
    print(f"CUDA detected, using gpu")
    device_string = "cuda"
else:
    print(f"warning! no CUDA detected, using cpu")
    device_string = "cpu"
_ = model.to(device_string)


# initialize other objects and setup training
global_step = 0
ignore_index_value = -100
mlm_criterion = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=ignore_index_value)
clf_criterion = torch.nn.CrossEntropyLoss(reduction="sum")
optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.init_lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config.max_lr,
    epochs=config.total_epochs,
    steps_per_epoch=len(train_dataloader) // config.accumulate_steps,
    pct_start=config.warmup_pct,
)

# training loop
for epoch in range(config.total_epochs):
    # train one epoch
    train_iterator = iter(train_dataloader)
    epoch_losses = []
    model.train()
    print(f"[{datetime.datetime.now().isoformat()}] epoch {epoch+1} starting training")
    time.sleep(0.5)
    for batch_idx, minibatch in enumerate(pbar := tqdm.tqdm(train_iterator, total=len(train_dataloader))):
        # format data and move to gpu
        x, y, seq_lens, mask_indices, targets = minibatch
        x = x.to(device_string)
        y = y.to(device_string)
        targets = targets.to(device_string)
        pad_mask = get_mask_from_lengths(seq_lens, config.max_length, x.device)
        # model forward
        clf_logits, mlm_logits = model.forward(x, pad_mask)
        # calculate losses
        # for mlm, only backprop on the chosen (default 15%) indices
        masked_y = ignore_index_value * torch.ones_like(y).to(device_string)
        for i in range(y.shape[0]):
            masked_y[i, mask_indices[i].long()] = y[i, mask_indices[i].long()]
        mlm_loss = mlm_criterion(torch.transpose(mlm_logits, 1, 2), masked_y)
        clf_loss = clf_criterion(clf_logits, targets)
        global_step += 1
        # display loss running avg
        epoch_losses.append((clf_loss.item(), mlm_loss.item()))
        clf_losses = np.mean([l[0] for l in epoch_losses][-config.disp_loss_win :])
        mlm_losses = np.mean([l[1] for l in epoch_losses][-config.disp_loss_win :])
        pbar.set_description(f"step {global_step}: clf: {clf_losses:.3f}, mlm: {mlm_losses:.3f}")
        # backgroup w/gradient accumulation
        mlm_loss /= config.accumulate_steps
        clf_loss /= config.accumulate_steps
        ttl_loss = mlm_loss + clf_loss
        ttl_loss.backward()
        if (batch_idx > 0 and batch_idx % config.accumulate_steps == 0) or (batch_idx == len(train_dataloader)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)  # type: ignore
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # evaluate on withheld dev set
    print(f"[{datetime.datetime.now().isoformat()}] epoch {epoch+1} dev eval")
    time.sleep(0.5)
    model.eval()
    epoch_dev_loss: list[float] = []
    dev_epoch_targets = []
    dev_epoch_predictions = []
    with torch.no_grad():
        dev_iterator = iter(dev_dataloader)
        for batch_idx, minibatch in enumerate(pbar := tqdm.tqdm(dev_iterator, total=len(dev_iterator))):
            # format data and move to gpu
            x, y, seq_lens, mask_indices, targets = minibatch
            x = x.to(device_string)
            y = y.to(device_string)
            targets = targets.to(device_string)
            pad_mask = get_mask_from_lengths(seq_lens, config.max_length, x.device)
            clf_logits, mlm_logits = model.forward(x, pad_mask)
            # calculate loss (clf only for eval)
            clf_loss = clf_criterion(clf_logits, targets)
            epoch_dev_loss.append(clf_loss.item())
            dev_epoch_targets += targets.detach().cpu().numpy().tolist()
            dev_epoch_predictions += clf_logits.max(1).indices.detach().cpu().numpy().tolist()
        time.sleep(0.1)
        print(f"[{datetime.datetime.now().isoformat()}] dev clf loss : {np.mean(epoch_dev_loss):.5f}")
        dev_results = evaluate_model(set_name="dev", targets=dev_epoch_targets, predictions=dev_epoch_predictions)
        time.sleep(0.1)

        # save checkpoint
        checkpoint_name = str(pathlib.PurePath(save_path, f"{config.trial_name}-checkpoint-{epoch+1:06d}.pt"))
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
                "config": dataclasses.asdict(config),
                "results_dev": dev_results,
            },
            checkpoint_name,
        )
        print(f"[{datetime.datetime.now().isoformat()}] checkpoint saved to '{checkpoint_name}'")

# eval on test
print(f"[{datetime.datetime.now().isoformat()}] starting evaluation on test set")
model.eval()
epoch_test_loss = []
test_epoch_targets = []
test_epoch_predictions = []
with torch.no_grad():
    test_iterator = iter(test_dataloader)
    for batch_idx, minibatch in enumerate(pbar := tqdm.tqdm(test_iterator, total=len(test_iterator))):
        # format data and move to gpu
        x, y, seq_lens, mask_indices, targets = minibatch
        x = x.to(device_string)
        y = y.to(device_string)
        targets = targets.to(device_string)
        pad_mask = get_mask_from_lengths(seq_lens, config.max_length, x.device)
        # model forward
        clf_logits, mlm_logits = model.forward(x, pad_mask)
        # calculate loss (clf only for eval)
        clf_loss = clf_criterion(clf_logits, targets)
        epoch_test_loss.append(clf_loss.item())
        test_epoch_targets += targets.detach().cpu().numpy().tolist()
        test_epoch_predictions += clf_logits.max(1).indices.detach().cpu().numpy().tolist()
    time.sleep(0.1)
    print(f"[{datetime.datetime.now().isoformat()}] test clf loss : {np.mean(epoch_test_loss):.5f}")
    test_results = evaluate_model(set_name="test", targets=test_epoch_targets, predictions=test_epoch_predictions)
    time.sleep(0.1)
