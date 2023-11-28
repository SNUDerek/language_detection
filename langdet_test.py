import argparse
import datetime
import pathlib
import random
from tabnanny import check
import time

import numpy as np
import torch
import tqdm

from language_detection.data import DataLoader, load_wili_2018_dataset, batch_collate_function, get_mask_from_lengths
from language_detection.model import TrainingConfig, TransformerClassifier, create_datasets, evaluate_model
from language_detection.model.utils import export_results_tsv


DEFAULT_DATASET = "./datasets/WiLi_2018"
DEFALT_SAVEPATH = "./experiments"

parser = argparse.ArgumentParser(description="train a language detection transformer classifier")
parser.add_argument("--checkpoint_file", type=str, required=True, help=f"filepath to checkpoint file")
parser.add_argument("--batch_size", type=int, default=32, help="batch size, default 32")
parser.add_argument("--data_path", type=str, default=DEFAULT_DATASET, help=f"path to data, default {DEFAULT_DATASET}")
parser.add_argument("--out_file", type=str, default="testset.tsv", help="filepath for output, default 'testset.tsv'")
args = parser.parse_args()


if not pathlib.Path(args.checkpoint_file).is_file():
    raise ValueError(f"checkpoint file '{args.checkpoint_file}' does not exist!")
checkpoint = torch.load(args.checkpoint_file)
config = TrainingConfig(**checkpoint["config"])
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

print(f"\nloading model from checkpoint '{checkpoint}'")
model = TransformerClassifier(num_classes=checkpoint["num_classes"])
model.load_state_dict(checkpoint["model_state_dict"])
if torch.cuda.is_available():
    print(f"CUDA detected, using gpu")
    device_string = "cuda"
else:
    print(f"warning! no CUDA detected, using cpu")
    device_string = "cpu"
_ = model.to(device_string)

print(f"\nloading data from '{config.data_path}'")
raw_data = load_wili_2018_dataset(config.data_path)
if config.debug:
    print(f"debug mode is true, so truncate test to few elements only")
    raw_data.x_test = raw_data.x_test[:1024]
    raw_data.y_test = raw_data.y_test[:1024]

_trn, _dev, test_dataset = create_datasets(raw_data, max_seq_len=config.max_length, dev_pct=config.dev_pct)
test_dataloader = DataLoader(
    test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=batch_collate_function
)
num_classes = len(raw_data.idx2lang)
if checkpoint["num_classes"] != num_classes:
    raise ValueError(
        f"model's {checkpoint['num_classes']} output classes != data's {num_classes} classes, is this the correct dataset?"
    )

# eval on test
clf_criterion = torch.nn.CrossEntropyLoss(reduction="sum")
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
        clf_logits, mlm_logits = model.forward(x, pad_mask)
        clf_loss = clf_criterion(clf_logits, targets)
        epoch_test_loss.append(clf_loss.item())
        test_epoch_targets += targets.detach().cpu().numpy().tolist()
        test_epoch_predictions += clf_logits.max(1).indices.detach().cpu().numpy().tolist()
    time.sleep(0.1)
    print(f"[{datetime.datetime.now().isoformat()}] test clf loss : {np.mean(epoch_test_loss):.5f}")
    test_results = evaluate_model(set_name="test", targets=test_epoch_targets, predictions=test_epoch_predictions)
    time.sleep(0.1)

print(f"writing test results to '{args.out_file}'")
export_results_tsv(
    samples=raw_data.x_test,
    targets=test_epoch_targets,
    predictions=test_epoch_predictions,
    idx_mapping=raw_data.idx2lang,
    output_filename=args.out_file,
    desc_mapping=raw_data.labels,
)
