from __future__ import absolute_import, division, print_function

import argparse
import csv
import os
import random
import pickle
import sys
import numpy as np
from typing import *

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, BCEWithLogitsLoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import AlbertTokenizer, BertTokenizer, XLNetTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from bert import MAG_BertForSequenceClassification
from xlnet import MAG_XLNetForSequenceClassification
from albert import MAG_AlbertForSequenceClassification

from argparse_utils import str2bool, seed
from global_configs import ACOUSTIC_DIM, VISUAL_DIM, DEVICE, DATASET_LOCATION

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    choices=["humor", "sarcasm"], default="humor")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument(
    "--model",
    type=str,
    choices=["bert-base-uncased", "xlnet-base-cased", "albert-base-v2"],
    default="bert-base-uncased",
)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=seed, default="random")


args = parser.parse_args()


def return_unk():
    return 0


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob


def get_inversion(tokens: List[str], SPIECE_MARKER="▁"):
    """
    Compute inversion indexes for list of tokens.

    Example:
        tokens = ["▁here", "▁is", "▁the", "▁sentence", "▁I", "▁want", "▁em", "bed", "ding", "s", "for"]
        inversions = get_inversion(tokens)
        inversions == [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 8]

    Args:
        tokens (List[str]): List of word tokens
        SPIECE_MARKER (str, optional): Special character to beginning of a single "word". Defaults to "▁".

    Returns:
        List[int]: Inversion indexes for each token
    """
    inversion_index = -1
    inversions = []
    for token in tokens:
        if SPIECE_MARKER in token:
            inversion_index += 1
        inversions.append(inversion_index)

    return inversions


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) == 0:
            tokens_b.pop()
        else:
            tokens_a.pop(0)


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):

        (
            (p_words, p_visual, p_acoustic, p_w_idx, p_concept_idx, p_vad, p_eigen),
            (c_words, c_visual, c_acoustic, c_w_idx, c_concept_idx, c_vad, c_eigen),
            hid,
            label,
        ) = example

        # add full stops
        c_words = ". ".join(c_words)
        p_words = p_words + "."

        tokens_a = tokenizer.tokenize(c_words)
        tokens_b = tokenizer.tokenize(p_words)

        inversions_a = get_inversion(tokens_a)
        inversions_b = get_inversion(tokens_b)

        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        if args.model == "bert-base-uncased" or args.model == 'albert-base-v2':
            prepare_text = preapre_bert_text_input
        elif args.model == "xlnet-base-cased":
            prepare_text = prepare_xlnet_text_input

        input_ids, input_mask, segment_ids = prepare_text(
            tokens_a, tokens_b, tokenizer, max_seq_length)

        inversions_a = inversions_a[-len(tokens_a):]
        inversions_b = inversions_b[:len(tokens_b)]

        visual_a = []
        acoustic_a = []
        for inv_id in inversions_a:
            visual_a.append(c_visual[inv_id, :])
            acoustic_a.append(c_acoustic[inv_id, :])

        visual_a = np.array(visual_a)
        acoustic_a = np.array(acoustic_a)

        visual_b = []
        acoustic_b = []
        for inv_id in inversions_b:
            visual_b.append(p_visual[inv_id, :])
            acoustic_b.append(p_acoustic[inv_id, :])

        visual_b = np.array(visual_b)
        acoustic_b = np.array(acoustic_b)

        if(len(tokens_a) > 0):
            visual = np.concatenate((visual_a, visual_b))
            acoustic = np.concatenate((acoustic_a, acoustic_b))
        else:
            visual = visual_b
            acoustic = acoustic_b

        visual = pad_multimodal_input(
            visual, input_ids, tokenizer, max_seq_length)
        acoustic = pad_multimodal_input(
            acoustic, input_ids, tokenizer, max_seq_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert acoustic.shape == (max_seq_length, ACOUSTIC_DIM)
        assert visual.shape == (max_seq_length, VISUAL_DIM)

        label_id = float(label)

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features


def preapre_bert_text_input(tokens_a, tokens_b, tokenizer, max_seq_length):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    PAD_ID = tokenizer.pad_token_id

    input_tokens = [CLS] + tokens_a + [SEP] + tokens_b + [SEP]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

    padding_len = (max_seq_length - len(input_ids))
    input_ids += [PAD_ID] * padding_len
    input_mask += [0] * padding_len
    segment_ids += [0] * padding_len

    return input_ids, input_mask, segment_ids


def prepare_xlnet_text_input(tokens_a, tokens_b, tokenizer, max_seq_length):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    PAD_ID = tokenizer.pad_token_id

    input_tokens = tokens_a + [SEP] + tokens_b + [SEP] + [CLS]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(tokens_a) + 1) + [1] * (len(tokens_b) + 1) + [2]

    padding_len = (max_seq_length - len(input_ids))

    input_ids = [PAD_ID] * padding_len + input_ids
    input_mask = [0] * padding_len + input_mask
    segment_ids = [3] * padding_len + segment_ids

    return input_ids, input_mask, segment_ids


def is_special_token_id(input_id, tokenizer):
    CLS_ID = tokenizer.cls_token_id
    PAD_ID = tokenizer.pad_token_id
    SEP_ID = tokenizer.sep_token_id

    return (input_id == CLS_ID or input_id == PAD_ID or input_id == SEP_ID)


def pad_multimodal_input(input, input_ids, tokenizer, max_seq_length):
    INPUT_DIM = input.shape[1]

    padded_input = np.zeros((max_seq_length, INPUT_DIM))
    input_index = 0

    # populate padded_input
    for index, id in enumerate(input_ids):
        # inject multimodal input if special token isn't present
        if not(is_special_token_id(id, tokenizer)):
            padded_input[index] = input[input_index]
            input_index += 1

    # ensure all inputs are consumed
    assert input_index == len(input)

    return padded_input


def get_tokenizer(model):
    if model == "bert-base-uncased":
        return BertTokenizer.from_pretrained(model)
    elif model == "xlnet-base-cased":
        return XLNetTokenizer.from_pretrained(model)
    elif model == "albert-base-v2":
        return AlbertTokenizer.from_pretrained(model)


def get_appropriate_dataset(data):

    tokenizer = get_tokenizer(args.model)

    features = convert_to_features(data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor(
        [f.acoustic for f in features], dtype=torch.float)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )
    return dataset


def set_up_data_loader():
    data_file = "all_mod_data_concept_vad_hem.pickle"
    with open(os.path.join(DATASET_LOCATION, args.dataset, data_file), "rb",) as handle:
        all_data = pickle.load(handle)
    train_data = all_data["train"]
    dev_data = all_data["dev"]
    test_data = all_data["test"]

    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step
        )
        * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prep_for_training(num_train_optimization_steps: int):
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
    )

    if args.model == "bert-base-uncased":
        model = MAG_BertForSequenceClassification.from_pretrained(
            args.model, multimodal_config=multimodal_config, num_labels=1,
        )
    elif args.model == "xlnet-base-cased":
        model = MAG_XLNetForSequenceClassification.from_pretrained(
            args.model, multimodal_config=multimodal_config, num_labels=1,
        )
    elif args.model == 'albert-base-v2':
        model = MAG_AlbertForSequenceClassification.from_pretrained(
            args.model, multimodal_config=multimodal_config, num_labels=1,
        )

    model.to(DEVICE)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_train_optimization_steps,
        num_training_steps=args.warmup_proportion * num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        outputs = model(
            input_ids,
            visual,
            acoustic,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=None,
        )
        logits = outputs[0]
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1), label_ids.view(-1))

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return tr_loss / nb_tr_steps


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader, optimizer):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
            logits = outputs[0]

            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )

            logits = torch.sigmoid(outputs[0])

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels


def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False):

    preds, y_test = test_epoch(model, test_dataloader)
    preds = preds.round()

    f_score = f1_score(y_test, preds, average="weighted")
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    acc = accuracy_score(y_test, preds)

    return acc, recall, precision, f_score


def train(
    model,
    train_dataloader,
    validation_dataloader,
    test_data_loader,
    optimizer,
    scheduler,
):
    valid_losses = []
    test_accuracies = []

    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, validation_dataloader, optimizer)
        test_acc, test_recall, test_precision, test_f_score = test_score_model(
            model, test_data_loader
        )

        print(
            "epoch:{}, train_loss:{}, valid_loss:{}, test_acc:{}".format(
                epoch_i, train_loss, valid_loss, test_acc
            )
        )

        valid_losses.append(valid_loss)
        test_accuracies.append(test_acc)

        wandb.log(
            (
                {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "test_acc": test_acc,
                    "test_f_score": test_f_score,
                    "test_recall": test_recall,
                    "test_precision": test_precision,
                    "best_valid_loss": min(valid_losses),
                    "best_test_acc": max(test_accuracies),
                }
            )
        )

    wandb.run.summary["best_valid_test_acc"] = test_accuracies[np.argmin(
        valid_losses)]


def main():
    wandb.init(project="MAG_HUMOR")
    wandb.config.update(args)
    set_random_seed(args.seed)

    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    model, optimizer, scheduler = prep_for_training(
        num_train_optimization_steps)

    train(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )


if __name__ == "__main__":
    main()
