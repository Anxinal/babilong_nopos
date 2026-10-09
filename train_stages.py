"""Train one model on BABILong in three curriculum stages, then test length extrapolation.

    python train_stages.py --model transformer_mask --encoder_mask CCCCFFFF --output_dir runs/vanilla
    python train_stages.py --model alibi --output_dir runs/alibi

The stages are ordered by how many facts an answer needs, and each adds tasks to the
ones before it, so the last stage trains on all 17:

    stage 1   one supporting fact        qa1 qa4 qa5 qa9 qa10 qa20
    stage 2   + two facts, or a tally    qa2 qa7 qa8 qa13 qa14 qa15 qa17 qa18
    stage 3   + three facts, or a path   qa3 qa16 qa19

qa6, qa11 and qa12 are left out: they ask the same question over the same stories as a
task that is kept (see ``STAGES``).

A sample is the bAbI facts hidden in PG19 text, followed by the question; the target is
the answer. Lengths are named as the benchmark names them: a sample of length L holds
``L - 300`` tokens of facts and noise, and 300 or less is the bare facts. Training draws
each sample's length uniformly from ``--train_len`` (0 to 2k). After every stage the
model is evaluated at each of ``--eval_lens``, which goes on to 4k and 8k, lengths it
never trained on. Writes ``stage<k>.pt`` after each stage and ``metrics.jsonl``.
``--start_stage 2`` continues from ``stage1.pt`` in the same folder.
"""

import argparse
import functools
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from babilong.babilong_utils import NoiseInjectionDataset, SentenceSampler
from tmodel.models import MODEL_TYPES, PE_TYPES, build_model

# Left out, because a kept task asks the same question over the same kind of story:
#   qa6  "Is P in the L?"  -- qa9 and qa10 ask it too, and add negation / "either ... or"
#   qa11 "Where is P?" with pronouns, qa12 with "P and P" -- qa13 has both
STAGES = [
    [1, 4, 5, 9, 10, 20],                 # one supporting fact
    [2, 7, 8, 13, 14, 15, 17, 18],        # two supporting facts, or a tally over several
    [3, 16, 19],                          # three supporting facts, or a path
]

# The benchmark's allowance for the prompt: its "2k" samples hold 1700 context tokens.
PROMPT_RESERVE = 300
# Room for the question on top of the context when nothing may be truncated.
QUESTION_ROOM = 64


def read_babi(path):
    """One dict per question: the facts stated before it, the question and the answer."""
    samples, facts = [], []
    with open(path) as f:
        for line in f:
            num, text = line.rstrip("\n").split(" ", 1)
            if int(num) == 1:
                facts = []
            if "\t" in text:
                question, answer, _ = text.split("\t")
                samples.append(dict(facts=list(facts), question=question.strip(), answer=answer))
            else:
                facts.append(text)
    return samples


class BabiTasks(Dataset):
    """The questions of several bAbI tasks, in the form NoiseInjectionDataset reads."""

    def __init__(self, babi_dir, tasks, split, max_n_facts=None, samples_per_task=None):
        self.samples = []
        for task in tasks:
            samples = read_babi(os.path.join(babi_dir, f"qa{task}_{split}.txt"))
            if max_n_facts is not None:     # as data/create_tasks.py does
                samples = [s for s in samples if len(s["facts"]) <= max_n_facts]
            for s in samples[:samples_per_task]:
                self.samples.append(dict(s, task=task))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return dict(self.samples[i])     # a copy: NoiseInjectionDataset adds keys to it


class BabilongPairs(Dataset):
    """(src, tgt, task) token tensors: facts in noise + question -> BOS answer EOS.

    Each sample's length is drawn uniformly from ``lens = (low, high)``. ``seed=None``
    draws differently every epoch (training); a seed fixes lengths and noise (evaluation).
    """

    def __init__(self, tasks, tokenizer, lens, src_len, noise=None, seed=None):
        self.tasks, self.tokenizer = tasks, tokenizer
        self.lens, self.src_len, self.seed = lens, src_len, seed
        self.rng = np.random.default_rng(seed)
        self.noisy = None
        if noise is not None:
            sampler = SentenceSampler(noise, tokenizer=tokenizer, shuffle=True, random_seed=seed)
            self.noisy = NoiseInjectionDataset(tasks, sampler, tokenizer)
            self.noisy.gen = self.rng

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, i):
        sample = self.tasks[i]
        encode = functools.partial(self.tokenizer.encode, add_special_tokens=False)
        context_len = int(self.rng.integers(self.lens[0], self.lens[1] + 1)) - PROMPT_RESERVE
        if self.noisy is not None and context_len > 0:
            self.noisy.sample_size = context_len
            context = self.noisy[i]["input_tokens"]
        else:
            context = encode(" " + " ".join(sample["facts"]))
        question = encode("\nQuestion: " + sample["question"] + "\nAnswer:")
        # Truncate on the left, so the question is never what gets cut.
        src = context[-(self.src_len - len(question)):] + question
        # Leading space: the answer then has the token it has inside the facts.
        tgt = [self.tokenizer.bos_token_id] + encode(" " + sample["answer"]) + [self.tokenizer.eos_token_id]
        return torch.tensor(src), torch.tensor(tgt), sample["task"]


def collate(batch, pad_id):
    srcs, tgts, tasks = zip(*batch)
    pad = functools.partial(nn.utils.rnn.pad_sequence, batch_first=True, padding_value=pad_id)
    return pad(srcs), pad(tgts), torch.tensor(tasks)


def reseed_worker(_):
    """Give each training DataLoader worker its own lengths and fact positions.

    The workers are copies of one dataset and would otherwise all draw the same numbers.
    Seeded (evaluation) datasets are left alone, so they stay reproducible.
    """
    ds = torch.utils.data.get_worker_info().dataset
    if ds.seed is None:
        ds.rng = np.random.default_rng()
        if ds.noisy is not None:
            ds.noisy.gen = ds.rng


def make_loader(args, tokenizer, tasks, split, noise, lens, samples_per_task, batch_size,
                src_len, seed=None):
    """A loader over *tasks* at lengths *lens*. ``seed=None`` is a shuffled training loader."""
    context_len = lens[1] - PROMPT_RESERVE
    babi = BabiTasks(args.babi_dir, tasks, split, context_len // 8 if context_len > 0 else None,
                     samples_per_task)
    ds = BabilongPairs(babi, tokenizer, lens, src_len, noise if context_len > 0 else None, seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=seed is None, num_workers=args.workers,
                      collate_fn=functools.partial(collate, pad_id=tokenizer.pad_token_id),
                      worker_init_fn=reseed_worker, pin_memory=True)


def forward(model, criterion, batch, device):
    src, tgt, tasks = batch
    src, tgt = src.to(device), tgt.to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        logits = model(src, tgt[:, :-1])
    logits, tgt_out = logits.float(), tgt[:, 1:]
    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
    return loss, logits, tgt_out, tasks


@torch.no_grad()
def evaluate(model, criterion, loader, device):
    """Mean loss, and per task the share of answers with every token right (teacher-forced)."""
    model.eval()
    losses, right, total = [], {}, {}
    for batch in loader:
        loss, logits, tgt_out, tasks = forward(model, criterion, batch, device)
        losses.append(loss.item())
        ignore = tgt_out.eq(criterion.ignore_index)
        correct = (logits.argmax(-1).eq(tgt_out) | ignore).all(dim=1).cpu()
        for task, ok in zip(tasks.tolist(), correct.tolist()):
            right[task] = right.get(task, 0) + ok
            total[task] = total.get(task, 0) + 1
    model.train()
    return sum(losses) / len(losses), {f"qa{t}": right[t] / total[t] for t in sorted(total)}


def evaluate_lengths(args, model, criterion, tokenizer, tasks, noise, device):
    """Accuracy at each of ``--eval_lens``: ``{length: {"mean": .., "qa1": .., ...}}``."""
    results = {}
    for length in args.eval_lens:
        # Nothing is truncated here: the source is as long as the sample is.
        loader = make_loader(args, tokenizer, tasks, "valid", noise, (length, length),
                             args.eval_samples_per_task, args.eval_batch_size,
                             src_len=max(length, args.src_len) + QUESTION_ROOM, seed=args.seed)
        _, accuracy = evaluate(model, criterion, loader, device)
        mean = sum(accuracy.values()) / len(accuracy)
        results[length] = dict(mean=mean, **accuracy)
        seen = "trained" if args.train_len[0] <= length <= args.train_len[1] else "extrapolation"
        print(f"  length {length:>6} ({seen:>13}) | acc {mean:.3f} | "
              + " ".join(f"{t} {a:.2f}" for t, a in accuracy.items()), flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", choices=MODEL_TYPES, default="transformer_mask")
    p.add_argument("--encoder_mask", default="B", help="transformer_mask only, e.g. CCCCFFFF")
    p.add_argument("--pe", default="none", choices=PE_TYPES, help="transformer_mask only")
    p.add_argument("--babi_dir", default="data/tasks_1-20_v1-2/en-valid-10k",
                   help="folder with qa<N>_train.txt and qa<N>_valid.txt")
    p.add_argument("--noise_dataset", default="pg19")
    p.add_argument("--train_len", type=int, nargs=2, default=[0, 2000], metavar=("LOW", "HIGH"),
                   help="each training sample's length is drawn uniformly from this range")
    p.add_argument("--eval_lens", type=int, nargs="+", default=[0, 1000, 2000, 4000, 8000],
                   help="lengths evaluated after every stage; those past --train_len are extrapolation")
    p.add_argument("--src_len", type=int, default=2048, help="longest training source")
    p.add_argument("--tgt_len", type=int, default=16)
    p.add_argument("--samples_per_task", type=int, default=None, help="default: all")
    p.add_argument("--val_samples_per_task", type=int, default=200, help="per-epoch validation")
    p.add_argument("--eval_samples_per_task", type=int, default=100, help="per length, after a stage")
    p.add_argument("--epochs", type=int, nargs=3, default=[5, 5, 5], help="epochs per stage")
    p.add_argument("--start_stage", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=2,
                   help="small: at 8k the per-head attention masks take ~2 GB per sample")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=3000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # A pad token of its own: GPT-2 has none, and reusing EOS would hide EOS from the loss.
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.save_pretrained(args.output_dir)
    vocab_size, pad_id = len(tokenizer), tokenizer.pad_token_id

    # max_len only reaches roformer: its position table has to cover the longest evaluation.
    longest = max(args.src_len, *args.eval_lens) + QUESTION_ROOM
    model = build_model(args.model, vocab_size, pad_id, mask_spec=args.encoder_mask, pe=args.pe,
                        max_len=longest + args.tgt_len).to(device)
    print(model.description)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    step = 0
    if args.start_stage > 1:
        ckpt = torch.load(os.path.join(args.output_dir, f"stage{args.start_stage - 1}.pt"),
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]

    noise = {}
    if max(args.train_len[1], *args.eval_lens) > PROMPT_RESERVE:
        from datasets import load_dataset
        # The benchmark's noise is PG19 test, so training and validation use the other splits.
        noise = {split: load_dataset(args.noise_dataset, split=split) for split in ("train", "validation")}

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    for stage in range(args.start_stage, len(STAGES) + 1):
        tasks = sorted(t for s in STAGES[:stage] for t in s)
        train_loader = make_loader(args, tokenizer, tasks, "train", noise.get("train"), args.train_len,
                                   args.samples_per_task, args.batch_size, args.src_len)
        # Validated at the training lengths; the seed keeps it the same set every epoch.
        val_loader = make_loader(args, tokenizer, tasks, "valid", noise.get("validation"),
                                 args.train_len, args.val_samples_per_task, args.batch_size,
                                 args.src_len, seed=args.seed)
        print(f"=== stage {stage}: tasks {tasks} | {len(train_loader.dataset)} train samples, "
              f"{len(val_loader.dataset)} val, {args.epochs[stage - 1]} epochs ===", flush=True)

        for epoch in range(1, args.epochs[stage - 1] + 1):
            running = []
            for i, batch in enumerate(train_loader, 1):
                # Linear warmup once, at the start of the run; constant afterwards.
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * min(1.0, (step + 1) / args.warmup_steps)
                loss = forward(model, criterion, batch, device)[0]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                step += 1
                running.append(loss.item())
                if not math.isfinite(running[-1]):
                    raise RuntimeError(f"non-finite loss at stage {stage} epoch {epoch} step {i}")
                if i % args.log_every == 0:
                    recent = running[-args.log_every:]
                    print(f"  stage {stage} epoch {epoch} step {i}/{len(train_loader)} | "
                          f"loss {sum(recent) / len(recent):.4f}", flush=True)

            val_loss, accuracy = evaluate(model, criterion, val_loader, device)
            mean_acc = sum(accuracy.values()) / len(accuracy)
            print(f"stage {stage} epoch {epoch} | train {sum(running) / len(running):.4f} | "
                  f"val {val_loss:.4f} | acc {mean_acc:.3f} | "
                  + " ".join(f"{t} {a:.2f}" for t, a in accuracy.items()), flush=True)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(dict(stage=stage, epoch=epoch, step=step,
                                        train_loss=sum(running) / len(running), val_loss=val_loss,
                                        val_accuracy=mean_acc, **accuracy)) + "\n")

        # Saved before the length evaluation, so a failure at 8k cannot lose the stage.
        # The same layout as the RULER checkpoints, plus what a later stage needs to continue.
        out = os.path.join(args.output_dir, f"stage{stage}.pt")
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                        stage=stage, tasks=tasks, args=vars(args), vocab_size=vocab_size,
                        pad_token_id=pad_id, bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id), out + ".tmp")
        os.replace(out + ".tmp", out)
        print(f"saved {out}")

        print(f"stage {stage} accuracy by length:")
        by_length = evaluate_lengths(args, model, criterion, tokenizer, tasks,
                                     noise.get("validation"), device)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(dict(stage=stage, step=step, train_len=args.train_len,
                                    accuracy_by_length=by_length)) + "\n")


if __name__ == "__main__":
    main()
