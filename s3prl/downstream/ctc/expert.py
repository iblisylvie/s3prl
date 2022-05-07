import os
import math
import torch
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from ..model import *
from ..asr.model import *
from .text import load_text_encoder
from .data import load_dataset
from .metric import *


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(
        self, upstream_dim, upstream_rate, downstream_expert, expdir, **kwargs
    ):
        super(DownstreamExpert, self).__init__()
        self.expdir = expdir
        self.upstream_dim = upstream_dim
        self.corpus = downstream_expert["corpus"]

        # Text tokenizer
        self.tokenizer = load_text_encoder(**downstream_expert["text"])

        modelrc = downstream_expert["model"]
        self.projector = nn.Linear(upstream_dim, modelrc["project_dim"])

        model_select = downstream_expert["model"]["select"]
        self.model = eval(model_select)(
            modelrc["project_dim"],
            self.tokenizer.vocab_size,
            upstream_rate=upstream_rate,
            **modelrc.get(model_select, {}),
        )
        self.objective = nn.CTCLoss(
            blank=self.tokenizer.pad_idx,
            zero_infinity=modelrc["zero_infinity"],
        )
        self.save_best_on = downstream_expert.get("save_best_on", "dev")
        self.metrics = downstream_expert["metric"]
        self.metric_higher_better = downstream_expert["metric_higher_better"]
        self.register_buffer(
            "best_score", torch.ones(1) * (0 if self.metric_higher_better else 1 << 31)
        )

    def _get_task_name(self):
        return f'ctc-{self.corpus["name"].lower()}'

    # Interface
    def get_dataloader(self, split):
        # return load_dataset(split, self.tokenizer, self.corpus)
        return None

    def encode(self, s):
      return self.tokenizer.encode(s)


    def beam_search(self, features, beam_size):
      from collections import defaultdict
      from ctc_utils import log_add

      # 1. Encoder forward and get CTC score
      features = pad_sequence(features, batch_first=True)
      features_len = torch.IntTensor([len(feat) for feat in features])
      features = self.projector(features)
      logits, log_probs_len = self.model(features, features_len)
      maxlen = logits.size(1)
      ctc_probs = nn.functional.log_softmax(logits, dim=-1)
      ctc_probs = ctc_probs.squeeze(0)
      # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
      cur_hyps = [(tuple(), (0.0, -float('inf')))]
      # 2. CTC beam search step by step
      for t in range(0, maxlen):
        logp = ctc_probs[t] # (vocab_size,)
        # key: prefix, value (pb, pnb), default value(-inf, -inf)
        next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
        # 2.1 First beam prune: select topk best
        top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
        for s in top_k_index:
          s = s.item()
          ps = logp[s].item()
          for prefix, (pb, pnb) in cur_hyps:
            last = prefix[-1] if len(prefix) > 0 else None
            if s == 0:  # blank
              n_pb, n_pnb = next_hyps[prefix]
              n_pb = log_add([n_pb, pb + ps, pnb + ps])
              next_hyps[prefix] = (n_pb, n_pnb)
            elif s == last:
              #  Update *ss -> *s;
              n_pb, n_pnb = next_hyps[prefix]
              n_pnb = log_add([n_pnb, pnb + ps])
              next_hyps[prefix] = (n_pb, n_pnb)
              # Update *s-s -> *ss, - is for blank
              n_prefix = prefix + (s, )
              n_pb, n_pnb = next_hyps[n_prefix]
              n_pnb = log_add([n_pnb, pb + ps])
              next_hyps[n_prefix] = (n_pb, n_pnb)
            else:
              n_prefix = prefix + (s, )
              n_pb, n_pnb = next_hyps[n_prefix]
              n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
              next_hyps[n_prefix] = (n_pb, n_pnb)

        # 2.2 Second beam prune
        next_hyps = sorted(next_hyps.items(),
                          key=lambda x: log_add(list(x[1])),
                          reverse=True)
        cur_hyps = next_hyps[:beam_size]
      hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
      print('beam search hyps')
      print(hyps)
      return hyps

    def decode(self, pred_tokens):
      filtered_tokens = [
          token
          for token in pred_tokens
          if token != self.tokenizer.pad_idx and token != self.tokenizer.eos_idx
      ]
      hypothesis = self.tokenizer.decode(filtered_tokens, ignore_repeat=True)
      return hypothesis


    def inference(self, features):
      features = pad_sequence(features, batch_first=True)
      features_len = torch.IntTensor([len(feat) for feat in features])
      features = self.projector(features)
      logits, log_probs_len = self.model(features, features_len)
      log_probs = nn.functional.log_softmax(logits, dim=-1)
      pred_tokens = log_probs.argmax(dim=-1)

      filtered_tokens = []
      for pred_token in pred_tokens:
        filtered_token = [
          token
          for token in pred_token.tolist()
          if token != self.tokenizer.pad_idx and token != self.tokenizer.eos_idx
        ]
        filtered_tokens.append(filtered_token)
        hypothesis = [
          self.tokenizer.decode(h, ignore_repeat=True) for h in filtered_tokens
        ]
        # print('-------> hypothesis')
        # print(hypothesis)
        return log_probs

    # Interface
    def forward(self, split, features, labels, filenames, records, **kwargs):
        device = features[0].device
        labels = [torch.LongTensor(label) for label in labels]
        features_len = torch.IntTensor([len(feat) for feat in features])
        labels_len = torch.IntTensor([len(label) for label in labels])
        features = pad_sequence(features, batch_first=True)
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.tokenizer.pad_idx,
        ).to(device=device)

        features = self.projector(features)
        logits, log_probs_len = self.model(features, features_len)
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        loss = self.objective(
            log_probs.transpose(0, 1),  # (N, T, C) -> (T, N, C)
            labels,
            log_probs_len,
            labels_len,
        )
        records["loss"].append(loss.item())

        pred_tokens = log_probs.argmax(dim=-1)
        filtered_tokens = []
        for pred_token in pred_tokens:
            filtered_token = [
                token
                for token in pred_token.tolist()
                if token != self.tokenizer.pad_idx and token != self.tokenizer.eos_idx
            ]
            filtered_tokens.append(filtered_token)

        hypothesis = [
            self.tokenizer.decode(h, ignore_repeat=True) for h in filtered_tokens
        ]
        groundtruth = [self.tokenizer.decode(g.tolist()) for g in labels]

        # store all text in a batch
        records["hypothesis"] += hypothesis
        records["groundtruth"] += groundtruth
        records["filename"] += filenames

        return loss

    # interface
    def log_records(self, split, records, logger, global_step, **kwargs):
        loss = torch.FloatTensor(records["loss"]).mean().item()
        results = {"loss": loss}

        for metric in self.metrics:
            results[metric] = eval(metric)(
                hypothesis=records["hypothesis"],
                groundtruth=records["groundtruth"],
            )

        save_names = []
        for key, value in results.items():
            print(f"{split} {key}: {value}")

            logger.add_scalar(
                f"{self._get_task_name()}/{split}-{key}", value, global_step=global_step
            )
            if key == self.metrics[0]:
                save_criterion = (
                    value > self.best_score
                    if self.metric_higher_better
                    else value < self.best_score
                )
                if split in self.save_best_on and save_criterion:
                    self.best_score = torch.ones(1) * value
                    save_names.append(f"{split}-best.ckpt")

        if "test" in split or "dev" in split:
            hyp_ark = open(os.path.join(self.expdir, f"{split}-hyp.ark"), "w")
            ref_ark = open(os.path.join(self.expdir, f"{split}-ref.ark"), "w")
            for filename, hyp, ref in zip(
                records["filename"], records["hypothesis"], records["groundtruth"]
            ):
                hyp_ark.write(f"{filename} {hyp}\n")
                ref_ark.write(f"{filename} {ref}\n")
            hyp_ark.close()
            ref_ark.close()

        return save_names
