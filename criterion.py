# Revised from fairseq.criterions.masked_lm
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("ntl")
class NTLLoss(FairseqCriterion):
    """
    Implementation for the loss used in non-transferable learning
    """

    def __init__(self, task, tpu=False):
        super().__init__(task)
        self.tpu = tpu

    def forward(self, model, sample, reduce=True):

        ## TODO: finish criterion
        print(model.encoder.lm_head.dense.bias)
        exit(0)
        def compute_mlm(input, target):
            """Compute the loss for the given sample.

            Returns a tuple with three elements:
            1) the loss
            2) the sample size, which is used as the denominator for the gradient
            3) logging outputs to display while training
            """
            masked_tokens = target.ne(self.padding_idx)
            sample_size = masked_tokens.int().sum()

            # Rare: when all tokens are masked, project all tokens.
            # We use torch.where to avoid device-to-host transfers,
            # except on CPU where torch.where is not well supported
            # (see github.com/pytorch/pytorch/issues/26247).
            if self.tpu:
                masked_tokens = None  # always project all tokens on TPU
            elif masked_tokens.device == torch.device("cpu"):
                if not masked_tokens.any():
                    masked_tokens = None
            else:
                masked_tokens = torch.where(
                    masked_tokens.any(),
                    masked_tokens,
                    masked_tokens.new([True]),
                )

            logits, extra = model(**input, return_all_hiddens=True, masked_tokens=masked_tokens)
            targets = model.get_targets(sample, [logits])
            if masked_tokens is not None:
                targets = targets[masked_tokens]

            loss = modules.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="sum",
                ignore_index=self.padding_idx,
            )
            return loss, sample_size, logits

        l_s, s_size,
        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
