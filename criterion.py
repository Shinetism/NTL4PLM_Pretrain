# Revised from fairseq.criterions.masked_lm
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from mmd_loss import MMD_loss
import numpy as np


@register_criterion("ntl")
class NTLLoss(FairseqCriterion):
    """
    Implementation for the loss used in non-transferable learning
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.mmd = MMD_loss()

    def forward(self, model, sample, reduce=True):

        def compute_mlm(input, target):
            """Compute the loss for the given sample.

            Returns a tuple with three elements:
            1) the loss
            2) the sample size, which is used as the denominator for the gradient
            3) logging outputs to display while training
            """
            masked_tokens = target.ne(self.padding_idx)
            sample_size = masked_tokens.int().sum().item()

            # (Rare case) When all tokens are masked, the model results in empty
            # tensor and gives CUDA error.
            if sample_size == 0:
                masked_tokens = None

            logits, extra = model(**input, return_all_hiddens=True, masked_tokens=masked_tokens.T)
            if sample_size != 0:
                targets = target[masked_tokens]

            loss = F.nll_loss(
                F.log_softmax(
                    logits.view(-1, logits.size(-1)),
                    dim=-1,
                    dtype=torch.float64,
                ),
                targets.view(-1),
                reduction="sum",
                ignore_index=self.padding_idx,
            )
            return loss/sample_size, sample_size, extra['inner_states'][-1][masked_tokens.T, :]

        l_s, s_size, feature_source = compute_mlm(sample['source_net_input'], sample['source_target'])
        l_a, a_size, feature_auxi = compute_mlm(sample['auxi_net_input'], sample['auxi_target'])
        l_a = l_a * 0.1
        s_indices = np.random.choice(s_size, min(s_size, 64), replace=False)
        f_indices = np.random.choice(a_size, min(a_size, 64), replace=False)
        l_dis = 0.1*self.mmd(feature_source[s_indices, ].double(), feature_auxi[f_indices, ].double())
        if l_dis > 1:
            l_dis = torch.clamp(l_dis, 0, 1)
        if l_a > 1:
            l_a = torch.clamp(l_a, 0, 1)
        loss = l_s - l_a*l_dis
        # print(l_s.dtype, l_a.dtype, loss.dtype)
        # print(round(l_s.item(), 2), round(l_a.item(), 2), round(l_dis.item(), 2), round(loss.item(), 2))
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            # "loss_s": l_s.item(),
            # "loss_a": l_a.item(),
            # "loss_dis": l_dis.item(),
            "ntokens": sample["source_ntokens"]+sample["auxi_ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": 1,
        }
        return loss, 1, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        # loss_s = sum(log.get('loss_s', 0) for log in logging_outputs)
        # loss_a = sum(log.get('loss_a', 0) for log in logging_outputs)
        # loss_dis = sum(log.get('loss_dis', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            # 'loss_s': loss_s / sample_size / math.log(2),
            # 'loss_a': loss_a / sample_size / math.log(2),
            # 'loss_dis': loss_dis / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output