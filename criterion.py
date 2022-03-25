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
        m_size = min(s_size, a_size)
        l_dis = max(1, 0.1*self.mmd(feature_source[:m_size, :].double(), feature_auxi[:m_size, :].double()))
        loss = l_s - max(1, 0.1*l_a*l_dis)
        # print(l_s.dtype, l_a.dtype, loss.dtype)
        # print(l_s, l_a, l_dis, loss)
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": (s_size, a_size),
        }
        return loss, 1, logging_output

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
