# Revised from fairseq.tasks.masked_lm
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from fairseq import utils
from fairseq.data import (
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.tasks import FairseqTask, register_task
from slice_dataset import SliceDataset


logger = logging.getLogger(__name__)


@register_task("ntl_pretrain")
class NTLPretrainTask(FairseqTask):
    """Task for non-transferable learning style pretraining."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "source_data",
            help="colon separated path to source domain data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "auxi_data",
            help="colon separated path to target domain data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--pretrained-model-name-or-path",
            help="The pretrained model to load",
        )
        parser.add_argument(
            "--sample-break-mode",
            default="complete",
            choices=["none", "complete", "complete_doc", "eos"],
            help='If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.',
        )
        parser.add_argument(
            "--tokens-per-sample",
            default=512,
            type=int,
            help="max number of total tokens over all segments "
            "per sample for BERT dataset",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.1,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.1,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--freq-weighted-replacement",
            default=False,
            action="store_true",
            help="sample random replacement words based on word frequencies",
        )
        parser.add_argument(
            "--mask-whole-words",
            default=False,
            action="store_true",
            help="mask whole words; you may also want to set --bpe",
        )
        parser.add_argument(
            "--shorten-method",
            default="none",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )
        parser.add_argument(
            "--shorten-data-split-list",
            default="",
            help="comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)',
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed

        # add mask token
        self.mask_idx = dictionary.add_symbol("<mask>")

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = args.source_data.split(':')
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def _load_single_dataset(self, data, split, seed_offset, epoch=0, combine=False, **kwargs):
        paths = data.split(':')
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.args.sample_break_mode,
        )
        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        # create masked input and targets
        mask_whole_words = (
            get_whole_word_mask(self.args, self.source_dictionary)
            if self.args.mask_whole_words
            else None
        )

        src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.args.seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
            freq_weighted_replacement=self.args.freq_weighted_replacement,
            mask_whole_words=mask_whole_words,
        )

        with data_utils.numpy_seed(self.args.seed + epoch*epoch + seed_offset):
            shuffle = np.random.permutation(len(src_dataset))

        src_dataset = SortDataset(src_dataset, sort_order=[shuffle, src_dataset.sizes])
        tgt_dataset = SortDataset(tgt_dataset, sort_order=[shuffle, tgt_dataset.sizes])
        return src_dataset, tgt_dataset

    # TODO: The length of sentence is 390
    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        source_data = self._load_single_dataset(self.args.source_data, split, 0, **kwargs)
        auxi_data = self._load_single_dataset(self.args.auxi_data, split, 1, **kwargs)

        dataset_len = min(len(source_data[0]), len(auxi_data[0]))
        source_data = [SliceDataset(dataset, dataset_len) for dataset in source_data]
        auxi_data = [SliceDataset(dataset, dataset_len) for dataset in auxi_data]

        self.datasets[split] = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "source_net_input": {
                    "src_tokens": RightPadDataset(
                        source_data[0],
                        pad_idx=self.source_dictionary.pad(),
                    ),
                    "src_lengths": NumelDataset(source_data[0], reduce=False),
                },
                "auxi_net_input": {
                    "src_tokens": RightPadDataset(
                        auxi_data[0],
                        pad_idx=self.source_dictionary.pad(),
                    ),
                    "src_lengths": NumelDataset(auxi_data[0], reduce=False),
                },
                "source_target": RightPadDataset(
                    source_data[1],
                    pad_idx=self.source_dictionary.pad(),
                ),
                "auxi_target": RightPadDataset(
                    auxi_data[1],
                    pad_idx=self.source_dictionary.pad(),
                ),
                "nsentences": NumSamplesDataset(),
                "source_ntokens": NumelDataset(source_data[0], reduce=True),
                "auxi_ntokens": NumelDataset(auxi_data[0], reduce=True),
            },
            sizes=[np.maximum(source_data[0].sizes, auxi_data[0].sizes)],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        raise NotImplementedError
        src_dataset = RightPadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            pad_idx=self.source_dictionary.pad(),
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    def build_model(self, args):
        """
        Build the :class:`~fairseq.models.BaseFairseqModel` instance 
        for ntl pretraining.
        The Pretraining is continued on a pretrained model.
        Args:
            args (argparse.Namespace): parsed command-line arguments
        Returns:
            a :class:`~fairseq.models.BaseFairseqModel` instance
        """
        from fairseq.models import ARCH_MODEL_REGISTRY

        return ARCH_MODEL_REGISTRY[args.arch].from_pretrained(args.pretrained_model_name_or_path).model

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
