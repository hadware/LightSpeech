#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Length regulator related loss."""

import torch

from lightspeech.utils.util import pad_2d_tensor


class LengthRegulator(torch.nn.Module):
    """Length regulator module for feed-forward Transformer.

    This is a module of length regulator described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The length regulator expands char or phoneme-level embedding features to frame-level by repeating each
    feature based on the corresponding predicted durations.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, pad_value: float = 0.0):
        """Initilize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.

        """
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(self,
                pho_embd_batch: torch.Tensor,
                durations_batch: torch.Tensor,
                pho_len_batch: torch.Tensor,
                alpha: float = 1.0) \
            -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            pho_embd_batch (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            durations_batch (LongTensor): Batch of durations of each frame (B, T).
            pho_len_batch (LongTensor): Batch of input lengths (B,).
            alpha (float, optional): Alpha value to control speed of speech.

        Returns:
            Tensor: replicated input tensor based on durations (B, T*, D).

        """
        assert alpha > 0
        if alpha != 1.0:
            durations_batch = torch.round(durations_batch.float() * alpha).long()
        pho_embd_batch = [x[:ilen] for x, ilen in zip(pho_embd_batch, pho_len_batch)]
        durations_batch = [d[:ilen] for d, ilen in zip(durations_batch, pho_len_batch)]

        pho_embd_batch = [self._repeat_one_sequence(x, d) for x, d in zip(pho_embd_batch, durations_batch)]

        return pad_2d_tensor(pho_embd_batch, 0.0)

    def _repeat_one_sequence(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Repeat each frame according to duration.

        Examples:
            >>> x = torch.tensor([[1], [2], [3]])
            tensor([[1],
                    [2],
                    [3]])
            >>> d = torch.tensor([1, 2, 3])
            tensor([1, 2, 3])
            >>> self._repeat_one_sequence(x, d)
            tensor([[1],
                    [2],
                    [2],
                    [3],
                    [3],
                    [3]])

        """
        if d.sum() == 0:
            # logging.warn("all of the predicted durations are 0. fill 0 with 1.")
            d = d.fill_(1)
        # return torch.cat([x_.repeat(int(d_), 1) for x_, d_ in zip(x, d) if d_ != 0], dim=0) for torchscript
        out = []
        for x_, d_ in zip(x, d):
            if d_ != 0:
                out.append(x_.repeat(int(d_), 1))

        return torch.cat(out, dim=0)
