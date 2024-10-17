#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""FastSpeech related loss."""

from typing import Dict, Tuple, Sequence

import torch

from lightspeech.core.alignments import GaussianUpsampling
from lightspeech.core.duration_modeling.duration_predictor import DurationPredictor
from lightspeech.core.duration_modeling.duration_predictor import DurationPredictorLoss
from lightspeech.core.duration_modeling.length_regulator import LengthRegulator
from lightspeech.core.embedding import PositionalEncoding, ScaledSinusoidalEmbedding
from lightspeech.core.encoder import Encoder
from lightspeech.core.modules import Postnet
from lightspeech.core.modules import initialize
from lightspeech.core.variance_predictor import EnergyPredictor, EnergyPredictorLoss
from lightspeech.core.variance_predictor import PitchPredictor, PitchPredictorLoss
from lightspeech.utils.util import make_non_pad_mask, sequence_mask
from lightspeech.utils.util import make_pad_mask


class FeedForwardTransformer(torch.nn.Module):
    """Feed Forward Transformer for TTS a.k.a. FastSpeech.
    This is a module of FastSpeech, feed-forward Transformer with duration predictor described in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_, which does not require any auto-regressive
    processing during inference, resulting in fast decoding compared with auto-regressive Transformer.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    """

    def __init__(self, idim: int, odim: int, hp: Dict):
        """Initialize feed-forward Transformer module.
        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
        """
        # initialize base classes
        # assert check_argument_types()
        torch.nn.Module.__init__(self)

        # fill missing arguments

        # store hyperparameters
        self.idim = idim
        self.odim = odim

        self.use_scaled_pos_enc = hp.model.use_scaled_pos_enc
        self.use_masking = hp.model.use_masking

        # use idx 0 as padding idx
        padding_idx = 0

        # get positional encoding class
        pos_enc_class = ScaledSinusoidalEmbedding if self.use_scaled_pos_enc else PositionalEncoding

        # define encoder
        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim,
            embedding_dim=hp.model.adim,
            padding_idx=padding_idx
        )
        self.encoder = Encoder(
            idim=idim,
            attention_dim=hp.model.adim,
            attention_heads=hp.model.aheads,
            linear_units=hp.model.eunits,
            num_blocks=hp.model.elayers,
            input_layer=encoder_input_layer,
            dropout_rate=0.2,
            positional_dropout_rate=0.2,
            attention_dropout_rate=0.2,
            pos_enc_class=pos_enc_class,
            normalize_before=hp.model.encoder_normalize_before,
            concat_after=hp.model.encoder_concat_after,
            positionwise_conv_kernel_sizes=hp.model.encoder_positionwise_conv_kernel_size
        )

        self.duration_predictor = DurationPredictor(
            idim=hp.model.adim,
            n_layers=hp.model.duration_predictor_layers,
            n_chans=hp.model.duration_predictor_chans,
            kernel_size=hp.model.duration_predictor_kernel_size,
            dropout_rate=hp.model.duration_predictor_dropout_rate,
        )

        self.energy_predictor = EnergyPredictor(
            idim=hp.model.adim,
            n_layers=hp.model.duration_predictor_layers,
            n_chans=hp.model.duration_predictor_chans,
            kernel_size=hp.model.duration_predictor_kernel_size,
            dropout_rate=hp.model.duration_predictor_dropout_rate,
            min=hp.data.e_min,
            max=hp.data.e_max,
        )
        self.energy_embed = torch.nn.Linear(hp.model.adim, hp.model.adim)

        self.pitch_predictor = PitchPredictor(
            idim=hp.model.adim,
            n_layers=hp.model.duration_predictor_layers,
            n_chans=hp.model.duration_predictor_chans,
            kernel_size=hp.model.duration_predictor_kernel_size,
            dropout_rate=hp.model.duration_predictor_dropout_rate,
            min=hp.data.p_min,
            max=hp.data.p_max,
        )
        self.pitch_embed = torch.nn.Linear(hp.model.adim, hp.model.adim)

        # define length regulator
        self.length_regulator = LengthRegulator()

        self.gaussian_upsampler = GaussianUpsampling()

        # define decoder
        # NOTE: we use encoder as decoder because fastspeech's decoder is the same as encoder
        self.decoder = Encoder(
            idim=256,
            attention_dim=256,
            attention_heads=hp.model.aheads,
            linear_units=hp.model.dunits,
            num_blocks=hp.model.dlayers,
            input_layer=None,
            dropout_rate=0.2,
            positional_dropout_rate=0.2,
            attention_dropout_rate=0.2,
            pos_enc_class=pos_enc_class,
            normalize_before=hp.model.decoder_normalize_before,
            concat_after=hp.model.decoder_concat_after,
            positionwise_conv_kernel_sizes=hp.model.decoder_positionwise_conv_kernel_size
        )

        # define postnet
        self.postnet = (
            None
            if hp.model.postnet_layers == 0
            else Postnet(
                idim=idim,
                odim=odim,
                n_layers=hp.model.postnet_layers,
                n_chans=hp.model.postnet_chans,
                n_filts=hp.model.postnet_filts,
                use_batch_norm=hp.model.use_batch_norm,
                dropout_rate=hp.model.postnet_dropout_rate,
            )
        )

        # define final projection
        self.feat_out = torch.nn.Linear(hp.model.adim, odim * hp.model.reduction_factor)

        # initialize parameters
        self._reset_parameters(init_type=hp.model.transformer_init,
                               init_enc_alpha=hp.model.initial_encoder_alpha,
                               init_dec_alpha=hp.model.initial_decoder_alpha)

        # define criterions
        self.duration_criterion = DurationPredictorLoss()
        self.energy_criterion = EnergyPredictorLoss()
        self.pitch_criterion = PitchPredictorLoss()
        self.criterion = torch.nn.L1Loss(reduction='mean')
        self.use_weighted_masking = hp.model.use_weighted_masking

    def _forward(self, xs: torch.Tensor,
                 ilens: torch.Tensor,
                 olens: torch.Tensor = None,
                 ds: torch.Tensor = None,
                 es: torch.Tensor = None,
                 ps: torch.Tensor = None,
                 is_inference: bool = False) -> Sequence[torch.Tensor]:
        # forward encoder
        x_masks = self._source_mask(ilens)  # (B, Tmax, Tmax) -> torch.Size([32, 121, 121])

        hs, _ = self.encoder(xs, x_masks)  # (B, Tmax, adim) -> torch.Size([32, 121, 256])
        # print("ys :", ys.shape)

        x_max_length = ilens.max()
        x_mask = torch.unsqueeze(sequence_mask(ilens, x_max_length), 1).type_as(xs)

        # forward duration predictor and length regulator
        d_masks = make_pad_mask(ilens).to(xs.device)

        if is_inference:
            d_outs = self.duration_predictor.inference(hs, d_masks)
            # hs = self.length_regulator(hs, d_outs, ilens)  # (B, Lmax, adim)

            y_lengths = d_outs.sum(dim=1)
            y_max_length = y_lengths.max()
            y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).type_as(xs)

            hs = self.gaussian_upsampler(
                hs=hs, ds=d_outs, h_masks=y_mask.squeeze(1).bool(), d_masks=x_mask.squeeze(1).bool()
            )

            one_hot_energy = self.energy_predictor.inference(hs)  # (B, Lmax, adim)
            one_hot_pitch = self.pitch_predictor.inference(hs)  # (B, Lmax, adim)
        else:
            with torch.no_grad():
                # Shapes for botch: (B, Mel_Max, Hidden_dim)
                one_hot_energy = self.energy_predictor.to_one_hot(es)  # (B, Lmax, adim)   torch.Size([32, 868, 256])
                one_hot_pitch = self.pitch_predictor.to_one_hot(ps)  # (B, Lmax, adim)   torch.Size([32, 868, 256])

            mel_masks = make_pad_mask(olens).to(xs.device)
            d_outs = self.duration_predictor(hs, d_masks)  # (B, Tmax)
            # hs = self.length_regulator(hs, ds, ilens)  # (B, Lmax, adim)
            mel_max_length = olens.max()
            mel_mask = torch.unsqueeze(sequence_mask(olens, mel_max_length), 1).type_as(xs)

            hs = self.gaussian_upsampler(
                hs=hs, ds=ds, h_masks=mel_mask.squeeze(1).bool(), d_masks=x_mask.squeeze(1).bool()
            )
            e_outs = self.energy_predictor(hs, mel_masks)
            p_outs = self.pitch_predictor(hs, mel_masks)
        hs = hs + self.pitch_embed(one_hot_pitch)  # (B, Lmax, adim)
        hs = hs + self.energy_embed(one_hot_energy)  # (B, Lmax, adim)
        # forward decoder
        if olens is not None:
            h_masks = self._source_mask(olens)
        else:
            h_masks = None

        zs, _ = self.decoder(hs, h_masks)  # (B, Lmax, adim)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)  # (B, Lmax, odim)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        if is_inference:
            return before_outs, after_outs, d_outs
        else:
            return before_outs, after_outs, d_outs, e_outs, p_outs

    def forward(self, xs: torch.Tensor,
                ilens: torch.Tensor,
                ys: torch.Tensor,
                olens: torch.Tensor,
                durations: torch.Tensor,
                es: torch.Tensor,
                ps: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of padded character ids (B, Tmax).
            ilens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor): Batch of padded mel tensors (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of mel tensor (B,).
            durations (LongTensor): Batch of duration of each phoneme (B, Tmax).
            es (LongTensor): Batch of the time-aligned energy (corresponding to mel sampling rate) (B, Lmax).
            ps (LongTensor): Batch of the time-aligned pitch (corresponding to mel sampling rate) (B, Lmax).
        Returns:
            Tensor: Loss value.
        """
        # remove unnecessary padded part (for multi-gpus)
        xs = xs[:, :max(ilens)]  # torch.Size([32, 121]) -> [B, Tmax]
        ys = ys[:, :max(olens)]  # torch.Size([32, 868, 80]) -> [B, Lmax, odim]

        # forward propagation
        before_outs, after_outs, d_outs, e_outs, p_outs = self._forward(xs, ilens, olens, durations, es, ps,
                                                                        is_inference=False)

        # apply mask to remove padded part
        if self.use_masking:
            in_masks = make_non_pad_mask(ilens).to(xs.device)
            d_outs = d_outs.masked_select(in_masks)
            durations = durations.masked_select(in_masks)
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            mel_masks = make_non_pad_mask(olens).to(ys.device)
            before_outs = before_outs.masked_select(out_masks)
            es = es.masked_select(mel_masks)  # Write size
            ps = ps.masked_select(mel_masks)  # Write size
            e_outs = e_outs.masked_select(mel_masks)  # Write size
            p_outs = p_outs.masked_select(mel_masks)  # Write size
            after_outs = (
                after_outs.masked_select(out_masks) if after_outs is not None else None
            )
            ys = ys.masked_select(out_masks)

        # calculate loss
        before_loss = self.criterion(before_outs, ys)
        after_loss = 0
        if after_outs is not None:
            after_loss = self.criterion(after_outs, ys)
            l1_loss = before_loss + after_loss
        duration_loss = self.duration_criterion(d_outs, durations)
        energy_loss = self.energy_criterion(e_outs, es)
        pitch_loss = self.pitch_criterion(p_outs, ps)

        loss: torch.Tensor = l1_loss + duration_loss + energy_loss + pitch_loss
        report_keys = [
            {"l1_loss": l1_loss.item()},
            {"before_loss": before_loss.item()},
            {"after_loss": after_loss.item()},
            {"duration_loss": duration_loss.item()},
            {"energy_loss": energy_loss.item()},
            {"pitch_loss": pitch_loss.item()},
            {"loss": loss.item()},
        ]

        # self.reporter.report(report_keys)

        return loss, report_keys

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Generate the sequence of features given the sequences of characters.
        Args:
            x (Tensor): Input sequence of characters (T,).
            inference_args (Namespace): Dummy for compatibility.
            spemb (Tensor, optional): Speaker embedding vector (spk_embed_dim).
        Returns:
            Tensor: Output sequence of features (1, L, odim).
            None: Dummy for compatibility.
            None: Dummy for compatibility.
        """
        # setup batch axis
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs = x.unsqueeze(0)

        # inference
        _, outs, _ = self._forward(xs, ilens, is_inference=True)  # (L, odim)

        return outs[0]

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.
        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)
        """
        x_masks = make_non_pad_mask(ilens).to(device=next(self.parameters()).device)
        return x_masks.unsqueeze(-2) & x_masks.unsqueeze(-1)

    def _reset_parameters(self, init_type: str, init_enc_alpha: float = 1.0, init_dec_alpha: float = 1.0):
        # initialize parameters
        initialize(self, init_type)

        # NOTE: commented out. Optispeech's default value for "scale" (not alpha) seems to be good enough
        # initialize alpha in scaled positional encoding
        # if self.use_scaled_pos_enc:
        #     self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
        #     self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)
