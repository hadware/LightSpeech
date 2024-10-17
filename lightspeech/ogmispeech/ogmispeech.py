#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""FastSpeech related loss."""

from typing import Callable, TypedDict
from typing import Dict, Tuple

import torch

from lightspeech.ogmispeech.alignments import GaussianUpsampling
from lightspeech.ogmispeech.duration_modeling.duration_predictor import DurationPredictor
from lightspeech.ogmispeech.embedding import PositionalEncoding, ScaledSinusoidalEmbedding
from lightspeech.ogmispeech.encoder import Encoder
from lightspeech.ogmispeech.modules import Postnet
from lightspeech.ogmispeech.loss import MelLoss, FastSpeech2Loss
from lightspeech.ogmispeech.variance_predictor import SpeechFeaturePredictor
from lightspeech.utils.util import sequence_mask


class GeneratorTrainingLosses(TypedDict):
    loss: torch.Tensor
    mel_loss: torch.Tensor
    duration_loss: torch.Tensor
    pitch_loss: torch.Tensor
    energy_loss: torch.Tensor


class GeneratorTrainingOutputs(TypedDict):
    mel: torch.Tensor
    pitch: torch.Tensor
    energy: torch.Tensor


class InferenceOutput(TypedDict):
    mel: torch.Tensor
    durations: torch.Tensor
    pitch: torch.Tensor
    energy: torch.Tensor
    am_infer: float


class OgmiosTransformer(torch.nn.Module):
    # TODO: doc

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 encoder: Callable[[int, int], Encoder],
                 decoder: Callable[[int, int], Encoder],
                 duration_predictor: Callable[[int], DurationPredictor],
                 energy_predictor: Callable[[int, int], SpeechFeaturePredictor],
                 pitch_predictor: Callable[[int, int], SpeechFeaturePredictor],

                 ):
        """Initialize feed-forward Transformer module.
        Args:
            input_dim (int): Dimension of the inputs.
            output_dim (int): Dimension of the outputs.
        """
        # initialize base classes
        # assert check_argument_types()
        torch.nn.Module.__init__(self)

        # fill missing arguments

        # store hyperparameters
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.use_scaled_pos_enc = hp.model.use_scaled_pos_enc
        self.use_masking = hp.model.use_masking

        # get positional encoding class
        # TODO: remove this and include it in class
        pos_enc_class = ScaledSinusoidalEmbedding if self.use_scaled_pos_enc else PositionalEncoding

        # TODO: pull out encoder input layer from encoder/decoder arch
        self.input_embedder = torch.nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=hidden_dim,
            padding_idx=0
        )
        self.encoder = encoder(input_dim, hidden_dim)
        self.decoder = decoder(hidden_dim, hidden_dim)

        self.duration_predictor = duration_predictor(input_dim)
        self.energy_predictor = energy_predictor(input_dim, hidden_dim)
        self.pitch_predictor = pitch_predictor(input_dim, hidden_dim)

        self.gaussian_upsampler = GaussianUpsampling()

        # define postnet
        self.postnet = (
            None
            if hp.model.postnet_layers == 0
            else Postnet(
                idim=input_dim,
                odim=output_dim,
                n_layers=hp.model.postnet_layers,
                n_chans=hp.model.postnet_chans,
                n_filts=hp.model.postnet_filts,
                use_batch_norm=hp.model.use_batch_norm,
                dropout_rate=hp.model.postnet_dropout_rate,
            )
        )

        # define final projection
        self.feat_out = torch.nn.Linear(hp.model.adim, output_dim * hp.model.reduction_factor)

        # define criterions
        # TODO: double check that pitch/energy/duration loss are the same as defined
        #  in the various losses
        self.loss_criterion = FastSpeech2Loss(regression_loss_type="mse")
        self.mel_criterion = MelLoss(regression_loss_type="l1")

    def forward(self,
                x: torch.Tensor,
                x_lengths: torch.Tensor,
                mels: torch.Tensor,
                mels_lengths: torch.Tensor,
                durations: torch.Tensor,
                pitches: torch.Tensor,
                energies: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            mel (torch.Tensor): batch of mels corresponding to each phonemized text.
                shape: (batch_size,max_mel_number, mel_number) TODO check order
            mel_lengths (torch.Tensor): lengths of mels in batch.
                shape: (batch_size,)
            durations (torch.Tensor): durations of each phonemes in batch.
                shape: (batch_size, max_text_length)
            pitches (torch.Tensor): mel frame-leve pitch values.
                shape: (batch_size, max_mel_duration)
            energies (torch.Tensor): mel frame-leve energy values.
                shape: (batch_size, max_mel_duration)

        Returns:
            loss: (torch.Tensor): scaler representing total loss
            alignment_loss: (torch.Tensor): scaler representing alignment loss
            duration_loss: (torch.Tensor): scaler representing durations loss
            pitch_loss: (torch.Tensor): scaler representing pitch loss
            energy_loss: (torch.Tensor): scaler representing energy loss
        """

        # forward propagation
        # forward encoder
        x_max_length = x_lengths.max()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_max_length), 1).type_as(x)

        mels_max_length = mels_lengths.max()
        mels_mask = torch.unsqueeze(sequence_mask(mels_lengths, mels_max_length), 1).type_as(x)

        input_padding_mask = ~x_mask.squeeze(1).bool().to(x.device)
        target_padding_mask = ~mels_mask.squeeze(1).bool().to(x.device)  # (B, Tmax, Tmax) -> torch.Size([32, 121, 121])

        # TODO: don't forget to add input embedding
        hs, _ = self.encoder(x, x_mask)  # (B, Tmax, adim) -> torch.Size([32, 121, 256])

        with torch.no_grad():
            # Shapes for botch: (B, Mel_Max, Hidden_dim)
            one_hot_energy = self.energy_predictor.to_one_hot(energies)  # (B, Lmax, adim)   torch.Size([32, 868, 256])
            one_hot_pitch = self.pitch_predictor.to_one_hot(pitches)  # (B, Lmax, adim)   torch.Size([32, 868, 256])

        duration_hat = self.duration_predictor(hs, input_padding_mask)  # (B, Tmax)

        hs = self.gaussian_upsampler(
            hs=hs, ds=durations, h_masks=mels_mask.squeeze(1).bool(), d_masks=x_mask.squeeze(1).bool()
        )

        energy_hat = self.energy_predictor(hs, target_padding_mask)
        pitch_hat = self.pitch_predictor(hs, target_padding_mask)

        hs = hs + self.pitch_embed(one_hot_pitch)  # (B, Lmax, adim)
        hs = hs + self.energy_embed(one_hot_energy)  # (B, Lmax, adim)

        zs, _ = self.decoder(hs, mels_mask)  # (B, Lmax, adim)
        mels_hat = self.feat_out(zs).view(zs.size(0), -1, self.output_dim)  # (B, Lmax, odim)

        # TODO: maybe re-add postnet at some point

        loss_coeffs = self.loss_coeffs
        duration_loss, pitch_loss, energy_loss = self.loss_criterion(
            d_outs=duration_hat.unsqueeze(-1),
            p_outs=pitch_hat.unsqueeze(-1),
            e_outs=energy_hat.unsqueeze(-1),
            ds=durations.unsqueeze(-1),
            ps=pitches.unsqueeze(-1),
            es=energies.unsqueeze(-1),
            ilens=x_lengths,
            olens=mels_lengths,
        )
        mel_loss = self.mel_criterion(mels_hat, mels, mels_lengths)
        loss = (
                (mel_loss * loss_coeffs.lambda_mel)
                + (duration_loss * loss_coeffs.lambda_duration)
                + (pitch_loss * loss_coeffs.lambda_pitch)
                + (energy_loss * loss_coeffs.lambda_energy)
        )

        loss_report = GeneratorTrainingLosses(
            loss=loss.cpu(),
            mel_loss=mel_loss.cpu(),
            duration_loss=duration_loss.detach().cpu(),
            pitch_loss=pitch_loss.detach().cpu(),
            energy_loss=energy_loss.detach().cpu())

        predictions_report = GeneratorTrainingOutputs(
            mel=mels_hat.detach().cpu(),
            pitch=pitch_hat.detach().cpu(),
            energy=energy_hat.detach().cpu()
        )

        return loss_report, predictions_report

    @torch.inference_mode()
    def inference(self, x: torch.Tensor, x_lengths: torch.Tensor) -> torch.Tensor:
        # setup batch axis

        x_max_length = x_lengths.max()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_max_length), 1).type_as(x)
        input_padding_mask = ~x_mask.squeeze(1).bool().to(x.device)

        hs, _ = self.encoder(x, x_mask)  # (B, Tmax, adim) -> torch.Size([32, 121, 256])

        durations = self.duration_predictor.inference(hs, input_padding_mask)

        y_lengths = durations.sum(dim=1)
        y_max_length = y_lengths.max()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).type_as(xs)

        hs = self.gaussian_upsampler(
            hs=hs, ds=durations, h_masks=y_mask.squeeze(1).bool(), d_masks=x_mask.squeeze(1).bool()
        )

        one_hot_energy = self.energy_predictor.inference(hs)  # (B, Lmax, adim)
        one_hot_pitch = self.pitch_predictor.inference(hs)  # (B, Lmax, adim)

        hs = hs + self.pitch_embed(one_hot_pitch)  # (B, Lmax, adim)
        hs = hs + self.energy_embed(one_hot_energy)  # (B, Lmax, adim)

        zs, _ = self.decoder(hs, y_mask)  # (B, Lmax, adim)
        mels = self.feat_out(zs).view(zs.size(0), -1, self.output_dim)  # (B, Lmax, odim)

        return mels

