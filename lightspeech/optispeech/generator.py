from time import perf_counter
from typing import TypedDict, Optional

import torch
from torch import nn

from .alignments import (
    AlignmentModule,
    GaussianUpsampling,
    average_by_duration,
)
from .loss import FastSpeech2Loss, MelLoss
from .modules import DurationPredictor, PitchPredictor, EnergyPredictor
from .utils import sequence_mask


class GeneratorTrainingOutput(TypedDict):
    loss: torch.Tensor
    mel_loss: torch.Tensor
    duration_loss: torch.Tensor
    pitch_loss: torch.Tensor
    energy_loss: torch.Tensor


class InferenceOutput(TypedDict):
    mel: torch.Tensor
    durations: torch.Tensor
    pitch: torch.Tensor
    energy: torch.Tensor
    am_infer: float


class OptiSpeechGenerator(nn.Module):
    def __init__(
            self,
            out_dim: int,
            hidden_dim: int,
            text_embedding: nn.Module,
            encoder: nn.Module,
            duration_predictor: DurationPredictor,
            pitch_predictor: PitchPredictor,
            energy_predictor: EnergyPredictor,
            decoder: nn.Module,
            loss_coeffs: dict[str, float],
            n_feats: int,  # number of mel basis
            n_fft: int,  # number of fft points
            hop_length: int,  # number of shift points
            sample_rate: int,  # window length
            **kwargs
    ):
        super().__init__()

        self.loss_coeffs = loss_coeffs
        self.n_feats = n_feats
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.hidden_dim = hidden_dim

        self.text_embedding = text_embedding(dim=hidden_dim)
        self.encoder = encoder(dim=hidden_dim)
        self.duration_predictor = duration_predictor(dim=hidden_dim)
        self.alignment_module = AlignmentModule(adim=hidden_dim, odim=self.n_feats)
        self.pitch_predictor = pitch_predictor(dim=hidden_dim)
        self.energy_predictor = energy_predictor(dim=hidden_dim)
        self.feature_upsampler = GaussianUpsampling()
        self.decoder = decoder(dim=hidden_dim)
        self.feat_out = torch.nn.Linear(hidden_dim, out_dim)

        self.loss_criterion = FastSpeech2Loss(regression_loss_type="mse")
        # self.forwardsum_loss = ForwardSumLoss()
        self.mel_criterion = MelLoss(regression_loss_type="l1")

    def forward(self, x: torch.Tensor,
                x_lengths: torch.Tensor,
                mel: torch.Tensor,
                mel_lengths: torch.Tensor,
                gt_durations: torch.Tensor,
                pitches: torch.Tensor,
                energies: torch.Tensor):
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
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size, max_mel_length)
            pitches (torch.Tensor): phoneme-level pitch values.
                shape: (batch_size, max_text_length)
            energies (torch.Tensor): phoneme-level energy values.
                shape: (batch_size, max_text_length)

        Returns:
            loss: (torch.Tensor): scaler representing total loss
            alignment_loss: (torch.Tensor): scaler representing alignment loss
            duration_loss: (torch.Tensor): scaler representing durations loss
            pitch_loss: (torch.Tensor): scaler representing pitch loss
            energy_loss: (torch.Tensor): scaler representing energy loss
        """
        x_max_length = x_lengths.max()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_max_length), 1).type_as(x)

        mel_max_length = mel_lengths.max()
        mel_mask = torch.unsqueeze(sequence_mask(mel_lengths, mel_max_length), 1).type_as(x)

        input_padding_mask = ~x_mask.squeeze(1).bool().to(x.device)
        target_padding_mask = ~mel_mask.squeeze(1).bool().to(x.device)

        # text embedding
        x, __ = self.text_embedding(x)

        # Encoder
        x = self.encoder(x, input_padding_mask)

        # alignment
        # log_p_attn = self.alignment_module(
        #     text=x,
        #     feats=mel.transpose(1, 2),
        #     text_lengths=x_lengths,
        #     feats_lengths=mel_lengths,
        #     x_masks=input_padding_mask,
        # )

        # durations are estimated using a viterbi decoding, instead of through a forced-aligment
        # durations, bin_loss = viterbi_decode(log_p_attn, x_lengths, mel_lengths)

        # durations of tokens are predicted using the duration predictor
        duration_hat = self.duration_predictor(x.detach(), input_padding_mask)

        # Average pitch and energy values based on durations
        pitches = average_by_duration(gt_durations, pitches.unsqueeze(-1), x_lengths, mel_lengths)
        energies = average_by_duration(gt_durations, energies.unsqueeze(-1), x_lengths, mel_lengths)

        # variance predictors
        x, pitch_hat = self.pitch_predictor(x, input_padding_mask, pitches)
        x, energy_hat = self.energy_predictor(x, input_padding_mask, energies)

        # upsample to mel lengths
        mel_hat = self.feature_upsampler(
            hs=x, ds=gt_durations, h_masks=mel_mask.squeeze(1).bool(), d_masks=x_mask.squeeze(1).bool()
        )

        # Decoder
        mel_hat = self.decoder(mel_hat, target_padding_mask)
        mel_hat = self.feat_out(mel_hat)

        # Losses
        loss_coeffs = self.loss_coeffs
        duration_loss, pitch_loss, energy_loss = self.loss_criterion(
            d_outs=duration_hat.unsqueeze(-1),
            p_outs=pitch_hat.unsqueeze(-1),
            e_outs=energy_hat.unsqueeze(-1),
            ds=gt_durations.unsqueeze(-1),
            ps=pitches.unsqueeze(-1),
            es=energies.unsqueeze(-1),
            ilens=x_lengths,
        )
        mel_loss = self.mel_criterion(mel_hat, mel, mel_lengths)
        # forwardsum_loss = self.forwardsum_loss(log_p_attn, x_lengths, mel_lengths)
        # align_loss = forwardsum_loss + bin_loss
        loss = (
                (mel_loss * loss_coeffs.lambda_mel)
                + (duration_loss * loss_coeffs.lambda_duration)
                + (pitch_loss * loss_coeffs.lambda_pitch)
                + (energy_loss * loss_coeffs.lambda_energy)
        )
        loss_report = GeneratorTrainingOutput(
            loss=loss.cpu(),
            mel_loss=mel_loss.cpu(),
            duration_loss=duration_loss.detach().cpu(),
            pitch_loss=pitch_loss.detach().cpu(),
            energy_loss=energy_loss.detach().cpu())
        return loss, loss_report

    @torch.inference_mode()
    def synthesise(self,
                   x: torch.Tensor,
                   x_lengths: torch.Tensor,
                   d_factor: float = 1.0,
                   p_factor: float = 1.0,
                   e_factor: float = 1.0):
        """
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            sids (Optional[torch.LongTensor]): list of speaker IDs for each input sentence.
                shape: (batch_size,)
            d_factor (Optional[float]): scaler to control phoneme durations.
            p_factor (Optional[float]): scaler to control pitch.
            e_factor (Optional[float]): scaler to control energy.

        Returns:
            wav (torch.Tensor): generated waveform
                shape: (batch_size, T)
            durations: (torch.Tensor): predicted phoneme durations
                shape: (batch_size, max_text_length)
            pitch: (torch.Tensor): predicted pitch
                shape: (batch_size, max_text_length)
            energy: (torch.Tensor): predicted energy
                shape: (batch_size, max_text_length)
            rtf: (float): total Realtime Factor (inference_t/audio_t)
        """
        am_t0 = perf_counter()

        x_max_length = x_lengths.max()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_max_length), 1).to(x.dtype)
        x_mask = x_mask.to(x.device)
        input_padding_mask = ~x_mask.squeeze(1).bool().to(x.device)

        # text embedding
        x, __ = self.text_embedding(x)

        # Encoder
        x = self.encoder(x, input_padding_mask)

        # duration predictor
        # NOTE: this could be substituted by the durations from espeak during inference
        durations = self.duration_predictor.infer(x, input_padding_mask, factor=d_factor)

        # variance predictors
        # NOTE: pitches could also be substituted by pitch values from mbrola
        x, pitch = self.pitch_predictor.infer(x, input_padding_mask, p_factor)
        if self.energy_predictor is not None:
            x, energy = self.energy_predictor.infer(x, input_padding_mask, e_factor)
        else:
            energy = None

        y_lengths = durations.sum(dim=1)
        y_max_length = y_lengths.max()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).type_as(x)
        target_padding_mask = ~y_mask.squeeze(1).bool()

        y = self.feature_upsampler(
            hs=x, ds=durations, h_masks=y_mask.squeeze(1).bool(), d_masks=x_mask.squeeze(1).bool()
        )

        # Decoder
        mel = self.decoder(y, target_padding_mask)
        mel = self.feat_out(mel)
        am_infer = (perf_counter() - am_t0) * 1000

        return {
            "mels": mel.detach().cpu(),
            "durations": durations.detach().cpu(),
            "pitch": pitch.detach().cpu(),
            "energy": energy.detach().cpu(),
            "am_infer": am_infer,
        }

    @torch.inference_mode()
    def synthesise_one(self,
                       x: torch.Tensor,
                       d_factor: float = 1.0,
                       p_factor: float = 1.0,
                       e_factor: float = 1.0):

        x = x.unsqueeze(0)
        x_length = torch.tensor([torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)])
        return self.synthesise(x, x_length, d_factor, p_factor, e_factor)
