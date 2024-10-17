from typing import Optional

import torch
import torch.nn.functional as F

from lightspeech.core.modules import LayerNorm
from lightspeech.core.modules import SepConv1d


class VariancePredictor(torch.nn.Module):

    def __init__(self, idim: int, n_layers: int = 2, n_chans: int = 256, out: int = 1, kernel_size: int = 3,
                 dropout_rate: float = 0.5, offset: float = 1.0):
        super(VariancePredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                SepConv1d(in_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
                torch.nn.ReLU(),
                LayerNorm(n_chans),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, out)

    def _forward(self, xs: torch.Tensor, is_inference: bool = False, is_log_output: bool = False) -> torch.Tensor:
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)  # (B, Tmax)

        if is_inference and is_log_output:
            #     # NOTE: calculate in linear domain
            xs = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long()  # avoid negative value
        return xs

    def forward(self, xs: torch.Tensor, x_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).

        """
        xs = self._forward(xs)
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)
        return xs

    def inference(self, xs: torch.Tensor, is_log_output: bool = False) -> torch.Tensor:
        """Inference duration.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).

        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).

        """
        return self._forward(xs, is_inference=True, is_log_output=is_log_output)


class SpeechFeaturePredictor(torch.nn.Module):
    """Either used to predict Pitch or Energy"""
    bins: torch.Tensor

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 log_scale: bool,
                 min_val: int,
                 max_val: int,
                 variance_predictor: VariancePredictor):
        super().__init__()
        self.hidden_dim = hidden_dim

        if log_scale:
            bins = torch.exp(torch.linspace(torch.log(torch.tensor(min_val)),
                                            torch.log(torch.tensor(max_val)),
                                            hidden_dim - 1))
        else:
            bins = torch.linspace(min_val, max_val, hidden_dim - 1)
        self.register_buffer("bins", bins)

        self.predictor: VariancePredictor = variance_predictor(input_dim)
        self.embed = torch.nn.Linear(hidden_dim, hidden_dim)

    def to_one_hot(self, x: torch.Tensor):
        quantize = torch.bucketize(x, self.bins).to(device=x.device)  # .cuda()
        return F.one_hot(quantize.long(), self.hidden_dim).float()

    def forward(self, xs: torch.Tensor, x_masks: torch.Tensor, series: torch.Tensor):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, hidden_dim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
            series (ByteTensor, optional): Batch of either pitch or energy series (B, Tmax).

        Returns:
            Tensor: input + pitch embedding (B, Tmax, hidden_dim).
            Tensor: Batch of predicted pitch or energy (B, Tmax, hidden_dim).
        """
        with torch.no_grad():
            one_hot_encoded = self.to_one_hot(series)

        predicted_series = self.predictor(xs, x_masks)
        # we're using teacher forcing: we're not using the prediction for embeddings yet
        embedded_series = self.embed(one_hot_encoded)
        return xs + embedded_series, predicted_series

    @torch.inference_mode()
    def infer(self, xs: torch.Tensor, alpha: float = 1.0):
        predicted_series = self.predictor.inference(xs, False)
        predicted_series = predicted_series * alpha
        one_hot_encoded = self.to_one_hot(predicted_series)
        embedded_series = self.embed(one_hot_encoded)
        return xs + embedded_series, predicted_series