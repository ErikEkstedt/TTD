import math
import torch
import torch.nn as nn
import librosa
from pysptk import swipe, rapt

from ttd.utils import find_island_idx_len

import torchaudio
import torchaudio.transforms as AT
from ttd.vad_helpers import percent_to_onehot
from tqdm import tqdm


def F0_swipe(
    waveform,
    hop_length=None,
    sr=None,
    hop_time=None,
    f_min=60,  # default in swipe
    f_max=240,  # default in swipe
    threshold=0.5,  # custom defualt (0.3 in swipe)
):
    if hop_length is not None:
        hopsize = hop_length
    else:
        hopsize = int(sr * hop_time)

    if waveform.ndim == 1:
        return torch.from_numpy(
            swipe(
                waveform.contiguous().double().numpy(),
                fs=sr,
                hopsize=hopsize,
                min=f_min,
                max=f_max,
                threshold=threshold,
                otype="f0",
            )
        ).float()
    elif waveform.ndim == 2:  # (B, N)
        f0 = []
        for audio in waveform:
            f0.append(
                torch.from_numpy(
                    swipe(
                        audio.contiguous().double().numpy(),
                        fs=sr,
                        hopsize=hopsize,
                        min=f_min,
                        max=f_max,
                        threshold=threshold,
                        otype="f0",
                    )
                ).float()
            )
        return torch.stack(f0)


def fix_size_last_dim(x, N):
    """
    Make sure that the length of the last dim of x is N.
    If N is longer we append the last value if it is shorter we
    remove superfluous frames.
    """
    if x.shape[-1] != N:
        diff = N - x.shape[-1]
        if diff > 0:
            if x.ndim > 1:
                x = torch.cat((x, x[:, -1].unsqueeze(-1)), dim=-1)
            else:
                x = torch.cat((x, x[-1]), dim=-1)
        else:  # diff is negative
            x = x[..., :diff]
    return x


def clean_f0(f0, f_min=None):
    """
    removes single valued frames i.e. [0, 0, 123, 0 0] -> [0, 0, 0, 0, 0]. The f0-extractor is sensitive and sometimes
    classifies noise as f0.
    """

    # the minimum value for f0 is sometimes triggered (even for 10-60Hz) and seems to be mostly due to noise
    # they seem very out of distribution compared to all the regular f0 peaks and valleys.
    # the are constant for several frames
    if f_min is not None:
        f0[f0 == f_min] = 0

    if f0.ndim == 1:
        discrete = (f0 > 0).float()
        idx, dur, val = find_island_idx_len(discrete)
        dur = dur[val == 1]  # all duration 1
        idx = idx[val == 1][dur == 1]  # index for duration 1
        f0[idx] = 0
    elif f0.ndim == 2:
        for i, ch_f0 in enumerate(f0):
            discrete = (ch_f0 > 0).float()
            idx, dur, val = find_island_idx_len(discrete)
            dur = dur[val == 1]  # all duration 1
            idx = idx[val == 1][dur == 1]  # index for duration 1
            f0[i, idx] = 0
    return f0


def pitch_statistics(f0):
    voiced = (f0 > 0).float()
    if f0.ndim == 1:
        m = f0[voiced == 1].mean()
        s = f0[voiced == 1].std()
    else:
        m = []
        s = []
        for f, v in zip(f0, voiced):
            m.append(f[v == 1].mean())
            s.append(f[v == 1].std())
        m = torch.stack(m, dim=-1)
        s = torch.stack(s, dim=-1)
    return m, s


def f0_z_normalize(f0, mean, std, eps=1e-8):
    assert mean.size() == std.size()
    voiced = f0 > 0
    n = f0.clone()
    if f0.ndim == 1:
        n[voiced] = (f0[voiced] - mean) / (std + eps)
    else:
        for i, (f, v) in enumerate(zip(f0, voiced)):
            n[i, v] = (f[v] - mean[i]) / (std[i] + eps)
    return n


def interpolate_forward(f0, voiced):
    f = f0.clone()
    for i, v in enumerate(voiced):
        idx, dur, val = find_island_idx_len(v.float())
        # unvoiced -> value prior unvoiced
        dur = dur[val == 0]
        idx = idx[val == 0]
        for ii, dd in zip(idx, dur):
            if ii - 1 < 0:
                tmp_val = f[i, 0]
            else:
                tmp_val = f[i, ii - 1]
            f[i, ii : ii + dd] = tmp_val
    return f


class F0(object):
    def __init__(
        self,
        sr=8000,
        hop_time=0.01,
        f0_min=60,
        f0_max=300,
        f0_threshold=0.5,
    ):
        self.sr = sr
        self.hop_time = hop_time
        self.hop_length = int(hop_time * sr)

        # F0 specific
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_threshold = f0_threshold
        self.eps = 1e-8

    def __repr__(self):
        s = self.__class__.__name__
        s += f"\n\tsr={self.sr},"
        s += f"\n\thop_time={self.hop_time},"
        s += f"\n\tf0_min={self.f0_min},"
        s += f"\n\tf0_max={self.f0_max},"
        s += f"\n\tf0_threshold={self.f0_threshold},"
        s += f"\n\tf0_max={self.f0_max},"
        return s

    def __call__(self, waveform):
        """
        :param y:   torch.Tensor, waveform (n_samples,) or (B, n_samples)
        Return:
            dict,  f0, mean, std
        """
        n_frames = int(waveform.shape[-1] // self.hop_length) + 1

        f0 = F0_swipe(
            waveform,
            sr=self.sr,
            hop_time=self.hop_time,
            threshold=self.f0_threshold,
            f_min=self.f0_min,
        )
        f0 = fix_size_last_dim(f0, n_frames)
        f0 = clean_f0(f0, f_min=self.f0_min)
        m, s = pitch_statistics(f0)
        return {"f0": f0, "mean": m, "std": s}


if __name__ == "__main__":
    import torchaudio
    import torchaudio.transforms as AT
    import torchaudio.functional as AF
    import time

    from ttd.utils import get_duration_sox

    import matplotlib.pyplot as plt

    mask = True
    sr = 8000
    hop_time = 0.01
    n_fft = int(0.05 * sr)
    hop_length = int(hop_time * sr)

    pitcher = F0(sr=sr, hop_time=hop_time, f0_min=60, f0_max=400, f0_threshold=0.3)
    melspecter = AT.MelSpectrogram(
        sr, n_fft=n_fft, hop_length=hop_length, f_min=60, f_max=sr // 2, n_mels=80
    )
    amp = AT.AmplitudeToDB()

    wav_path = "data/maptask/audio/q1ec6.wav"
    dur = round(get_duration_sox(wav_path), 1)

    t = time.time()
    waveform, tmp_sr = torchaudio.load(wav_path, normalization=True)
    vad_percent = torch.load("data/maptask/VAD/q1ec6.pt")  # percentage
    if tmp_sr != sr:
        waveform = AT.Resample(orig_freq=tmp_sr, new_freq=sr)(waveform)
    sp = melspecter(waveform)  # B, n_mels, T
    sp = amp(sp)
    if mask:
        vad_samples = percent_to_onehot(vad_percent, waveform.shape[-1])
        pitch = pitcher(waveform * vad_samples)
    else:
        pitch = pitcher(waveform)
    voiced = pitch["f0"] != 0
    p = f0_z_normalize(pitch["f0"], pitch["mean"], pitch["std"])
    p = interpolate_forward(p, voiced)
    t = round(time.time() - t, 2)
    print(f"load-f0-spec extraction took {t} for {dur}s audio")

    torch.save(pitch, "test.pt")

    n_frames = pitch["f0"].shape[-1]

    vad = percent_to_onehot(vad_percent, n_frames)

    s = 0
    d = 200
    fig, ax = plt.subplots(2, 1)
    for s in torch.linspace(0, n_frames, d // 2):
        s = s.long()
        e = s + d
        for a in ax:
            a.cla()
        ax[0].imshow(sp[0, :, s:e], aspect="auto", origin="lower")
        # ax[1].plot(pitch["f0"][0, s:e])
        ax[1].plot(p[0, s:e])
        ax[1].set_ylim([-4, 4])
        # ax[1].plot(vad[0, s:e] * p.max())
        ax[1].set_xlim([0, d])
        plt.tight_layout()
        plt.pause(0.001)
        input()
