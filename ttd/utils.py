import math
import json
import subprocess
from os.path import split, join

import torch


def get_run_dir(current_file, run_dir="runs"):
    dir_path = split(current_file)[0]
    if dir_path == "":
        save_dir = run_dir
    else:
        save_dir = join(dir_path, run_dir)
    return save_dir


def read_json(path, encoding="utf8"):
    with open(path, "r", encoding=encoding) as f:
        data = json.loads(f.read())
    return data


def write_json(data, filename):
    with open(filename, "w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False)


def read_txt(path, encoding="utf-8"):
    data = []
    with open(path, "r", encoding=encoding) as f:
        for line in f.readlines():
            data.append(line.strip())
    return data


def write_txt(txt, name):
    """
    Argument:
        txt:    list of strings
        name:   filename
    """
    with open(name, "w") as f:
        f.write("\n".join(txt))


def wget(url, to):
    cmd = ["wget", "-O", to, url]
    # system(cmd)
    subprocess.call(cmd)


def get_duration_sox(fpath):
    out = (
        subprocess.check_output(f"sox --i {fpath}", shell=True)
        .decode("utf-8")
        .strip()
        .split("\n")
    )
    for line in out:
        if line.lower().startswith("duration"):
            l = [f for f in line.split(" ") if not f == ""]
            duration = l[2].split(":")
            hh, mm, ss = duration
            total = int(hh) * 60 * 60 + int(mm) * 60 + float(ss)
    return total


def get_sample_rate_sox(fpath):
    out = (
        subprocess.check_output(f"sox --i {fpath}", shell=True)
        .decode("utf-8")
        .strip()
        .split("\n")
    )
    for line in out:
        if line.lower().startswith("sample rate"):
            l = [f for f in line.split(" ") if not f == ""]
            return int(l[-1])


def percent_to_onehot(vad, n_frames, pad=0):
    v = torch.zeros(len(vad), n_frames)
    for channel in range(len(vad)):
        for s, e in vad[channel]:
            ss = int(s * n_frames) - pad
            if ss < 0:
                ss = 0
            v[channel, ss : math.ceil(e * n_frames) + pad] = 1
    return v


def get_speaker_shift_indices(input_ids, sp1_idx, sp2_idx):
    inp = input_ids.clone()
    inp[inp == sp2_idx] = sp1_idx
    sp_b, sp_inds = torch.where(inp == sp1_idx)  # all speaker 1 tokens
    return (sp_b, sp_inds)


def get_turn_shift_indices(input_ids, sp1_idx, sp2_idx):
    ts_bs, ts_inds = get_speaker_shift_indices(input_ids, sp1_idx, sp2_idx)
    ts_inds = ts_inds - 1  # turn-shift are
    ts_bs = ts_bs[ts_inds != -1]
    ts_inds = ts_inds[ts_inds != -1]
    return (ts_bs, ts_inds)


def get_turns(input_ids, sp1_idx, sp2_idx):
    assert input_ids.ndim == 2
    sp_b, sp_inds = get_speaker_shift_indices(input_ids, sp1_idx, sp2_idx)
    turns = []
    for b in range(input_ids.shape[0]):
        turns.append(sp_inds[sp_b == b].unfold(0, 2, 1))
    return turns


def get_positive_and_negative_indices(input_ids, sp1_idx, sp2_idx, pad_idx):
    """
    Finds positive and negative indices for turn-shifts.

    * Positive turn-shifts are the indices prior to a <speaker1/2> token
    * Negative turn-shifts are all other indices (except pad_tokens)

    Returns:
        turn_shift_indices:     tuple, (batch, inds) e.g.  input_ids[turn_shift_indices]
        non_turn_shift_indices: tuple, (batch, inds) e.g.  input_ids[non_turn_shift_indices]
    """
    (ts_bs, ts_inds) = get_turn_shift_indices(input_ids, sp1_idx, sp2_idx)
    bp, indp = torch.where(input_ids != pad_idx)  # all valid places

    # TODO:
    # Remove the speaker-id tokens from negatives?

    neg_bs, neg_inds = [], []
    for i in bp.unique():
        neg_ind = indp[bp == i]  # valid indices (not pad) # [1:]  # omit 0
        ts = ts_inds[ts_bs == i]  # turn-shifts in batch i
        neg_ind[ts] = -1  # mark these
        neg_ind = neg_ind[neg_ind != -1]
        neg_bs.append(torch.ones_like(neg_ind) * i)
        neg_inds.append(neg_ind)

    neg_bs = torch.cat(neg_bs)
    neg_inds = torch.cat(neg_inds)
    return (ts_bs, ts_inds), (neg_bs, neg_inds)


def find_island_idx_len(x):
    """
    Finds patches of the same value.

    starts_idx, duration, values = find_island_idx_len(x)

    e.g:
        ends = starts_idx + duration

        s_n = starts_idx[values==n]
        ends_n = s_n + duration[values==n]  # find all patches with N value

    """
    assert x.ndim == 1
    n = len(x)
    y = x[1:] != x[:-1]  # pairwise unequal (string safe)
    i = torch.cat((torch.where(y)[0], torch.tensor(n - 1).unsqueeze(0))).long()
    it = torch.cat((torch.tensor(-1).unsqueeze(0), i))
    dur = it[1:] - it[:-1]
    idx = torch.cumsum(torch.cat((torch.LongTensor([0]), dur)), dim=0)[:-1]  # positions
    return idx, dur, x[i]


def fscores(pos, neg):
    """fscores.

    :param pos:     torch.tensor, containing the predictions for all positive classes
    :param neg:     torch.tensor, containing the predictions for all negative classes
    """

    def get_weighted(xp, xn, p, n):
        return (p * xp + n * xn) / (n + p)

    cutoff = torch.linspace(0.01, 0.99, 99).unsqueeze(1)
    n_pos = pos.nelement()
    n_neg = neg.nelement()
    total = n_pos + n_neg

    tp = (pos >= cutoff).sum(dim=-1, dtype=torch.float)
    fn = n_pos - tp  # fn = (pos < cutoff).sum(dim=-1, dtype=torch.float)
    fp = (neg >= cutoff).sum(dim=-1, dtype=torch.float)
    tn = n_neg - fp  # tn = (neg < cutoff).sum(dim=-1, dtype=torch.float)

    ################################
    # Postive (regular)
    # Precision: PPV
    p_den = tp + fp
    p = torch.zeros_like(p_den)
    p[p_den > 0] = tp[p_den > 0] / p_den[p_den > 0]

    # Recall:  TPR
    r_den = tp + fn
    r = torch.zeros_like(r_den)
    r[r_den > 0] = tp[r_den > 0] / r_den[r_den > 0]

    # F1-score
    den = p + r
    f1 = torch.zeros_like(den)
    f1[den > 0] = 2 * (p[den > 0] * r[den > 0]) / den[den > 0]

    ################################
    # NEGATIVE
    # Precision: PPV
    pn_den = tn + fn
    pn = torch.zeros_like(pn_den)
    pn[pn_den > 0] = tn[pn_den > 0] / pn_den[pn_den > 0]

    # Recall:  TPR
    rn_den = tn + fp
    rn = torch.zeros_like(rn_den)
    rn[rn_den > 0] = tn[rn_den > 0] / rn_den[rn_den > 0]

    # F1-score
    nden = pn + rn
    nf1 = torch.zeros_like(nden)
    nf1[nden > 0] = 2 * (pn[nden > 0] * rn[nden > 0]) / nden[nden > 0]

    # Accuracies
    # Acc
    acc = (tp + tn) / (tp + tn + fp + fn)

    # Bacc / UAR
    tnr_den = tn + fp
    tnr = torch.zeros_like(tnr_den)
    tnr[tnr_den > 0] = tn[tnr_den > 0] / tnr_den[tnr_den > 0]
    bacc = (r + tnr) / 2

    # Concatenate Fscore, Recall, precision
    f1w = get_weighted(f1, nf1, n_pos, n_neg)
    pw = get_weighted(p, pn, n_pos, n_neg)
    rw = get_weighted(r, rn, n_pos, n_neg)

    f1 = torch.stack((nf1, f1, f1w), dim=-1)
    p = torch.stack((pn, p, pw), dim=-1)
    r = torch.stack((rn, r, rw), dim=-1)

    return {
        "f1": f1,
        "precision": p,
        "recall": r,
        "acc": acc,
        "bacc": bacc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "cutoffs": cutoff.squeeze(-1),
    }
