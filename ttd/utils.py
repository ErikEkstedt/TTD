import math
import json
from subprocess import check_output
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


def get_duration_sox(fpath):
    out = (
        check_output(f"sox --i {fpath}", shell=True).decode("utf-8").strip().split("\n")
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
        check_output(f"sox --i {fpath}", shell=True).decode("utf-8").strip().split("\n")
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
