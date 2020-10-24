import math
import torch
from ttd.utils import find_island_idx_len


def vad_from_word_level(dialog_words, duration):
    """ Hardcoded for dialog / two speakers """
    vad = [[], []]
    for dw in dialog_words:
        speaker_id = dw["speaker_id"]
        assert isinstance(speaker_id, int), "speaker_id must be an int"

        start = dw["start"] / duration
        end = dw["end"] / duration
        vad[speaker_id].append((start, end))
    return vad


def percent_to_onehot(vad, n_frames, pad=0):
    v = torch.zeros(len(vad), n_frames)
    for channel in range(len(vad)):
        for s, e in vad[channel]:
            ss = int(s * n_frames) - pad
            if ss < 0:
                ss = 0
            v[channel, ss : math.ceil(e * n_frames) + pad] = 1
    return v


def vad_to_frames(vad, n_samples, step_time, sr, pad=0):
    hop_length = int(step_time * sr)
    # target_frames = math.ceil(n_samples / hop_length)
    target_frames = math.floor(n_samples / hop_length) + 1
    return percent_to_onehot(vad, target_frames, pad=pad)


def vad_to_dialog_vad_states(vad):
    """Vad to the full state of a 2 person vad dialog
    0: only speaker 0
    1: none
    2: both
    3: only speaker 1
    """
    assert vad.ndim >= 2
    if vad.ndim == 2:
        return (2 * vad[1] - vad[0]).long() + 1
    else:
        return (2 * vad[:, 1] - vad[:, 0]).long() + 1


# should this be in VAD?
def vad_to_ipu(vad, ipu_frame_thresh):
    """
    All silences in a single channel that are shorter than `ipu_frame_thresh` are filled.
    """
    ipu = vad.clone()
    for channel in range(2):
        start, dur, val = find_island_idx_len(vad[channel])
        sil_start = start[val == 0]
        sil_dur = dur[val == 0]

        fill_these = torch.where(sil_dur <= ipu_frame_thresh)[0]
        if len(fill_these) > 0:
            # if fill_these[0] == 0: # omit the start
            #     fill_these = fill_these[1:]
            fill_durs = sil_dur[fill_these]
            fill_starts = sil_start[fill_these]
            fill_ends = fill_starts + fill_durs
            for s, e in zip(fill_starts, fill_ends):
                ipu[channel, s:e] = 1
    return ipu


# should this be in VAD?
def ipu_to_turns(ipu):
    # ipus separated by mutual silence are condensed into turns
    turns = ipu.clone()
    ipu_states = vad_to_dialog_vad_states(ipu)
    # state: 1 = both = Mutual silence
    for channel in range(2):
        starts, dur, val = find_island_idx_len(ipu[channel])
        starts = starts[val == 0]  # start of silence
        dur = dur[val == 0]
        for s, d in zip(starts, dur):
            tmp_states = ipu_states[s : s + d]
            # fill silences for ipus in one channel if there is only 'mutual silence' in the pauses
            if not (tmp_states != 1).sum() > 0:
                turns[channel, s : s + d] = 1
    return turns


# should this be in VAD?
def omit_inside_overlap(turn):
    new_turn = turn.clone()
    # state 2: both = overlap
    turn_states = vad_to_dialog_vad_states(turn)
    starts, dur, val = find_island_idx_len(turn_states)
    # single - both - single
    # 0 - 2 - 0  | channel 1 should be omitted
    # 3 - 2 - 3  | channel 0 should be omitted
    omitted = 0
    for i, (prev, cur, post) in enumerate(zip(val[:-2], val[1:-1], val[2:])):
        current_index = i + 1
        if cur == 2:
            if prev == post == 0:
                s = starts[current_index]
                e = s + dur[current_index]
                new_turn[1, s:e] = 0
                omitted += 1
            elif prev == post == 3:
                s = starts[current_index]
                e = s + dur[current_index]
                new_turn[0, s:e] = 0
                omitted += 1
    # print("omitted: ", omitted)
    return new_turn
