import math
import numpy as np
import torch

from ttd.vad_helpers import (
    vad_to_frames,
    vad_to_ipu,
    vad_to_frames,
    ipu_to_turns,
    omit_inside_overlap,
)
from ttd.utils import find_island_idx_len


def find_next_word_index_start(i, x):
    """
    start from the next token in the data and check if it belonged to the same word as the current.
    If the next token belongs to the same word as the current token we check one step further until we
    reach the first token of the next word.
    """
    cur_word_index = x["word_ids"][i]
    cur_end = x["ends"][i]
    cur_speaker = x["speaker_ids"][i]
    max_i = len(x["speaker_ids"])
    is_last = False
    pause = 0  # binary label of pause: 0-false, 1-true
    j = i + 1
    while True:
        if x["word_ids"][j] != cur_word_index:
            t = x["starts"][j] - cur_end
            if t < 0:
                t = 0.0
            if x["speaker_ids"][j] == cur_speaker:
                pause = 1
            break
        j += 1
        if j == max_i:
            is_last = True
            return None, None, is_last
    return t, pause, is_last


def join_consecutive_utterances(dialog, sort_key="start"):
    """ join consecutive utterances by the same speaker """
    dialog.sort(key=lambda x: x[sort_key])
    new_dialog = []
    last_utt = dialog[0].copy()
    for d in dialog[1:]:
        if d["speaker_id"] == last_utt["speaker_id"]:
            for k in last_utt.keys():
                if k == "text":
                    last_utt["text"] += " " + d["text"]
                else:
                    last_utt[k] += d[k]
        else:
            new_dialog.append(last_utt)
            last_utt = d.copy()
    return new_dialog


def get_frame_word_level_lookup(word_level_dialog, n_frames, duration, fill="start"):
    """
    converts the time of a word to a frame number. Associates that particular frame with and index in
    word_level_dialog.

    When operating on a frame-wise basis the `frame_array_lookup` may be used to access corresponding word entries.
    """
    assert fill in ["start", "end", "all"]
    frame_array_lookup = np.ones((2, n_frames), dtype=np.int) * -1

    for word_level_index, dw in enumerate(word_level_dialog):
        speaker_id = dw["speaker_id"]
        assert isinstance(speaker_id, int), "speaker_id must be an int"
        start = dw["start"] / duration
        end = dw["end"] / duration
        try:
            if fill == "start":
                ss = int(start * n_frames)  # percent to frame
                frame_array_lookup[speaker_id, ss] = word_level_index
            elif fill == "end":
                ee = math.ceil(end * n_frames)
                frame_array_lookup[speaker_id, ee] = word_level_index
            else:  # all
                ss = int(start * n_frames)  # percent to frame
                ee = math.ceil(end * n_frames)
                frame_array_lookup[speaker_id, ss:ee] = word_level_index
        except:
            ss = int(start * n_frames)  # percent to frame
            ee = math.ceil(end * n_frames)
            print("n_frames: ", n_frames)
            print("start: ", ss)
            print("end: ", ee)
            __import__("ipdb").set_trace()
    return frame_array_lookup


def get_word_level_turns(word_level_dialog, frame_array_lookup, new_turn):
    word_level_turns = []
    for channel in range(2):
        tmp_text = []
        tmp_starts = []
        tmp_ends = []
        # last_start = 0
        last_start_time = 0
        last_activity = 0
        for i, (lookup_ind, activity) in enumerate(
            zip(frame_array_lookup[channel], new_turn[channel])
        ):
            if activity == 1:  # must be a valid word we are inside a valid turn
                if last_activity == 0:  # new turn -> record start
                    last_start = i
                if lookup_ind != -1:
                    dw = word_level_dialog[lookup_ind]
                    tmp_text.append(dw["word"])
                    tmp_starts.append(dw["start"])
                    tmp_ends.append(dw["end"])

            else:  # silence: activity == 0
                if (
                    last_activity == 1
                ):  # we should have collected all the words in the prev turn
                    word_level_turns.append(
                        {
                            "text": " ".join(tmp_text),
                            "starts": tmp_starts,  # all word starts
                            "ends": tmp_ends,  # all word ends
                            "speaker_id": channel,
                            "start": tmp_starts[0],  # the time of turn start
                        }
                    )
                    tmp_text = []
                    tmp_starts = []
                    tmp_ends = []
            last_activity = activity
    word_level_turns = join_consecutive_utterances(word_level_turns, sort_key="start")

    # sanity check for turns
    turns_in_a_row = 0
    last_speaker = word_level_turns[0]["speaker_id"]
    for i, wt in enumerate(word_level_turns[1:]):
        if wt["speaker_id"] == last_speaker:
            turns_in_a_row += 1
        last_speaker = wt["speaker_id"]

    if not turns_in_a_row == 0:
        return None
    return word_level_turns


def word_level_to_turns(
    word_level_dialog,
    vad,
    duration,
    sr,
    vad_step_time=0.01,
    vad_pad=0,
    ipu_thresh=0.2,
):
    """
    Extract turn level dialog from word_level dialogs (must include time for each word)
    """
    # how many vad frames for each ipu_frames.  i.e 20 frames
    ipu_frame_thresh = int(ipu_thresh / vad_step_time)
    n_samples = int(duration * sr)
    vad = vad_to_frames(
        vad,
        # n_samples=y.shape[-1],
        n_samples=n_samples,
        step_time=vad_step_time,
        sr=sr,
        pad=vad_pad,
    )

    # Operates only on VAD ------------
    ipu = vad_to_ipu(vad, ipu_frame_thresh)  # Words -> IPUs
    turn = ipu_to_turns(ipu)  # IPU -> Turns
    new_turn = omit_inside_overlap(
        turn
    )  # Remove overlaps that are completely inside the other channels turn
    # ---------------------------------

    # get all the words associated with the new turns
    frame_array_lookup = get_frame_word_level_lookup(
        word_level_dialog, n_frames=vad.shape[-1], duration=duration, fill="start"
    )

    # combine the words from the lookup to Turn-level dialog data
    word_level_turns = get_word_level_turns(
        word_level_dialog, frame_array_lookup, new_turn
    )
    return word_level_turns
