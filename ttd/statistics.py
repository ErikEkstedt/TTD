import seaborn as sns
from os.path import join
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torchaudio import load as load_audio

from ttd.basebuilder import create_builders, add_builder_specific_args
from ttd.vad_helpers import vad_to_dialog_vad_states, vad_to_frames, vad_to_ipu
from ttd.utils import find_island_idx_len, get_duration_sox


def extract_turn_floor_offset(vad, ipu_thresh=-1):
    """
    Extract the turn-floor offset based on VAD-dialog-states.

    :Caution: Use with smaller frame sizes for more accurate assesement

    1. Converts vad to states
    2. Iterate over the states
        - Find silences
            - omit pauses but add gap durations
            - ONLY states following SPEAKER - SILENCE - OTHER-SPEAKER is used
        - Find Overlaps
            - ONLY states following SPEAKER - OVERLAP - OTHER-SPEAKER is used

    SILENCES:
        NOT states:
            SPEAKER - OVERLAP - SILENCE - OTHER-SPEAKER
    OVERLAPS:
        Not states:
            SPEAKER - SILENCE - OVERLAP - OTHER-SPEAKER
        which would be a moment of competition for the turn-floor

    Return:
        turn_floor_offsets:     List
    """
    turn_floor_offsets = []
    if ipu_thresh > 0:
        vad = vad_to_ipu(vad, ipu_thresh)
    ds_vad = vad_to_dialog_vad_states(vad)
    idx, dur, val = find_island_idx_len(ds_vad)  #
    for i, (state, state_duration) in enumerate(zip(val, dur)):
        if i == 0 or i == len(val) - 1:  # skip start/end
            continue
        prev_state = val[i - 1]
        next_state = val[i + 1]
        if state == 1:  #  current state is silence
            if prev_state in [0, 3]:
                if prev_state != next_state:  # not pause
                    turn_floor_offsets.append(state_duration.item())
        elif state == 2:  # overlap
            if prev_state in [0, 3]:
                if prev_state != next_state:  # actual turn-shift after overlap
                    turn_floor_offsets.append(-state_duration.item())
    return turn_floor_offsets


def extract_pauses(vad):
    pauses = []
    ds_vad = vad_to_dialog_vad_states(vad)
    idx, dur, val = find_island_idx_len(ds_vad)  #
    for i, (state, state_duration) in enumerate(zip(val, dur)):
        if i == 0 or i == len(val) - 1:  # skip start/end
            continue
        prev_state = val[i - 1]
        next_state = val[i + 1]
        if state == 1:  #  current state is silence
            if prev_state in [0, 3]:
                if prev_state == next_state:  # pause
                    pauses.append(state_duration.item())
    return pauses


def extract_state_duration(vad, state="silence"):
    """
    Extract the duration of a VAD-dialog `state`: 'silences', 'speaker' or 'overlap'

    Returns:
        duration:       Torch.Tensor
    """
    state_idx = {"speaker": 0, "silence": 1, "overlap": 2}
    assert state in state_idx

    i = state_idx[state]
    ds_vad = vad_to_dialog_vad_states(vad)
    idx, dur, val = find_island_idx_len(ds_vad)  #
    if state == "speaker":
        dur = torch.cat((dur[val == 0], dur[val == 3]))
    else:
        dur = dur[val == i]
    return dur


def builder_vad_stats(builder, vad_step_time=0.01, ipu_frame_thresh=-1):
    filenames = builder.val_filepaths + builder.train_filepaths + builder.test_filepaths

    turn_floor_offsets = []
    all_pauses = []
    for filename in tqdm(filenames):
        vad = torch.load(join(builder.vad_root, filename.replace(".json", ".pt")))
        audio_path = builder.get_audio_path(filename.replace(".json", ""))
        y, sr = load_audio(audio_path)
        duration = get_duration_sox(audio_path)
        vad = vad_to_frames(
            vad,
            n_samples=y.shape[-1],
            step_time=vad_step_time,
            sr=sr,
        )
        all_pauses += extract_pauses(vad)
        turn_floor_offsets += extract_turn_floor_offset(vad, ipu_frame_thresh)

    time_scale = 1000 * vad_step_time
    return (
        torch.tensor(turn_floor_offsets) * time_scale,
        torch.tensor(all_pauses) * time_scale,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser
    from os import makedirs
    import matplotlib.pyplot as plt

    parser = ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=str,
        default=["maptask"],
    )
    parser.add_argument("--vad_step_time", type=float, default=0.01)
    parser.add_argument("--ipu_frame_thresh", type=float, default=-1)
    parser.add_argument("--savepath", type=str, default=None)
    datasets = parser.parse_args().datasets
    parser = add_builder_specific_args(parser, datasets)  # add for all builders
    args = parser.parse_args()
    builders = create_builders(vars(args))

    for builder in builders:
        builder.prepare_vad()
        tfo, pauses = builder_vad_stats(
            builder,
            vad_step_time=args.vad_step_time,
            ipu_frame_thresh=args.ipu_frame_thresh,
        )
        fig, ax = plt.subplots(2, 1)
        ax[0].hist(tfo.unsqueeze(0), range=(-2000, 3000), bins=100)
        ax[0].set_xlabel("ms")
        ax[1].hist(pauses.unsqueeze(0), range=(0, 3000), bins=100)
        ax[1].set_xlabel("ms")
        fig.suptitle(
            builder.NAME
            + f"vad_step: {args.vad_step_time}, ipu_frames: {args.ipu_frame_thresh*args.vad_step_time}"
        )
        plt.pause(0.01)

        if args.savepath is not None:
            makedirs(args.savepath, exist_ok=True)
            torch.save(tfo, join(args.savepath, builder.NAME + "_tfo.pt"))
            torch.save(pauses, join(args.savepath, builder.NAME + "_pauses.pt"))
            fig.savefig(join(args.savepath, builder.NAME + ".png"))
            print(f"Saved {builder.NAME} data -> ", args.savepath)

    input("Enter to close")
