from os.path import join
import re

import torch
import numpy as np

from ttd.utils import write_json, find_island_idx_len
from ttd.dialog_helpers import join_consecutive_utterances


def remove_punctuation_capitalization(s, punct_token=None):
    s = s.lower()
    s = re.sub("\((.*)\)", r"\1", s)  # remove parenthesis (but keep the content)
    s = re.sub("\[(.*)\]", r"\1", s)  # remove parenthesis (but keep the content)
    s = re.sub("\:\)\s", "", s)  # remove smiley
    s = re.sub("\:\(\s", "", s)  # remove smiley
    s = re.sub("\:\/\s", "", s)  # remove smiley
    s = re.sub("\:\|\s", "", s)  # remove smiley
    s = re.sub("\:D\s", "", s)  # remove smiley
    s = re.sub("\:p\s", "", s)  # remove smiley
    s = re.sub("\:P\s", "", s)  # remove smiley
    s = re.sub("\:\)", "", s)  # remove smiley
    s = re.sub(r"^[.,!?:;]+", "", s)  # Remove punctuation at start of turn
    s = re.sub('"', "", s)
    s = re.sub("[`´]+", "'", s)
    s = re.sub(r"[\n\t]", "", s)  # Remove newline and tab
    s = re.sub(r"[:;]", "", s)  # remove :;
    if punct_token is not None:
        s = re.sub(r"[.,!?:;]", f" {punct_token}", s)  # all punctuation as one symbol
    else:
        s = re.sub(r"[.,!?:;]", " ", s)
    s = re.sub(r"\s+", " ", s)  # clean whitespaces
    s = re.sub(r"\s$", "", s)
    # s = re.sub("\s\)", "", s)
    return s


def tokenizer_info(tokenizer, dirpath=None):
    tokenizer_info = {
        "name": tokenizer.__class__.__name__,
        "vocab_size": tokenizer.vocab_size,
        "len": len(tokenizer),
        "special_tokens_map": tokenizer.special_tokens_map,
    }
    if dirpath is not None:
        write_json(tokenizer_info, join(dirpath, "tokenizer_info"))
    return tokenizer_info


def tokenize_string(word, tokenizer, prefix=" ", suffix=""):
    """
    when a word gets tokenized it uses special tokens to represent spaces.
    The tokenizer/model knows if its the starting of a new word if there is a "space prefix"

    'yesterday' -> ['yes', 'terday']        # only the word with no whitespace
    ' yesterday' -> ['Ġyesterday']          # with space prefix

    if a space is not included as a prefix then the model have a higher likelihood that it will shop down the word
    into subwords.  That is it represents the common word 'yesterday'  (given a space prefix) into a single token
    (which it will have if it's common enough) or it thinks that 'yesterday' is actually following another word as
    in "yesyesterday" and produces subtokens:

    "yesyesterday" -> ['Ġyes', 'yes', 'terday']

    Therefore by default we should always include a space prefix to get the most relevant token
    """
    assert isinstance(
        word, str
    ), "this function only operates on strings. Single word or seqeuence"
    return tokenizer.encode(prefix + word + suffix)


def tokenize_word_level_dialog(word_level_dialog, tokenizer, remove_punctuation=True):
    """
    Tokenizer dialogs that contain word-level information. Each word is tokenized and then combined to full length list
    with relevant information
    """
    word_level_dialog.sort(key=lambda x: x["start"])
    sp1_idx = tokenizer.convert_tokens_to_ids("<speaker1>")
    sp2_idx = tokenizer.convert_tokens_to_ids("<speaker2>")

    word_ids = []
    input_ids = []
    speaker_ids = []
    starts = []
    ends = []
    inp, sp, st, en = [], [], [], []  # tmp values
    for i, dw in enumerate(word_level_dialog):
        cur_speaker = sp1_idx if dw["speaker_id"] == 0 else sp2_idx

        word = dw["word"]
        if remove_punctuation:
            word = remove_punctuation_capitalization(word)
        toks = tokenize_string(word, tokenizer)
        input_ids += toks

        # a single word may result in several tokens
        cur_speaker = [cur_speaker] * len(toks)
        start = [dw["start"]] * len(toks)
        end = [dw["end"]] * len(toks)
        wi = [i] * len(toks)

        speaker_ids += cur_speaker
        starts += start
        ends += end
        word_ids += wi
    assert (
        len(input_ids) == len(speaker_ids) == len(starts) == len(ends) == len(word_ids)
    )
    return input_ids, speaker_ids, word_ids, starts, ends


def tokenize_turn_level_dialog(turn_level_dialog, tokenizer, remove_punctuation=True):
    """
    tokenize the turn level dialogs

    Same as tokenize_word_level_dialog but now the dialog is on a turn-wise representation so we iterate over turns and
    then over each word in the turn to extract the same information (that is available) as for the tokenize_word_level.

    Tokenizer dialogs that contain word-level information. Each word is tokenized and then combined to full length list
    with relevant information
    """

    # this sorts and make sures that all two consecutive turns belong to different speakers
    turn_level_dialog = join_consecutive_utterances(turn_level_dialog)

    sp1_idx = tokenizer.convert_tokens_to_ids("<speaker1>")
    sp2_idx = tokenizer.convert_tokens_to_ids("<speaker2>")

    input_ids = []
    speaker_ids = []
    starts = []
    ends = []
    word_ids = []
    tmp_word_index = 0
    for i, turn in enumerate(turn_level_dialog):
        cur_sp_idx = sp1_idx if turn["speaker_id"] == 0 else sp2_idx

        text = turn["text"]
        if remove_punctuation:
            text = remove_punctuation_capitalization(text)

        for jj, word_string in enumerate(text.split()):
            t = tokenize_string(word_string, tokenizer)

            # append data
            input_ids += t
            speaker_ids += [cur_sp_idx] * len(t)
            word_ids += [tmp_word_index] * len(t)
            if "starts" in turn:
                starts += [turn["starts"][jj]] * len(t)
            if "ends" in turn:
                ends += [turn["ends"][jj]] * len(t)

            tmp_word_index += 1  # increase for each word

    assert len(input_ids) == len(speaker_ids) == len(word_ids)

    if len(starts) > 0:
        assert len(starts) == len(input_ids)

    if len(ends) > 0:
        assert len(ends) == len(input_ids)

    return input_ids, speaker_ids, word_ids, starts, ends


def add_explicit_turn_shift_token(x, EOT_token_id=None):
    input_ids = x["input_ids"]
    speaker_ids = x["speaker_ids"]
    word_ids = x.get("word_ids")
    starts = x.get("starts")  # only available for word_level dialogs
    ends = x.get("ends")  # only available for word_level dialogs

    # Sanity check
    if starts is None:
        assert len(input_ids) == len(speaker_ids)
    else:
        assert (
            len(input_ids)
            == len(speaker_ids)
            == len(starts)
            == len(ends)
            == len(word_ids)
        )

    expl_input_ids = []
    expl_speaker_ids = []
    expl_word_ids = []
    expl_starts = []
    expl_ends = []
    expl_word_ids = []
    speaker_starts, dur, vval = find_island_idx_len(torch.tensor(speaker_ids))
    for s, d in zip(speaker_starts, dur):
        e = s + d

        # Add single EOT token at turn-shifts or use the next speaker token
        current_speaker = speaker_ids[s]
        if EOT_token_id is not None:
            expl_input_ids += [EOT_token_id] + input_ids[s:e]
            expl_speaker_ids += [current_speaker] + speaker_ids[s:e]
        else:
            expl_input_ids += [current_speaker] + input_ids[s:e]
            expl_speaker_ids += [current_speaker] + speaker_ids[s:e]

        expl_word_ids += [word_ids[s]] + word_ids[s:e]
        if starts is not None:
            expl_starts += [starts[s]] + starts[s:e]
        if ends is not None:
            expl_ends += [ends[s]] + ends[s:e]

    # sanity checks
    assert len(expl_input_ids) == len(expl_speaker_ids) == len(expl_word_ids)
    ret = {
        "input_ids": expl_input_ids,
        "speaker_ids": expl_speaker_ids,
        "word_ids": expl_word_ids,
    }
    if starts is not None:
        assert len(expl_starts) == len(expl_input_ids)
        ret["starts"] = expl_starts

    if ends is not None:
        assert len(expl_ends) == len(expl_input_ids)
        ret["ends"] = expl_ends

    return ret


def chunk_tokenized_dialog(tokenized_dialog, chunk_size, overlap, keep_length):
    return_dialogs = []
    N = len(tokenized_dialog["input_ids"])
    if N >= chunk_size:
        step = chunk_size - overlap

        chunked_inp_ids = torch.tensor(tokenized_dialog["input_ids"]).unfold(
            0, size=chunk_size, step=step
        )
        chunked_sp_ids = torch.tensor(tokenized_dialog["speaker_ids"]).unfold(
            0, size=chunk_size, step=step
        )
        chunked_word_ids = torch.tensor(tokenized_dialog["word_ids"]).unfold(
            0, size=chunk_size, step=step
        )

        # Only word level data got access to word starts/ends
        add_starts = False
        if tokenized_dialog.get("starts") is not None:
            chunked_starts = torch.tensor(tokenized_dialog["starts"]).unfold(
                0, size=chunk_size, step=step
            )
            add_starts = True

        add_ends = False
        if tokenized_dialog.get("ends") is not None:
            chunked_ends = torch.tensor(tokenized_dialog["ends"]).unfold(
                0, size=chunk_size, step=step
            )
            add_ends = True

        # name = basename(filepath).replace(".json", "")
        for i, (input_ids, speaker_ids, word_ids) in enumerate(
            zip(chunked_inp_ids, chunked_sp_ids, chunked_word_ids)
        ):
            tmp_dialog = {
                "input_ids": input_ids.tolist(),
                "speaker_ids": speaker_ids.tolist(),
                "word_ids": word_ids.tolist(),
            }
            if add_starts:
                tmp_dialog["starts"] = chunked_starts[i].tolist()
            if add_ends:
                tmp_dialog["ends"] = chunked_ends[i].tolist()
            return_dialogs.append(tmp_dialog)

        # find out how many tokens we actually added
        total_tokens_added = step * (len(return_dialogs) - 1) + chunk_size
        # print(N - total_tokens_added, "missing!")
        if N - total_tokens_added > keep_length:
            # If there is a substantial amount of tokens missing after the `unfold` above
            # we add the last `chunk size` to include these values
            tmp_dialog = {
                "input_ids": tokenized_dialog["input_ids"][-chunk_size:],
                "speaker_ids": tokenized_dialog["speaker_ids"][-chunk_size:],
                "word_ids": tokenized_dialog["word_ids"][-chunk_size:],
            }
            if add_starts:
                tmp_dialog["starts"] = tokenized_dialog["starts"][-chunk_size:]
            if add_ends:
                tmp_dialog["ends"] = tokenized_dialog["ends"][-chunk_size:]
            return_dialogs.append(tmp_dialog)
    else:
        return_dialogs.append(tokenized_dialog)  # return dialog as is
    return return_dialogs


def format_special_chars(tokens):
    """https://github.com/jessevig/bertviz"""
    if isinstance(tokens, list):
        return [t.replace("Ġ", " ").replace("</w>", "").strip() for t in tokens]
    else:
        return tokens.replace("Ġ", " ").replace("</w>", "").strip()


def convert_ids_to_tokens(idx, tokenizer, clean_whitespace_char=True):
    tokens = []
    if idx.ndim == 1:
        t = tokenizer.convert_ids_to_tokens(idx.tolist())
        if clean_whitespace_char:
            t = format_special_chars(t)
        t = np.array(t, dtype="<U32")
        return t
    elif idx.ndim == 2:
        for batch_list in idx.tolist():
            t = tokenizer.convert_ids_to_tokens(batch_list)
            if clean_whitespace_char:
                t = format_special_chars(t)
            t = np.array(t, dtype="<U32")
            tokens.append(t)
        return np.stack(tokens)
    elif idx.ndim > 2:
        # B, N, k
        for tmp_idx in idx:
            t = convert_ids_to_tokens(tmp_idx, tokenizer)
            tokens.append(t)
        return np.stack(tokens)


def test_remove_punctuation():
    test = [
        'hello there :) my name is foo (HEHE), they call me "bar".',
        "and i live in Baz. well I, you know, live there.",
        "oh :/ :/ :/ :) :D :p :P that`s a shame... well [you know] that's (yeah) I's ",
    ]
    print()
    for t in test:
        p = remove_punctuation_capitalization(t)
        print(t)
        print("-" * 30)
        print(p)
        print("=" * 60)


if __name__ == "__main__":

    from research.tokenizer import load_turngpt_tokenizer, get_special_tokens_dict

    special_tokens_dict = get_special_tokens_dict("turngpt_tokens")
    tokenizer = load_turngpt_tokenizer(
        pretrained="gpt2",
        special_token_dict=special_tokens_dict,
    )
    string = "hello i would like to buy some fruit today"
    input_ids = torch.tensor(tokenize_string(string, tokenizer)).unsqueeze(0)
    idx = torch.cat([input_ids] * 4)
    # idx = torch.stack([idx] * 2)
    # idx = torch.stack([idx] * 3)
    print(idx.shape)
    tokens = convert_ids_to_tokens(idx, tokenizer)
    print(tokens.shape)
    print(tokens)

    # tokens = np.array(idx.tolist(), dtype='<U32')
    # print(tokens.shape)

    test_remove_punctuation()
