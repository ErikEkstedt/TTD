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
        ret["ends"] = expl_starts

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


class DatasetTokenizer(object):
    def prepare_turn_level_tokens(self, tokenizer):
        if not self.check_if_dir_exists(self.tokenized_turn_level_root, ".json"):
            self.prepare_turn_level()

            makedirs(self.tokenized_turn_level_root, exist_ok=True)

            # TOKENIZER SANITY CHECK
            _ = tokenizer_info(
                tokenizer, self.tokenized_turn_level_root
            )  # Save tokenizer info for checks

            t = time.time()
            broken = 0
            broken_files = []
            for turn_level_path in tqdm(
                glob(join(self.turn_level_root, "*.json")),
                desc=f"Tokenizing Turn-level {self.NAME}",
            ):
                turn_level_dialog = read_json(turn_level_path)

                (
                    input_ids,
                    speaker_ids,
                    word_ids,
                    starts,
                    ends,
                ) = tokenize_turn_level_dialog(
                    turn_level_dialog, tokenizer, remove_punctuation=True
                )

                data = {
                    "input_ids": input_ids,
                    "speaker_ids": speaker_ids,
                    "word_ids": word_ids,
                }

                if len(starts) > 0:
                    data["starts"] = starts

                if len(ends) > 0:
                    data["ends"] = ends

                write_json(
                    data,
                    join(self.tokenized_turn_level_root, basename(turn_level_path)),
                )

            t = time.time() - t
            print(f"{self.NAME} tokenization took {round(t, 1)} seconds")
            print(f"{self.NAME} broken", broken)
            write_txt(broken_files, join(self.root, "broken_tokenize.txt"))
        return self.tokenized_turn_level_root

    def prepare_word_level_tokens(self, tokenizer):
        if not self.check_if_dir_exists(self.tokenized_word_level_root):
            self.prepare_word_level()

            makedirs(self.tokenized_word_level_root, exist_ok=True)

            # TOKENIZER SANITY CHECK
            _ = tokenizer_info(
                tokenizer, self.tokenized_word_level_root
            )  # Save tokenizer info for checks

            desc = f"Tokenizing Word-level {self.NAME}"
            t = time.time()
            broken = 0
            broken_files = []
            for word_level_path in tqdm(
                glob(join(self.word_level_root, "*.json")), desc=desc
            ):
                json_name = basename(word_level_path)
                word_level_dialog = read_json(word_level_path)

                (
                    input_ids,
                    speaker_ids,
                    word_ids,
                    starts,
                    ends,
                ) = tokenize_word_level_dialog(
                    word_level_dialog,
                    tokenizer,
                )
                write_json(
                    {
                        "input_ids": input_ids,
                        "speaker_ids": speaker_ids,
                        "starts": starts,
                        "ends": ends,
                        "word_ids": word_ids,
                    },
                    join(self.tokenized_word_level_root, json_name),
                )

            t = time.time() - t
            print(f"{self.NAME} tokenization took {round(t, 1)} seconds")
            print(f"{self.NAME} broken", broken)
            write_txt(broken_files, join(self.root, "broken_tokenize.txt"))

    def prepare_explicit_turn_level_tokens(self, tokenizer, EOT_token_id=None):
        """
        loads all tokenized turn-level dialogs and inserts, either a special EOT_token_id (if not None)
        or the index of the next speaker token, in between the turns.
        """

        tokenized_explicit_turn_path = self.get_tokenized_root(
            level="turn", explicit_turns=True, EOT_token_id=EOT_token_id, chunk_size=-1
        )

        if not self.check_if_dir_exists(tokenized_explicit_turn_path, ".json"):
            self.prepare_turn_level_tokens(tokenizer)  # check the necessary data exists

            makedirs(tokenized_explicit_turn_path, exist_ok=True)

            # Copy tokenizer info
            src = join(self.tokenized_turn_level_root, "tokenizer_info")
            dst = join(tokenized_explicit_turn_path, "tokenizer_info")
            shutil.copy(src, dst)

            tok_files = glob(join(self.tokenized_turn_level_root, "*.json"))
            for tokenized_turn_level_path in tqdm(
                tok_files, desc=f"{self.NAME} Explicit turns"
            ):
                tokenized_turn_level_dialog = read_json(tokenized_turn_level_path)
                explicit_turns = add_explicit_turn_shift_token(
                    tokenized_turn_level_dialog, EOT_token_id
                )
                json_name = basename(tokenized_turn_level_path)
                write_json(
                    explicit_turns, join(tokenized_explicit_turn_path, json_name)
                )
        return tokenized_explicit_turn_path

    def prepare_chunked_tokens(
        self, tokenized_path, chunk_size, overlap, keep_length, sep="_#"
    ):
        assert chunk_size > 0, "chunk size must be larger than 0"
        tokenized_chunk_path = tokenized_path + f"_chunk-{chunk_size}"

        if not self.check_if_dir_exists(tokenized_chunk_path, ".json"):
            print(f"Chunk {self.NAME} -> {chunk_size}")
            makedirs(tokenized_chunk_path, exist_ok=True)

            # Copy tokenizer used
            # tokenizer_info(tokenizer, self.tokenized_turn_level_root)
            src = join(tokenized_path, "tokenizer_info")
            dst = join(tokenized_chunk_path, "tokenizer_info")
            shutil.copy(src, dst)

            tokenized_files = glob(join(tokenized_path, "*.json"))
            for json_path in tqdm(tokenized_files, desc=f"{self.NAME} Chunk"):
                tokenized_dialog = read_json(json_path)
                chunked_dialogs = chunk_tokenized_dialog(
                    tokenized_dialog, chunk_size, overlap, keep_length
                )

                # Save the chunked files
                name = basename(json_path).replace(".json", "")
                for i, chunked_dialog in enumerate(chunked_dialogs):
                    tmp_name = name
                    if i > 0:
                        tmp_name += sep + str(i)
                    write_json(
                        chunked_dialog, join(tokenized_chunk_path, tmp_name + ".json")
                    )

        print("Chunk size: ", chunk_size)
        return tokenized_chunk_path

    def transform_split_filepaths_with_chunks(self, chunked_path, sep="_#"):
        chunked_files = glob(join(chunked_path, "*.json"))

        train_extended = []
        val_extended = []
        test_extended = []
        for f in chunked_files:
            path = split(f)[0]
            filename = basename(f)
            name = filename.replace(".json", "").split(sep)[0]
            orig_name = name + ".json"
            if orig_name in self.train_filepaths:
                train_extended.append(filename)
            if orig_name in self.test_filepaths:
                val_extended.append(filename)
            if orig_name in self.val_filepaths:
                test_extended.append(filename)

        print(self.NAME)
        print(f"Train {len(self.train_filepaths)} -> {len(train_extended)}")
        print(f"val {len(self.val_filepaths)} -> {len(val_extended)}")
        print(f"test {len(self.test_filepaths)} -> {len(test_extended)}")
        print("-" * 50)
        self.train_filepaths = train_extended
        self.val_filepaths = val_extended
        self.test_filepaths = test_extended

    def get_tokenized_filepaths(self, split="train", level="word"):
        filepaths = []
        if level == "word":
            if split == "train":
                for filename in self.train_filepaths:
                    filepaths.append(join(self.tokenized_word_level_root, filename))
            elif split == "val":
                for filename in self.val_filepaths:
                    filepaths.append(join(self.tokenized_word_level_root, filename))
            elif split == "test":
                for filename in self.test_filepaths:
                    filepaths.append(join(self.tokenized_word_level_root, filename))
        elif level == "turn":
            print("Turn level tokenized")

        return filepaths
