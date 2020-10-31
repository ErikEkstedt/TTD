from transformers import AutoTokenizer

"""
Using the tokenizer from the transformer (Huggingface) framework
"""

TurnGPT_TOKENS = {
    "pad_token": "<|endoftext|>",
    "additional_special_tokens": [
        "<speaker1>",
        "<speaker2>",
    ],
}


TurnGPT_EOT_TOKENS = {
    "pad_token": "<|endoftext|>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<TS>"],
}


TURNGPT_TOKENS_OLD = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "unk_token": "<|endoftext|>",
    "sep_token": "<PUNCT>",
    "pad_token": "!",
    "additional_special_tokens": ["<speaker1>", "<speaker2>"],
}


def get_special_tokens_dict(dict_name):
    if dict_name.lower() == "turngpt_tokens":
        return TurnGPT_TOKENS
    elif dict_name.lower() == "turngpt_eot_tokens":
        return TurnGPT_EOT_TOKENS
    elif dict_name.lower() == "turngpt_token_dict_old":
        return TURNGPT_TOKENS_OLD
    else:
        raise NotImplementedError(
            '["turngpt_tokens", "turngpt_eot_tokens", "turngpt_token_dict_old"]'
        )


def load_turngpt_tokenizer(pretrained="gpt2", special_token_dict=None):
    tokenizer = load_tokenizer(pretrained)
    if special_token_dict is None:
        special_token_dict = TurnGPT_TOKENS
    num_added_toks = tokenizer.add_special_tokens(special_token_dict)
    print(f"Extended {pretrained} tokenizer with {num_added_toks}")
    for special_token in tokenizer.additional_special_tokens:
        print("\t" + special_token)
    return tokenizer


def load_tokenizer(pretrained="gpt2"):
    return AutoTokenizer.from_pretrained(pretrained)
