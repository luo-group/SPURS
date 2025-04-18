# https://github.com/BytedProtein/ByProt/blob/dd279dc85f76ee2c28c819b71bf3911b90159f0a/src/byprot/datamodules/datasets/transforms.py
import json
from copy import deepcopy
from functools import lru_cache
from typing import Any, Callable, List, Optional, Union

import torch
import torchtext  # noqa: F401
from torch import Tensor
from torch.nn import Module
from torchtext import functional as F
from torchtext.data.functional import load_sp_model
from torchtext.utils import get_asset_local_path
from torchtext.vocab import Vocab

__all__ = [
    "SentencePieceTokenizer",
    "PlainTokenizer"
    "VocabTransform",
    "ToTensor",
    "LabelToIndex",
    "Truncate",
    "AddToken",
    "PadTransform",
    "StrToIntTransform",
    "GPT2BPETokenizer",
    "Sequential",
]


class PlainTokenizer(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sentence or list of sentences on which to apply tokenizer.
        :type input: Union[str, List[str]]
        :return: tokenized text
        :rtype: Union[List[str], List[List(str)]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: List[List[str]] = []
            for text in input:
                tokens.append(text.strip().split(' '))
            return tokens
        elif torch.jit.isinstance(input, str):
            return input.strip().split(' ')
        else:
            print(input)
            raise TypeError("Input type not supported")


class SentencePieceTokenizer(Module):
    """
    Transform for Sentence Piece tokenizer from pre-trained sentencepiece model

    Additiona details: https://github.com/google/sentencepiece

    :param sp_model_path: Path to pre-trained sentencepiece model
    :type sp_model_path: str

    Example
        >>> from torchtext.transforms import SentencePieceTokenizer
        >>> transform = SentencePieceTokenizer("spm_model")
        >>> transform(["hello world", "attention is all you need!"])
    """

    def __init__(self, sp_model_path: str):
        super().__init__()
        self.sp_model = load_sp_model(get_asset_local_path(sp_model_path))

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sentence or list of sentences on which to apply tokenizer.
        :type input: Union[str, List[str]]
        :return: tokenized text
        :rtype: Union[List[str], List[List(str)]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: List[List[str]] = []
            for text in input:
                tokens.append(self.sp_model.EncodeAsPieces(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            return self.sp_model.EncodeAsPieces(input)
        else:
            raise TypeError("Input type not supported")


class VocabTransform(Module):
    r"""Vocab transform to convert input batch of tokens into corresponding token ids

    :param vocab: an instance of :class:`torchtext.vocab.Vocab` class.

    Example:
        >>> import torch
        >>> from torchtext.vocab import vocab
        >>> from torchtext.transforms import VocabTransform
        >>> from collections import OrderedDict
        >>> vocab_obj = vocab(OrderedDict([('a', 1), ('b', 1), ('c', 1)]))
        >>> vocab_transform = VocabTransform(vocab_obj)
        >>> output = vocab_transform([['a','b'],['a','b','c']])
        >>> jit_vocab_transform = torch.jit.script(vocab_transform)
    """

    def __init__(self, vocab: Vocab):
        super().__init__()
        assert isinstance(vocab, Vocab)
        self.vocab = vocab

    def forward(self, input: Any) -> Any:
        """
        :param input: Input batch of token to convert to correspnding token ids
        :type input: Union[List[str], List[List[str]]]
        :return: Converted input into corresponding token ids
        :rtype: Union[List[int], List[List[int]]]
        """

        if torch.jit.isinstance(input, List[str]):
            return self.vocab.lookup_indices(input)
        elif torch.jit.isinstance(input, List[List[str]]):
            output: List[List[int]] = []
            for tokens in input:
                output.append(self.vocab.lookup_indices(tokens))

            return output
        else:
            raise TypeError("Input type not supported")


class ToTensor(Module):
    r"""Convert input to torch tensor

    :param padding_value: Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    :type padding_value: Optional[int]
    :param dtype: :class:`torch.dtype` of output tensor
    :type dtype: :class:`torch.dtype`
    """

    def __init__(self, padding_value: Optional[int] = None, dtype: torch.dtype = torch.long) -> None:
        super().__init__()
        self.padding_value = padding_value
        self.dtype = dtype

    def forward(self, input: Any) -> Tensor:
        """
        :param input: Sequence or batch of token ids
        :type input: Union[List[int], List[List[int]]]
        :rtype: Tensor
        """
        return F.to_tensor(input, padding_value=self.padding_value, dtype=self.dtype)


class LabelToIndex(Module):
    r"""
    Transform labels from string names to ids.

    :param label_names: a list of unique label names
    :type label_names: Optional[List[str]]
    :param label_path: a path to file containing unique label names containing 1 label per line. Note that either label_names or label_path should be supplied
                       but not both.
    :type label_path: Optional[str]
    """

    def __init__(
        self,
        label_names: Optional[List[str]] = None,
        label_path: Optional[str] = None,
        sort_names=False,
    ):
        assert label_names or label_path, "label_names or label_path is required"
        assert not (label_names and label_path), "label_names and label_path are mutually exclusive"
        super().__init__()

        if label_path:
            with open(label_path, "r") as f:
                label_names = [line.strip() for line in f if line.strip()]
        else:
            label_names = label_names

        if sort_names:
            label_names = sorted(label_names)
        self._label_vocab = Vocab(torch.classes.torchtext.Vocab(label_names, None))
        self._label_names = self._label_vocab.get_itos()

    def forward(self, input: Any) -> Any:
        """
        :param input: Input labels to convert to corresponding ids
        :type input: Union[str, List[str]]
        :rtype: Union[int, List[int]]
        """
        if torch.jit.isinstance(input, List[str]):
            return self._label_vocab.lookup_indices(input)
        elif torch.jit.isinstance(input, str):
            return self._label_vocab.__getitem__(input)
        else:
            raise TypeError("Input type not supported")

    @property
    def label_names(self) -> List[str]:
        return self._label_names


class Truncate(Module):
    r"""Truncate input sequence

    :param max_seq_len: The maximum allowable length for input sequence
    :type max_seq_len: int
    """

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch of sequence to be truncated
        :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        :return: Truncated sequence
        :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """
        return F.truncate(input, self.max_seq_len)


class AddToken(Module):
    """Add token to beginning or end of sequence

    :param token: The token to be added
    :type token: Union[int, str]
    :param begin: Whether to insert token at start or end or sequence, defaults to True
    :type begin: bool, optional
    """

    def __init__(self, token: Union[int, str], begin: bool = True) -> None:
        super().__init__()
        self.token = token
        self.begin = begin

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch
        :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """
        return F.add_token(input, self.token, self.begin)


class PadTransform(Module):
    """Pad tensor to a fixed length with given padding value.

    :param max_length: Maximum length to pad to
    :type max_length: int
    :param pad_value: Value to pad the tensor with
    :type pad_value: bool
    """

    def __init__(self, max_length: int, pad_value: int):
        super().__init__()
        self.max_length = max_length
        self.pad_value = float(pad_value)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: The tensor to pad
        :type x: Tensor
        :return: Tensor padded up to max_length with pad_value
        :rtype: Tensor
        """
        max_encoded_length = x.size(-1)
        if max_encoded_length < self.max_length:
            pad_amount = self.max_length - max_encoded_length
            x = torch.nn.functional.pad(x, (0, pad_amount), value=self.pad_value)
        return x


class StrToIntTransform(Module):
    """Convert string tokens to integers (either single sequence or batch)."""

    def __init__(self):
        super().__init__()

    def forward(self, input: Any) -> Any:
        """
        :param input: sequence or batch of string tokens to convert
        :type input: Union[List[str], List[List[str]]]
        :return: sequence or batch converted into corresponding token ids
        :rtype: Union[List[int], List[List[int]]]
        """
        return F.str_to_int(input)


class GPT2BPETokenizer(Module):
    """
    Transform for GPT-2 BPE Tokenizer.

    Reimplements openai GPT-2 BPE in TorchScript. Original openai implementation
    https://github.com/openai/gpt-2/blob/master/src/encoder.py

    :param encoder_json_path: Path to GPT-2 BPE encoder json file.
    :type encoder_json_path: str
    :param vocab_bpe_path: Path to bpe vocab file.
    :type vocab_bpe_path: str
    :param return_tokens: Indicate whether to return split tokens. If False, it will return encoded token IDs as strings (default: False)
    :type return_input: bool
    """

    __jit_unused_properties__ = ["is_jitable"]
    _seperator: torch.jit.Final[str]

    def __init__(self, encoder_json_path: str, vocab_bpe_path: str, return_tokens: bool = False):
        super().__init__()
        self._seperator = "\u0001"
        # load bpe encoder and bpe decoder
        with open(get_asset_local_path(encoder_json_path), "r", encoding="utf-8") as f:
            bpe_encoder = json.load(f)
        # load bpe vocab
        with open(get_asset_local_path(vocab_bpe_path), "r", encoding="utf-8") as f:
            bpe_vocab = f.read()
        bpe_merge_ranks = {
            self._seperator.join(merge_pair.split()): i for i, merge_pair in enumerate(bpe_vocab.split("\n")[1:-1])
        }
        # Caching is enabled in Eager mode
        self.bpe = GPT2BPEEncoderPyBind(bpe_encoder, bpe_merge_ranks, self._seperator, bytes_to_unicode(), True)

        self._return_tokens = return_tokens

    @property
    def is_jitable(self):
        return isinstance(self.bpe, torch._C.ScriptObject)

    @torch.jit.export
    def _encode(self, text: str) -> List[str]:
        """Encode text into a list of tokens IDs

        Args:
            text: An input text string.

        Returns:
            A list of bpe token ids represents each bpe tokens

        For example: "awesome,awe"
            --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", e]
            --> bpe encode --> bpe token ids: [707, 5927, 11, 707, 68]
        """
        bpe_token_ids: List[int] = self.bpe.encode(text)
        bpe_tokens: List[str] = []

        for bpe_token_id in bpe_token_ids:
            bpe_tokens.append(str(bpe_token_id))

        return bpe_tokens

    @torch.jit.export
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of tokens

        Args:
            text: An input text string.

        Returns:
            A list of bpe token ids represents each bpe tokens

        For example: "awesome,awe"
            --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", e]
        """
        return self.bpe.tokenize(text)

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sentence or list of sentences on which to apply tokenizer.
        :type input: Union[str, List[str]]
        :return: tokenized text
        :rtype: Union[List[str], List[List(str)]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: List[List[str]] = []
            for text in input:
                if self._return_tokens:
                    tokens.append(self._tokenize(text))
                else:
                    tokens.append(self._encode(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            if self._return_tokens:
                return self._tokenize(input)
            else:
                return self._encode(input)
        else:
            raise TypeError("Input type not supported")

    def __prepare_scriptable__(self):
        r"""Return a JITable tokenizer."""
        if not self.is_jitable:
            tokenizer_copy = deepcopy(self)
            # Disable caching in script mode
            tokenizer_copy.bpe = torch.classes.torchtext.GPT2BPEEncoder(
                self.bpe.bpe_encoder_, self.bpe.bpe_merge_ranks_, self.bpe.seperator_, self.bpe.byte_encoder_, False
            )
            return tokenizer_copy
        return self


class CLIPTokenizer(Module):
    """
    Transform for CLIP Tokenizer. Based on Byte-Level BPE.

    Reimplements CLIP Tokenizer in TorchScript. Original implementation:
    https://github.com/mlfoundations/open_clip/blob/main/src/clip/tokenizer.py

    This tokenizer has been trained to treat spaces like parts of the tokens
    (a bit like sentencepiece) so a word will be encoded differently whether it
    is at the beginning of the sentence (without space) or not.

    The below code snippet shows how to use the CLIP tokenizer with encoder and merges file
    taken from the original paper implementation.

    Example
        >>> from torchtext.transforms import CLIPTokenizer
        >>> MERGES_FILE = "http://download.pytorch.org/models/text/clip_merges.bpe"
        >>> ENCODER_FILE = "http://download.pytorch.org/models/text/clip_encoder.json"
        >>> tokenizer = CLIPTokenizer(merges_path=MERGES_FILE, encoder_json_path=ENCODER_FILE)
        >>> tokenizer("the quick brown fox jumped over the lazy dog")

    :param merges_path: Path to bpe merges file.
    :type merges_path: str
    :param encoder_json_path: Optional, path to BPE encoder json file. When specified, this is used
        to infer num_merges.
    :type encoder_json_path: str
    :param num_merges: Optional, number of merges to read from the bpe merges file.
    :type num_merges: int
    :param return_tokens: Indicate whether to return split tokens. If False, it will return encoded token IDs as strings (default: False)
    :type return_input: bool
    """

    __jit_unused_properties__ = ["is_jitable"]
    _seperator: torch.jit.Final[str]

    def __init__(
        self,
        merges_path: str,
        encoder_json_path: Optional[str] = None,
        num_merges: Optional[int] = None,
        return_tokens: bool = False,
    ):
        super().__init__()
        self._seperator = "\u0001"
        # load bpe merges
        with open(get_asset_local_path(merges_path), "r", encoding="utf-8") as f:
            bpe_merges = f.read().split("\n")[1:]

        if encoder_json_path:
            # load bpe encoder
            with open(get_asset_local_path(encoder_json_path), "r", encoding="utf-8") as f:
                bpe_encoder = json.load(f)
            # 256 * 2 for each byte. For each byte we have ['a', 'a</w>']
            # Additional 2 tokens for bos and eos
            num_merges = len(bpe_encoder) - (256 * 2 + 2)
            bpe_merge_ranks = {
                self._seperator.join(merge_pair.split()): i for i, merge_pair in enumerate(bpe_merges[:num_merges])
            }
        else:
            num_merges = num_merges or len(bpe_merges)
            bpe_merge_ranks = {
                self._seperator.join(merge_pair.split()): i for i, merge_pair in enumerate(bpe_merges[:num_merges])
            }
            bpe_vocab = list(bytes_to_unicode().values())
            bpe_vocab = bpe_vocab + [v + "</w>" for v in bpe_vocab]
            bpe_vocab.extend(["".join(merge_pair.split()) for merge_pair in bpe_merges[:num_merges]])
            bpe_vocab.extend(["<|startoftext|>", "<|endoftext|>"])
            bpe_encoder = {v: i for i, v in enumerate(bpe_vocab)}

        # Caching is enabled in Eager mode
        self.bpe = CLIPEncoderPyBind(bpe_encoder, bpe_merge_ranks, self._seperator, bytes_to_unicode(), True)

        self._return_tokens = return_tokens

    @property
    def is_jitable(self):
        return isinstance(self.bpe, torch._C.ScriptObject)

    @torch.jit.export
    def _encode(self, text: str) -> List[str]:
        """Encode text into a list of tokens IDs

        Args:
            text: An input text string.

        Returns:
            A list of bpe token ids represents each bpe tokens

        For example: "awesome,awe"
            --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", "e"]
            --> bpe encode --> bpe token ids: [707, 5927, 11, 707, 68]
        """
        text = text.lower().strip()
        bpe_token_ids: List[int] = self.bpe.encode(text)
        bpe_tokens: List[str] = []

        for bpe_token_id in bpe_token_ids:
            bpe_tokens.append(str(bpe_token_id))

        return bpe_tokens

    @torch.jit.export
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of tokens

        Args:
            text: An input text string.

        Returns:
            A list of bpe token ids represents each bpe tokens

        For example: "awesome,awe"
            --> bpe --> bpe tokens: ["aw", "esome"], [","], ["aw", "e"]
        """
        text = text.lower().strip()
        return self.bpe.tokenize(text)

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sentence or list of sentences on which to apply tokenizer.
        :type input: Union[str, List[str]]
        :return: tokenized text
        :rtype: Union[List[str], List[List(str)]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: List[List[str]] = []
            for text in input:
                if self._return_tokens:
                    tokens.append(self._tokenize(text))
                else:
                    tokens.append(self._encode(text))
            return tokens
        elif torch.jit.isinstance(input, str):
            if self._return_tokens:
                return self._tokenize(input)
            else:
                return self._encode(input)
        else:
            raise TypeError("Input type not supported")

    def __prepare_scriptable__(self):
        r"""Return a JITable tokenizer."""
        if not self.is_jitable:
            tokenizer_copy = deepcopy(self)
            # Disable caching in script mode
            tokenizer_copy.bpe = torch.classes.torchtext.CLIPEncoder(
                self.bpe.bpe_encoder_, self.bpe.bpe_merge_ranks_, self.bpe.seperator_, self.bpe.byte_encoder_, False
            )
            return tokenizer_copy
        return self


class BERTTokenizer(Module):
    """
    Transform for BERT Tokenizer.

    Based on WordPiece algorithm introduced in paper:
    https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf

    The backend kernel implementation is taken and modified from https://github.com/LieluoboAi/radish.

    See PR https://github.com/pytorch/text/pull/1707 summary for more details.

    The below code snippet shows how to use the BERT tokenizer using the pre-trained vocab files.

    Example
        >>> from torchtext.transforms import BERTTokenizer
        >>> VOCAB_FILE = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
        >>> tokenizer = BERTTokenizer(vocab_path=VOCAB_FILE, do_lower_case=True, return_tokens=True)
        >>> tokenizer("Hello World, How are you!") # single sentence input
        >>> tokenizer(["Hello World","How are you!"]) # batch input

    :param vocab_path: Path to pre-trained vocabulary file. The path can be either local or URL.
    :type vocab_path: str
    :param do_lower_case: Indicate whether to do lower case. (default: True)
    :type do_lower_case: Optional[bool]
    :param strip_accents: Indicate whether to strip accents. (default: None)
    :type strip_accents: Optional[bool]
    :param return_tokens: Indicate whether to return tokens. If false, returns corresponding token IDs as strings (default: False)
    :type return_tokens: bool
    """

    __jit_unused_properties__ = ["is_jitable"]

    def __init__(
        self, vocab_path: str, do_lower_case: bool = True, strip_accents: Optional[bool] = None, return_tokens=False
    ) -> None:
        super().__init__()
        self.bert_model = BERTEncoderPyBind(get_asset_local_path(vocab_path), do_lower_case, strip_accents)
        self._return_tokens = return_tokens
        self._vocab_path = vocab_path
        self._do_lower_case = do_lower_case
        self._strip_accents = strip_accents

    @property
    def is_jitable(self):
        return isinstance(self.bert_model, torch._C.ScriptObject)

    @torch.jit.export
    def _encode(self, text: str) -> List[str]:
        """Encode text into a list of tokens IDs

        Args:
            text: An input text string.

        Returns:
            A list of token ids represents each sub-word

        For example:
            --> "Hello world!" --> token ids: [707, 5927, 11, 707, 68]
        """
        token_ids: List[int] = self.bert_model.encode(text.strip())
        tokens_ids_str: List[str] = [str(token_id) for token_id in token_ids]
        return tokens_ids_str

    @torch.jit.export
    def _batch_encode(self, text: List[str]) -> List[List[str]]:
        """Batch version of _encode i.e operate on list of str"""
        token_ids: List[List[int]] = self.bert_model.batch_encode([t.strip() for t in text])
        tokens_ids_str: List[List[str]] = [[str(t) for t in token_id] for token_id in token_ids]
        return tokens_ids_str

    @torch.jit.export
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of tokens

        Args:
            text: An input text string.

        Returns:
            A list of tokens (sub-words)

        For example:
            --> "Hello World!": ["Hello", "World", "!"]
        """
        return self.bert_model.tokenize(text.strip())

    @torch.jit.export
    def _batch_tokenize(self, text: List[str]) -> List[List[str]]:
        """Batch version of _tokenize i.e operate on list of str"""
        return self.bert_model.batch_tokenize([t.strip() for t in text])

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sentence or list of sentences on which to apply tokenizer.
        :type input: Union[str, List[str]]
        :return: tokenized text
        :rtype: Union[List[str], List[List(str)]]
        """
        if torch.jit.isinstance(input, List[str]):
            tokens: List[List[str]] = []
            if self._return_tokens:
                tokens = self._batch_tokenize(input)
            else:
                tokens = self._batch_encode(input)
            return tokens
        elif torch.jit.isinstance(input, str):
            if self._return_tokens:
                return self._tokenize(input)
            else:
                return self._encode(input)
        else:
            raise TypeError("Input type not supported")

    def __prepare_scriptable__(self):
        if not self.is_jitable:
            tokenizer_copy = deepcopy(self)
            tokenizer_copy.bert_model = torch.classes.torchtext.BERTEncoder(
                self._vocab_path, self._do_lower_case, self._strip_accents
            )
            return tokenizer_copy

        return self


@lru_cache()
def bytes_to_unicode():
    """
    Original Source: https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9

    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class Compose(object):
    r"""A container to host a sequence of text transforms."""

    def __init__(self, *args):
        self.modules = []
        for idx, module in enumerate(args):
            assert isinstance(module, Callable)
            self.modules.append(module)

    def __call__(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch. The input type must be supported by the first transform in the sequence.
        :type input: `Any`
        """
        for module in self.modules:
            input = module(input)
        return input
