from typing import Iterable, Iterator
from .train_bpe import PAT
import regex as re
class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # Construct a tokenizer from a given
        # vocabulary, list of merges, and (optionally) a list of special tokens
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # 构建反向映射：bytes → token_id（用于 encode）
        self.anti_vocab = {v: k for k, v in vocab.items()}
        self.merges_rank = {merge:num for num, merge in enumerate(merges)}

        
        return

    def _apply_bpe_to_pre_token(
        self, pre_token_str: str
    ) -> list[int]:
        merges = self.merges
        merges_rank = self.merges_rank
        anti_vocab = self.anti_vocab

        pre_token_bytes = pre_token_str.encode("utf-8")
        pre_token_bytes_list = [
            pre_token_bytes[i : i + 1] for i in range(len(pre_token_bytes))
        ]
        cur_token_bytes_list = pre_token_bytes_list

        while len(cur_token_bytes_list) >= 2:
            new_token_bytes_list = []
            # 训练阶段的merge选取看频率 编码阶段取决于merge产生的顺序
            candidate_pair = []
            for i in range(len(cur_token_bytes_list) - 1):
                cur_pair = (cur_token_bytes_list[i], cur_token_bytes_list[i + 1])
                if cur_pair in merges:
                    candidate_pair.append(cur_pair)
            if candidate_pair == []:
                break
            else:
                chosen_pair = min(candidate_pair, key=lambda x: merges_rank[x])
                i = 0
                while i <= len(cur_token_bytes_list) - 1:
                    if i <= len(cur_token_bytes_list) - 2:
                        if (
                            chosen_pair[0] == cur_token_bytes_list[i]
                            and chosen_pair[1] == cur_token_bytes_list[i + 1]
                        ):
                            new_token_bytes_list.append(chosen_pair[0] + chosen_pair[1])
                            i+=2
                        else:
                            new_token_bytes_list.append(cur_token_bytes_list[i])
                            i+=1
                    else:
                        new_token_bytes_list.append(cur_token_bytes_list[i])
                        i+=1
            cur_token_bytes_list = new_token_bytes_list
        this_token_id = [anti_vocab[i] for i in cur_token_bytes_list]

        return this_token_id

    # TODO: Optimize BPE training to pass the speed test.
    # TODO: Implement Tokenizer.from_files.

    # def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) Class method
    # that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the
    # same format that your BPE training code output) and (optionally) a list of special tokens.
    # This method should accept the following additional parameters:

    def encode(self, text: str) -> list[int]:
        token_ids = [] # 存储最终的输出id序列
        special_tokens  = self.special_tokens 
        sorted_special_tokens = sorted(special_tokens, key = len, reverse=True)
        # 添加捕获pattern
        special_token_pattern = "|".join(re.escape(tok) for tok in sorted_special_tokens)
        special_token_pattern = "("+ special_token_pattern +")"
        if special_tokens != []:
            text_segment_list = re.split(special_token_pattern, text)
        else:
            text_segment_list = [text]

        for segment in text_segment_list:
            if segment in special_tokens:
                encoded_seg = self.anti_vocab[segment.encode('utf-8')]
                token_ids.append(encoded_seg)
                continue

            for match in re.finditer(PAT, segment):
                # match.group():str 为PAT直接提取出的pre-token string
                cur_pre_token_str = match.group()
                # 转化为bytes格式
                this_token_id = self._apply_bpe_to_pre_token(cur_pre_token_str)
                token_ids += this_token_id

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Given an iterable of
        # strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        # required for memory-efficient tokenization of large files that we cannot directly load into
        # memory.
        for text in iterable:
            token_ids = self.encode(text)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        vocab = self.vocab
        bytes_list = []
        for token_id in ids:
            bytes_list.append(vocab[token_id])
        long_bytes = b''.join(bytes_list)
        text = long_bytes.decode('utf-8', errors="replace")

        return text
