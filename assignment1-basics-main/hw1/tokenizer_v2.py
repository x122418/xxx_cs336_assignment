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
        max_token_id = max(vocab.keys())
        # 处理特殊情况 存在 specical token不在vocab中
        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes not in self.vocab.values():
                self.vocab[max_token_id+1] = token_bytes
                max_token_id += 1
                
        # 构建反向映射：bytes → token_id（用于 encode）
        self.anti_vocab = {v: k for k, v in vocab.items()}
        self.merges_rank = {merge:num for num, merge in enumerate(merges)}


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