import os
import regex as re
from collections import defaultdict

# pre-token pattern  来自gpt-2论文原文
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # 以基础256个字节（token最小原子集合） 初始化vocab
    vocab = {num: bytes([num]) for num in range(256)}
    pair_count = defaultdict(int)
    pre_token_count = defaultdict(int)
    merges = []

    # 将 special token 加入 vocab
    for idx, special_token in enumerate(special_tokens):
        vocab[idx + 256] = special_token.encode("utf-8")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 将text根据specail token 切分为 segment
    special_token_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if special_tokens != []:
        text_segment_list = re.split(special_token_pattern, text)
    else:
        text_segment_list = [text]


    for segment in text_segment_list:
        for match in re.finditer(PAT, segment):
            # match.group():str 为PAT直接提取出的pre-token string
            cur_pre_token_str = match.group()
            # 合并相同的pre_token_str 统一处理 记在计数字典中 {str1:5, str2:3 ...}
            pre_token_count[cur_pre_token_str] += 1


    # 初始化倒排索引 包含pair的 pre_token_id
    # {pair1:{0,3,5}, pair1:{0,2,5} ...}
    pair_pre_tk_id = defaultdict(set)

    # {0: pre_token1, 1: pre_token2 ... }   这里的值是需要转化为utf8 后动态变化的 （是token序列）
    pre_token_id_dict = {}
    # {0: 40, 1: 15 ... }    pre_token 出现次数
    pre_token_id_cnt = {}
    for idx, k in enumerate(pre_token_count):
        pre_token_bytes = k.encode('utf-8')
        tmp_len = len(pre_token_bytes)
        byte_tuple = tuple(pre_token_bytes[i : i + 1] for i in range(tmp_len))
        pre_token_id_cnt[idx] = pre_token_count[k]
        pre_token_id_dict[idx] = byte_tuple
    
    for idx, byte_tuple in pre_token_id_dict.items():
        cnt = pre_token_id_cnt[idx]

        tmp_len = len(byte_tuple)

        for i in range(tmp_len - 1):
            pair = (byte_tuple[i], byte_tuple[i + 1])
            pair_count[pair] += cnt
            pair_pre_tk_id[pair].add(idx)

    vocab_cur_size = len(vocab)

    # 这里有个边界条件是需要还有东西可以合并，不过，这个边界正式训练一般不会遇到
    while vocab_cur_size < vocab_size and pair_count:
        best_pair = max(pair_count, key=lambda x: (pair_count[x], x))
        best_pair_combine = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        # 要修改受影响pre-token的 对应的字节序列的形式
        affected_pre_token_id = pair_pre_tk_id[best_pair].copy()

        for idx in affected_pre_token_id:
            byte_tuple = pre_token_id_dict[idx]
            # 计算旧的pair统计情况
            tmp_len = len(byte_tuple)
            old_local_pair_count = defaultdict(int)
            for i in range(tmp_len - 1):
                pair = (byte_tuple[i], byte_tuple[i + 1])
                old_local_pair_count[pair] += 1


            new_byte_tuple = []
            i = 0
            while i <= len(byte_tuple) - 1:
                # 如果右侧有元素 且符合best pair 前进2
                if (
                    (i + 1 <= len(byte_tuple) - 1)
                    and (byte_tuple[i] == best_pair[0])
                    and (byte_tuple[i + 1] == best_pair[1])
                ):
                    new_byte_tuple.append(best_pair_combine)
                    i += 2
                else:
                    new_byte_tuple.append(byte_tuple[i])
                    i += 1

            new_byte_tuple = tuple(new_byte_tuple)
            pre_token_id_dict[idx] = new_byte_tuple
            new_local_pair_count = defaultdict(int)
            tmp_len = len(new_byte_tuple)
            for i in range(tmp_len - 1):
                pair = (new_byte_tuple[i], new_byte_tuple[i + 1])
                new_local_pair_count[pair] += 1

            all_pairs = new_local_pair_count.keys()|old_local_pair_count.keys()
            for pair in all_pairs:
                old_cnt = old_local_pair_count.get(pair, 0)
                new_cnt = new_local_pair_count.get(pair, 0)
                delta = new_cnt - old_cnt
                pair_count[pair] += delta*pre_token_id_cnt[idx]

            for pair in new_local_pair_count.keys() - old_local_pair_count.keys():
                pair_pre_tk_id[pair].add(idx)
            for pair in old_local_pair_count.keys() -new_local_pair_count.keys():
                pair_pre_tk_id[pair].remove(idx)

        # 删除piar count为0 实际已经不存在的piar的痕迹
        zero_pairs = []
        for k, v in pair_count.items():
            if v==0:
                zero_pairs.append(k)
        for pair in zero_pairs:
            del pair_pre_tk_id[pair]
            del pair_count[pair]


        vocab[vocab_cur_size] = best_pair_combine
        vocab_cur_size += 1

    return vocab, merges