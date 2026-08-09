import os
import regex as re
from collections import defaultdict

# pre-token pattern  来自gpt-2论文原文
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 分块读取文档辅助函数
def read_file_in_chunks(input_path, chunk_size=1024 * 1024):
    with open(input_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


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

    # 处理pre_token_str计数字典 {str1:5, str2:3 ...} 统计pair频率
    # 重构一个新的byte_tuple 计数字典  原始的pre_token_str字典后面没什么用了
    byte_tuple_cnt = defaultdict(int)
    for pre_token_str, cnt in pre_token_count.items():

        pre_token_bytes = pre_token_str.encode("utf-8")

        tmp_len = len(pre_token_bytes)
        byte_tuple = tuple(pre_token_bytes[i : i + 1] for i in range(tmp_len))
        byte_tuple_cnt[byte_tuple] += cnt

        for i in range(tmp_len - 1):
            pair = (byte_tuple[i], byte_tuple[i + 1])
            pair_count[pair] += cnt

    vocab_cur_size = len(vocab)

    # 这里有个边界条件是需要还有东西可以合并，不过，这个边界正式训练一般不会遇到
    while vocab_cur_size < vocab_size and pair_count:
        best_pair = max(pair_count, key=lambda x: (pair_count[x], x))
        best_pair_combine = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        # 取出best_pair 需要更新 byte_tuple_cnt 和 （重新计数）pair_count
        new_byte_tuple_cnt = defaultdict(int)
        new_pair_count = defaultdict(int)
        for byte_tuple, cnt in byte_tuple_cnt.items():
            i = 0
            new_byte_tuple = []
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
            new_byte_tuple_cnt[new_byte_tuple] += cnt
        byte_tuple_cnt = new_byte_tuple_cnt

        for byte_tuple, cnt in byte_tuple_cnt.items():
            tmp_len = len(byte_tuple)
            for i in range(tmp_len - 1):
                pair = (byte_tuple[i], byte_tuple[i + 1])
                new_pair_count[pair] += cnt
            pair_count = new_pair_count
        vocab[vocab_cur_size] = best_pair_combine
        vocab_cur_size += 1

    return vocab, merges

if __name__ == "main":
    # 只在直接运行 train_bpe.py 时执行的临时测试
    vocab, merges = train_bpe(
        input_path="/home/huangjiaqi/xxx_cs336_assignment/assignment1-basics-main/data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=257,
        special_tokens=[],
    )

    print(len(vocab))
    print(merges)
    print(vocab[256])