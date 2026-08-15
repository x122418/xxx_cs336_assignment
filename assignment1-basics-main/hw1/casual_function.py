# from collections import defaultdict

# a = '说的话'
# b = a.encode(encoding="utf-8")
# print(b)

# for i in b:
#     print(i)
# def get_sep_bytes(A):
#     l = len(A)
#     tuple_A = tuple(A[i:i+1] for i in range(l))
#     return tuple_A
# print(get_sep_bytes(b))
# print(type(b))
# print(type(get_sep_bytes(b)[1]))


# byte_tuple = tuple(b)
# print(byte_tuple)  #(97, 115, 100, 106, 110, 110)

# dic = defaultdict(int)
# for i in range(len(byte_tuple)-1):
#     pair = (byte_tuple[i],byte_tuple[i+1])
#     dic[pair] += 1

# vocab = {num:bytes([num]) for num in range(10)}


# print(vocab)
# import regex as re

# # special_tokens = []
# special_tokens = ['<|endoftext|>']
# if special_tokens == []:
#     # 单独处理
#     pass

# special_token_pattern = '|'.join(re.escape(tok) for tok in special_tokens)


# document = "<|endoftext|>BBB<|endoftext|>"
# after = re.split(special_token_pattern, document)
# print(after)

# import regex as re
# PAT = 'AB'
# s =  'mmmmmAB nnn AB'
# ans1 = re.findall(PAT, s)
# print(ans1)
# for item in re.finditer(PAT, s):
#     print(item.group())


# 旧状态：(a, a, a) → 2
# best_pair：(a, a)
# 正确的新状态：(aa, a) → 2

# x = (a, a, a)
# d = {x: 2}
# b_p = (a, a)


# def apply_bpe_to_pre_token(pre_token_str: str, merges, vocab, merges_rank, anti_vocab) -> list[int]:
#     pre_token_bytes = pre_token_str.encode("utf-8")
#     pre_token_bytes_list = [
#         pre_token_bytes[i : i + 1] for i in range(len(pre_token_bytes))
#     ]
#     cur_token_bytes_list = pre_token_bytes_list

#     while len(cur_token_bytes_list) >= 2:
#         new_token_bytes_list = []
#         # 训练阶段的merge选取看频率 编码阶段取决于merge产生的顺序
#         candidate_pair = []
#         for i in range(len(cur_token_bytes_list) - 1):
#             cur_pair = (cur_token_bytes_list[i], cur_token_bytes_list[i + 1])
#             if cur_pair in merges:
#                 candidate_pair.append(cur_pair)
#         if candidate_pair == []:
#             break
#         else:
#             chosen_pair = min(candidate_pair, key=lambda x: merges_rank[x])
#             i = 0
#             while i <= len(cur_token_bytes_list) - 1:
#                 if i <= len(cur_token_bytes_list) - 2:
#                     if (
#                         chosen_pair[0] == cur_token_bytes_list[i]
#                         and chosen_pair[1] == cur_token_bytes_list[i + 1]
#                     ):
#                         new_token_bytes_list.append(chosen_pair[0] + chosen_pair[1])
#                         i+=2
#                     else:
#                         new_token_bytes_list.append(cur_token_bytes_list[i])
#                         i+=1
#                 else:
#                     new_token_bytes_list.append(cur_token_bytes_list[i])
#                     i+=1
#         cur_token_bytes_list = new_token_bytes_list
#     this_token_id = [anti_vocab[i] for i in cur_token_bytes_list]

#     return this_token_id


# merges = [
#     (b"a", b"a"),
#     (b"aa", b"aa"),
# ]

# merges_rank = {
#     (b"a", b"a"): 0,
#     (b"aa", b"aa"): 1,
# }

# anti_vocab = {
#     b"a": 97,
#     b"aa": 256,
#     b"aaaa": 257,
# }

# result = apply_bpe_to_pre_token(
#     "aaa",
#     merges,
#     None,
#     merges_rank,
#     anti_vocab,
# )

# print(result)
# assert result == [257]

# print(48*(10240000+20582400+3200))

# print(80411200 + 1479628800 +1600 +80411200)

# print(2 * 1024 * 1600)

# print(2 * 1600 * 1600)

# print(2 * 1600 * 4288)

# print(2 * 1600 * 50257)
# print(3*1024*400000*3.52/(3600*297.5))

import torch
starts = torch.randint(0, 5, (2, 5))
print(starts)