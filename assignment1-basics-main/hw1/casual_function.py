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

vocab = {num:bytes([num]) for num in range(10)}


print(vocab)
import regex as re

# special_tokens = []
special_tokens = ['<|endoftext|>']
if special_tokens == []: 
    # 单独处理
    pass

special_token_pattern = '|'.join(re.escape(tok) for tok in special_tokens)


document = "<|endoftext|>BBB<|endoftext|>"
after = re.split(special_token_pattern, document)
print(after)
