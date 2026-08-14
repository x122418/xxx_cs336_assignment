# CS336 学习与作业进度

这是Aaron(xxx)的 Stanford CS336 作业仓库，用于在本地、GitHub 和 服务器之间同步代码与学习进度。

## 当前进度

### Assignment 1: Basics


#### 已完成

- [x] BPE train baseline实现, Tokenizer 类的baseline版本构建
- [x] 学习einops用法, 实现class Linear, RMSnorm等
- [x] 实现了transformer lm 以及其所需的所有类


#### TODO

- [ ] 优化 BPE trainer 性能，通过 speed test（使用倒排索引 避免反复读取整个序列
- [ ] 优化 Tokenizer （类似前一条的优化
- [ ] 分块阅读文件优化，调用官方的文件边界代码
- [ ] 实现 `Tokenizer.from_files`


#### 已知限制

- 当前 BPE trainer 使用每轮重新扫描并统计 pair 的 baseline 实现，正确性测试已通过，但暂未通过速度测试。
- `Tokenizer.from_files` 尚未实现；当前测试通过 adapter 直接传入 vocab 和 merges。
- Tokenizer 的内存测试显示为 `XFAIL`。
