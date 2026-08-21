# CS336 学习与作业进度

Aaron 的 Stanford CS336 自学与作业同步仓库，用于在本地、GitHub 和服务器之间同步代码并记录学习进度。

## 总体进度

| 作业 | 主题 | 当前状态 |
| --- | --- | --- |
| Assignment 1 | Basics | 核心流程已完成 |
| Assignment 2 | Systems | 已加入仓库，准备开始 |

## Assignment 1: Basics

### 已完成

- [x] 实现并优化 byte-level BPE 训练，完成词表与 merges 的序列化
- [x] 实现 Tokenizer 的 encode、decode、encode_iterable 与 from_files
- [x] 实现 Linear、Embedding、RMSNorm、RoPE、SwiGLU 和多头自注意力
- [x] 搭建 Transformer Block 与 Transformer Language Model
- [x] 实现 cross-entropy、AdamW、学习率调度与梯度裁剪
- [x] 实现数据采样、checkpoint 保存与恢复以及完整训练循环
- [x] 完成 TinyStories 的分词、训练、TensorBoard 监控与文本生成
- [x] 进行学习率、RMSNorm 和位置编码等小规模实验

### 后续优化

- [ ] 继续优化 BPE 预分词、分块读取与并行处理性能
- [ ] 还没有仔细研读weiruirui笔记5.1关于官方Dataset数据原语部分的内容
- [ ] 在需要时补充 KV cache 和更高效的推理实现

## Assignment 2: Systems

### 当前状态

- [x] 将 Assignment 2 作业目录加入同步仓库
- [ ] 阅读作业说明并完成环境检查
- [ ] 梳理各章节任务、测试入口与实现顺序
- [ ] 开始系统与性能优化相关实现


