# Cookiee

基于 Hugging Face Trainer 构建的训练框架，面向大规模文本与多模态模型训练，设计目标大致覆盖 **32B 以下模型** 与 **1T token 以内数据** 的训练场景。

## 主要特性

* **统一支持文本与多模态训练**：支持文本/多模态的预训练与 SFT。
* **自定义数据处理流水线**：数据处理流程大致分为 **read → convert → preprocess → mix**。其中 `read` 阶段支持并发读取多个数据集，`convert` 与 `preprocess` 用于将不同来源的数据统一到同一训练格式，`mix` 则用于在指定阶段完成多数据集混合。这套设计更适合大规模训练场景，也更方便检查每个阶段的中间结果。
* **更可控的缓存管理**：相比直接使用 Hugging Face Datasets，更方便管理中间缓存与最终缓存，避免出现难以追踪的隐式缓存，在大规模数据处理时更易复用与排查问题。
* **支持 packing**：提升训练吞吐与数据利用效率。
* **支持多轮对话训练**：支持 chat template、history masking 等能力。
* **支持 domain loss 观测**：可以在训练过程中按 domain 统计和观察 loss 变化，而不只是看一个全局平均 loss。在大规模混合数据训练中，不同 domain 的收敛速度往往并不一致，某些 domain 可能已经训练充分，而另一些 domain 仍然欠训练。通过 domain loss，可以更直观地分析数据配比、采样策略和训练阶段切换是否合理，也更方便定位某些数据域效果异常的来源，为后续的数据混合设计与训练调度提供依据。

## 安装

```bash
git clone https://github.com/wwwbq/Cookiee.git
cd Cookiee
pip install -e .
```

安装完成后，可使用命令行入口 `cookiee`。

## 程序入口

### `cookiee train`

使用指定的 Python 训练脚本启动训练。

```bash
cookiee train examples/vlm/llava/pretrain.py --config_path examples/vlm/llava/config/pretrain.yaml
```

该命令会统一处理分布式训练启动逻辑，并可结合以下环境变量进行多机/多卡训练：

* `NNODES`
* `NODE_RANK`
* `MASTER_ADDR`
* `MASTER_PORT`

## 示例

当前仓库中的示例主要分为 **LLM** 与 **VLM** 两类。

### LLM 示例

位于：

```text
examples/llm/
```

包括：

* `pretrain.py`：文本预训练示例
* `midtrain.py`：文本继续训练 / 中间阶段训练示例
* `sft.py`：文本监督微调示例
* `configs/`：对应配置文件

示例命令：

```bash
cookiee train examples/llm/pretrain.py --config_path examples/llm/configs/pretrain.yaml
cookiee train examples/llm/midtrain.py --config_path examples/llm/configs/midtrain.yaml
cookiee train examples/llm/sft.py --config_path examples/llm/configs/sft.yaml
```

### VLM 示例

位于：

```text
examples/vlm/
```

当前包括：

* `llava/`：LLaVA 风格训练示例
* `qwenvl/`：Qwen-VL 风格训练示例

其中以 `examples/vlm/llava/` 为例，包括：

* `pretrain.py`：多模态预训练示例
* `sft.py`：多模态监督微调示例
* `chat.py`：对话/推理示例
* `config/`：对应配置文件

示例命令：

```bash
cookiee train examples/vlm/llava/pretrain.py --config_path examples/vlm/llava/config/pretrain.yaml
cookiee train examples/vlm/llava/sft.py --config_path examples/vlm/llava/config/sft.yaml
```
