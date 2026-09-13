---
name: qwen38-dual3080
description: 在双 RTX 3080 20GB + WSL2 Ubuntu 机器上还原和运行 Qwen3.8-27B NVFP4 + DFlash2 + HiCache SGLang 推理栈（三套配置：视觉版 / 无视觉 256K / EPD 独立视觉 encoder）。用于重装系统或换机后还原本服务、启动/停止/验证 127.0.0.1:8788 服务、排查速度慢或服务消失问题。
---

# Qwen3.8-27B 双 3080 20G SGLang

完整细节在本仓库 README.md（11 节 + 快速还原），技术推导在 docs/Qwen3.8-27B-NVFP4-DFlash2-HiCache-2x3080.md。本 skill 只给执行入口。

## 前提检查

```bash
nvidia-smi -L    # 需要 2 x RTX 3080 20GB（SM86）
wsl -l -v        # Ubuntu 应为 Running
```

WSL 没在运行：先做 README 第 8 节保活（.wslconfig + wsl-keepalive.vbs + wsl --shutdown），否则服务起来也会随机消失。

## 快速启动（环境已还原的机器）

```powershell
wsl -d Ubuntu -- sudo systemctl start sglang-qwen38          # 配置 A 视觉版
# 或
wsl -d Ubuntu -- sudo systemctl start sglang-qwen38-novision # 配置 B 无视觉 256K
Invoke-RestMethod http://127.0.0.1:8788/health
```

停止：对应 `systemctl stop`。日志：/home/&lt;user&gt;/logs/sglang-qwen38-fp4-dflash-test.log。

## 从零还原（重装 / 新机器）

1. Windows 11 + WSL2 Ubuntu + NVIDIA 驱动，WSL 内 nvidia-smi 确认两张 3080 20G。
2. WSL 保活（README 第 8 节，必须先做）。
3. 模型（README 第 2 节，魔塔没有，走 HF / hf-mirror）：
   - 主模型 piscesbody/Qwen3.8-27B-Uncensored-NVFP4 到 /home/&lt;user&gt;/models/
   - 草稿 z-lab/Qwen3.8-27B-DFlash2 到 /home/&lt;user&gt;/models/
4. 环境（README 第 3 节）：conda env sglang-awq-test（python 3.10，按 artifacts/pip-freeze-sglang-awq-test.txt 装）；artifacts/sglang-commit.tar.gz 解压后重命名为 sglang-src；artifacts/flashinfer-0.6.17-test.tar.gz 解压到 home 目录。
5. 替换 scripts/start-sglang-qwen38-fp4-dflash-test.sh 顶部 4 个硬编码路径（SRC/PY/MODEL/DRAFT）。
6. 安装 systemd 单元（systemd/ 两个 .service；ExecStart 默认指向 /mnt/d/qwen38-dual3080/，仓库换位置要改）。
7. 验证：health + v1/models + 测试客户端 --kind text / --kind long 与 README 第 9 节对账。

## 三套配置

| 配置 | 关键参数 | 上下文 | 视觉 |
|---|---|---|---|
| A 视觉版 | MEM_FRACTION_STATIC=0.93，--enable-multimodal | KV 池 226816（约 222K） | 内置 |
| B 无视觉 | NO_VISION=1（--language-only）+ 0.95 | KV 池 262144（满 256K） | 不可用 |
| C EPD | B + --enable-multimodal --language-only --encoder-urls http://127.0.0.1:8779 --encoder-transfer-backend zmq_to_scheduler | 主服务满 256K | 走另一张空闲卡的 --encoder-only 服务 |

要点：无视觉（B）可配合另一张卡搭配辅助视觉使用，即配置 C；encoder 脚本在 stack-package/scripts/（start-sglang-qwen38-encoder.sh、boot-sgl-stack.sh），GPU/端口在 conf 里配（ENCODER_GPU/ENCODER_PORT）。
注意 --language-only（只载 LM 权重，省显存）和 --language-model-only（EPD 进程角色标志，不省显存）不要混用。

## 验证基线（对账用）

- decode：视觉版约 84.8 tok/s，无视觉约 85.5 tok/s，5K 上下文 113.7 tok/s，250K prefill 约 770 tok/s（TTFT 325.8s）
- 显存每卡：峰值 19.5G、终态空闲 0.65G（20G 卡 @0.95）
- 系统内存：WSL 稳态约 30GB（含 HiCache 12G×2）
- 冷启动首条请求约 30 tok/s 属正常，第 2 条恢复
- 速度明显低于 60 tok/s 时依次查：SGLang 树是否 53bbd2f6 tree、FlashInfer 0.6.17-test 是否被 PYTHONPATH 覆盖、DFlash 参数是否生效、WSL 保活

## 常见故障

- 服务随机消失：WSL 保活（README 第 8 节）
- 启动 OOM：mem-fraction 不超 0.95、HiCache 不超 12G/rank
- 图片请求失败：当前跑的是 B，换 A 或 C
- 速度低但显存正常：FlashInfer 覆盖没生效或 SGLang 树版本不对
- 401：启动脚本若加了 --api-key，请求头必须带 Authorization
