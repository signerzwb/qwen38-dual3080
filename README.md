# Qwen3.8-27B 双 3080 20G SGLang 本地推理还原包

在 `双 x RTX 3080 20GB + WSL2 Ubuntu` 上跑 `Qwen3.8-27B NVFP4 + DFlash2 投机解码 + HiCache` 的
完整还原方案。本仓库包含：启动脚本、systemd 服务、WSL 保活脚本、测试客户端、SGLang 打补丁源码树
（artifacts/sglang-commit.tar.gz）、FlashInfer 0.6.17 运行目录（artifacts/flashinfer-0.6.17-test.tar.gz）、
pip 清单，以及上游复刻包 stack-package/（38 个 patch + 完整技术指南）。

## 三套配置一览

| 配置 | GPU 占用 | 上下文 | 视觉 | decode 速度 | 用途 |
|---|---|---|---|---|---|
| A 视觉版（默认生产） | GPU1+GPU2 | 226816 tokens（~222K，实测 220K 稳定） | 内置 | ~85 tok/s | 日常使用，可传图 |
| B 无视觉 256K | GPU1+GPU2 | 262144 tokens（满 256K，实测 250K 通过） | 不可用 | ~85.5 tok/s | 纯文本超长上下文 |
| C 无视觉主服务 + 独立视觉 encoder（EPD） | GPU1+GPU2 主 + 另一张空闲卡 encoder | 主服务 262144（~256K） | 可用（走独立 encoder） | ~85 tok/s | 既要 256K 又要看图 |

★ 配置 C 说明：无视觉版主服务可以配合另一张空闲卡跑独立视觉 encoder（SGLang EPD 模式），
主服务不在本地保留视觉塔权重，仍保有满 256K 上下文；图片请求经 zmq 转发到 encoder 卡处理。
详见第 7 节。

## 目录结构

```
scripts/
  start-sglang-qwen38-fp4-dflash-test.sh   # 核心启动脚本（NO_VISION 开关 + MEM_FRACTION_STATIC 可调）
  start-sglang-3080x2-qwen38.bat           # 配置 A 桌面启动器（含 WSL 保活检查）
  start-sglang-3080x2-qwen38-novision.bat  # 配置 B 桌面启动器
  wsl-keepalive.vbs                        # WSL 保活（放入启动目录，开机自启）
  sglang_fp4_test_client.py                # 测试客户端（text/image/long）
systemd/
  sglang-qwen38.service                    # 配置 A systemd 单元
  sglang-qwen38-novision.service           # 配置 B systemd 单元
config/
  wslconfig.txt                            # 拷到 C:\Users\<user>\.wslconfig
docs/
  Qwen3.8-27B-NVFP4-DFlash2-HiCache-2x3080.md   # 完整技术文档（含 WSL 保活根因、256K 实测）
stack-package/                             # 上游复刻包 v2.3（patches/conf/scripts/monitor/指南）
artifacts/
  sglang-commit.tar.gz                     # 打补丁后的 SGLang 源码树（30MB，免 clone+apply）
  flashinfer-0.6.17-test.tar.gz            # FlashInfer 0.6.17 运行目录（17MB）
  pip-freeze-sglang-awq-test.txt           # conda 环境完整包清单
## 1. 硬件与软件环境

| 项 | 说明 |
|---|---|
| GPU | 双 x RTX 3080 20GB（SM86/Ampere），TP=2 单并发；本机用物理 GPU1+GPU2（CUDA_VISIBLE_DEVICES=1,2），GPU0 上有其它服务，勿动 |
| 系统内存 | 128GB+；WSL 上限 110GB，运行稳态实际约 30GB（含 HiCache 12G×2 rank） |
| 系统 | Windows 11 + WSL2 Ubuntu（WSL 2.5） |
| Python | 3.10，conda 环境 sglang-awq-test（约 9.3GB，未进仓库，按 pip 清单重建） |
| 核心依赖 | torch 2.11.0+cu130、sglang-kernel 0.4.2.post2、flashinfer-python 0.6.11.post1（运行时用 0.6.17-test 经 PYTHONPATH 覆盖）、transformers 5.6.0、triton 3.6.0 |
| SGLang | 定制打补丁源码树：base commit 0da6a6685648a415818bfa2e44471cb884009f35 + 38 个 patch（合并版 stack-package/patches/qwen38-0909-merged.patch） |

注意：这是为 Ampere（SM86）NVFP4 专门调通的方案，依赖本仓库的 SGLang 源码树 + FlashInfer 0.6.17 运行目录，不能直接 pip install sglang 通用版本。

## 2. 模型下载

魔塔（ModelScope）没有这两个模型（截至 2026-09-13），走 Hugging Face，国内可用 hf-mirror 镜像。

| 文件 | 地址 | 大小 | 本地目标路径 |
|---|---|---|---|
| 主模型 NVFP4（含视觉塔） | https://huggingface.co/piscesbody/Qwen3.8-27B-Uncensored-NVFP4 | 约 21.1GB | /home/&lt;user&gt;/models/Qwen3.8-27B-Uncensored-NVFP4 |
| DFlash2 草稿模型 | https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2 | 约 3.6GB | /home/&lt;user&gt;/models/Qwen3.8-27B-DFlash2 |

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download piscesbody/Qwen3.8-27B-Uncensored-NVFP4 --local-dir /home/<user>/models/Qwen3.8-27B-Uncensored-NVFP4
huggingface-cli download z-lab/Qwen3.8-27B-DFlash2 --local-dir /home/<user>/models/Qwen3.8-27B-DFlash2
```

要求：
- 模型必须放 WSL Linux 本地文件系统，放 Windows 挂载盘（/mnt/d 等）加载和推理都会明显变慢。
- 主模型目录内含视觉塔（约 0.5GB）；无视觉模式只是不加载它。
- 第 7 节的 EPD 独立 encoder 也需要含视觉塔的模型目录，可直接复用主模型目录。

## 3. 环境还原

### 3.1 conda 环境

```bash
conda create -n sglang-awq-test python=3.10 -y
conda activate sglang-awq-test
pip install -r artifacts/pip-freeze-sglang-awq-test.txt
python -c "import torch, transformers, triton; print(torch.__version__, transformers.__version__, triton.__version__)"
```

预期打印：2.11.0+cu130、5.6.0、3.6.0。

### 3.2 SGLang 打补丁源码树

方式一（推荐，直接解压现成的打补丁树）：

```bash
mkdir -p /home/<user>/sglang-qwen38-fp4-test
tar -xzf artifacts/sglang-commit.tar.gz -C /home/<user>/sglang-qwen38-fp4-test
mv /home/<user>/sglang-qwen38-fp4-test/sglang-0da6a6685648a415818bfa2e44471cb884009f35 /home/<user>/sglang-qwen38-fp4-test/sglang-src
```

artifacts/ 里的 30MB tar 包就是打好 38 个 patch 的完整源码树，启动脚本的 PYTHONPATH 指向 sglang-src/python。

方式二（clone + apply 复现，需要看 diff 时用）：

```bash
git clone https://github.com/sgl-project/sglang /home/<user>/sglang-qwen38-fp4-test/sglang-src
cd /home/<user>/sglang-qwen38-fp4-test/sglang-src
git fetch origin 0da6a6685648a415818bfa2e44471cb884009f35
git checkout FETCH_HEAD
git apply /path/to/repo/stack-package/patches/qwen38-0909-merged.patch
git rev-parse HEAD^{tree}
```

最终 tree hash 应为 53bbd2f6d78d9021a1e0e7021a263fea3081b507。

### 3.3 FlashInfer 0.6.17 运行目录

```bash
tar -xzf artifacts/flashinfer-0.6.17-test.tar.gz -C /home/<user>/
```

启动脚本会把 /home/<user>/flashinfer-0.6.17-test 加进 PYTHONPATH，覆盖 pip 装的 flashinfer 0.6.11.post1，不要删这个目录。

## 4. 脚本与服务安装

1. systemd 单元的 ExecStart 指向 /mnt/d/qwen38-dual3080/scripts/...（WSL 里的 D 盘仓库路径）。仓库换位置必须同步改 systemd/ 下两个单元。
2. 把 scripts/start-sglang-qwen38-fp4-dflash-test.sh 顶部的 4 个硬编码路径替换为实际路径：SRC（sglang-src/python）、PY（conda 环境 python）、MODEL（主模型目录）、DRAFT（草稿模型目录）。
3. 安装单元：

```bash
sudo cp systemd/sglang-qwen38.service systemd/sglang-qwen38-novision.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sglang-qwen38
```

配置 A 和配置 B 共用端口 8788，同一时间只能跑一套。

4. 桌面启动器（可选）：start-sglang-3080x2-qwen38.bat = 配置 A，start-sglang-3080x2-qwen38-novision.bat = 配置 B，双击即启动，会顺带自检 WSL 保活。
5. API key：默认不设置，请求头 Authorization 可以带任意值（本机用 sk-local 占位）。要启用 key 就在启动脚本加 --api-key <yourkey>。
## 5. 配置 A：视觉版（默认生产）

- 单元 sglang-qwen38.service：NO_VISION=0、MEM_FRACTION_STATIC=0.93，context-length / max-total-tokens 均为 262144。
- 0.93 下 KV 池只能分到 226816 tokens（约 222K），实际可用上下文约 222K。
- 视觉内置（--enable-multimodal），图片请求可用。
- 实测：220K 单请求稳定、220K×3 请求也稳定，decode 约 84.8-85 tok/s。

```bash
sudo systemctl start sglang-qwen38
```

## 6. 配置 B：无视觉 256K 满上下文

- 单元 sglang-qwen38-novision.service：NO_VISION=1、MEM_FRACTION_STATIC=0.95。
- 脚本自动加 --language-only（只加载 LM 权重，省掉约 0.5G 视觉塔），KV 池扩到 262144 = 满 256K。
- 0.95 是 20G 卡的上限：峰值 19.5G/rank、终态剩 0.65G/rank，不要再高。
- 实测：250K 输入 prefill 约 770 tok/s（TTFT 325.8s），decode 约 85.5 tok/s，无 OOM。
- 无视觉：不能处理图片请求。
- 星号说明：无视觉版可配合另一张空闲卡搭配辅助视觉使用——主服务保持满 256K 上下文的同时也能收图，做法是 SGLang EPD 拆分（配置 C，见第 7 节）。

```bash
sudo systemctl start sglang-qwen38-novision
```

## 7. 配置 C：无视觉主服务 + 独立视觉 encoder（EPD）

目标：主服务不在本地保留视觉塔（保住满 256K 上下文），图片请求经 zmq 转发给另一张空闲卡上的独立 encoder 处理。

架构：

```text
客户端 -> 主服务 127.0.0.1:8788   （GPU1+GPU2，TP=2，无视觉 256K）
              | 图片特征请求（zmq_to_scheduler）
              v
        encoder 服务 127.0.0.1:8779 （另一张空闲卡，--encoder-only，持有视觉塔）
```

主服务 = 配置 B 基础 + 4 个参数：

```bash
--enable-multimodal \
--language-only \
--encoder-urls http://127.0.0.1:8779 \
--encoder-transfer-backend zmq_to_scheduler
```

encoder 服务 = 在一张空闲卡上跑（本机候选 GPU3，约 10G 空闲；encoder 只做视觉塔和投影，不跑 LM，显存需求不高）：

```bash
CUDA_VISIBLE_DEVICES=<encoder_gpu> python -m sglang.launch_server \
  --model-path /home/<user>/models/Qwen3.8-27B-Uncensored-NVFP4 \
  --served-model-name Qwen3.8-27B-encoder \
  --host 127.0.0.1 --port 8779 \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler
```

启动顺序：先起 encoder，等它 /health 就绪再启主服务（boot-sgl-stack.sh 里有这段等待逻辑，可照抄）。

现成脚本（上游 stack-package v2.3）：
- stack-package/scripts/start-sglang-qwen38-encoder.sh 和 stop-sglang-qwen38-encoder.sh
- stack-package/scripts/boot-sgl-stack.sh（先 encoder 后主服务，带健康等待）
- GPU / 端口 / 模型路径在 stack-package/conf/sglang_qwen38_27b_dflash2.conf.sh 里可配：ENCODER_GPU、ENCODER_PORT、ENCODER_MODEL_PATH

说明：配置 C 是基于上游 stack-package 的 EPD 实践整理的参考配置，本机尚未做端到端验证；要启用时按本节顺序测试（encoder 起来、主服务起来、图片请求冒烟）。文本速度与配置 B 一致（约 85 tok/s）。

## 8. WSL 保活（必做，否则服务会随机消失）

根因（2026-09-13 定位）：WSL 2.5 的 idle 停机机制。vmIdleTimeout 默认 60 秒，WSL VM 空闲超时后会停掉发行版用户空间（所有用户服务被杀），但 WSL 内核进程保留，所以现象像每隔几分钟整体重启一次，实际是用户空间被 idle 停机。

修复三件套，缺一不可：

1. 写 C:\Users\<user>\.wslconfig（内容见 config/wslconfig.txt）：

```ini
[wsl2]
memory=110GB
processors=40
vmIdleTimeout=86400000
```

2. 常驻进程 wsl -d Ubuntu --exec tail -f /dev/null（让 WSL 永远处于非 idle 状态）。scripts/wsl-keepalive.vbs 会静默拉起它，放进启动项目录（shell:startup）即开机自启。
3. 改完配置后在 Windows 侧执行 wsl --shutdown 再重新进入，使 .wslconfig 生效。

验证：

```powershell
wsl -l -v
wsl -d Ubuntu -- pgrep -f "tail -f /dev/null"
```

Ubuntu 应为 Running，且保活进程存在。

没做保活时的症状：服务之前正常，过几分钟 127.0.0.1:8788 就不响应了，wsl -l -v 显示 Stopped，发行版内 systemd 单元也没了。
## 9. 实测数据（双 3080 20G、TP=2、单并发、DFlash2 n=8、HiCache 12G/rank）

速度：

| 场景 | 结果 |
|---|---|
| 配置 A 视觉版 decode | 约 84.8 tok/s |
| 配置 B 无视觉 decode | 约 85.5 tok/s |
| 5K 上下文 decode | 113.7 tok/s |
| 220K 上下文 decode | 80-88 tok/s（稳定） |
| 250K 输入 prefill | 约 770 tok/s，TTFT 325.8s |
| 冷启动首条请求 | 约 30 tok/s（内核懒加载，第 2 条恢复满速，属正常） |

显存（每卡 20.5G，/rank）：

| 项 | 配置 A（视觉） | 配置 B（无视觉） |
|---|---|---|
| 主模型权重 | 10.09G | 9.60G |
| DFlash2 草稿 | 2.09-2.19G | 2.09-2.19G |
| Mamba 状态 | 约 0.34G | 约 0.34G |
| KV cache | fp8_e4m3，约 10.5KB/token | 同 |
| KV 池 | 226816 tokens（0.93） | 262144 tokens 满 256K（0.95） |
| 峰值占用 | 约 19.5G/rank | 19.5G/rank |
| 终态空闲显存 | 约 0.65G/rank | 约 0.65G/rank |

系统内存：WSL 上限 110GB，稳态运行约 30GB（含 HiCache 12G×2 rank）。不需要长文本缓存复用时可把 HICACHE_SIZE_GB 调小或关闭分层缓存，进一步降内存。

## 10. 验证命令

```powershell
Invoke-RestMethod http://127.0.0.1:8788/health
Invoke-RestMethod http://127.0.0.1:8788/v1/models
```

```bash
curl -sS http://127.0.0.1:8788/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen3.8-27B-FP4-Dflash-test\",\"messages\":[{\"role\":\"user\",\"content\":\"只回复：文本正常\"}],\"max_tokens\":32,\"temperature\":0}"
```

测试客户端（必须用 WSL 里的 python 跑，tokenizer 路径是 Linux 路径）：

```bash
PY=/home/<user>/miniconda3/envs/sglang-awq-test/bin/python
$PY /mnt/d/qwen38-dual3080/scripts/sglang_fp4_test_client.py --kind text
$PY /mnt/d/qwen38-dual3080/scripts/sglang_fp4_test_client.py --kind image
$PY /mnt/d/qwen38-dual3080/scripts/sglang_fp4_test_client.py --kind long --tokens 250000 --max-tokens 128
```

显存 / 日志：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
tail -n 200 /home/<user>/logs/sglang-qwen38-fp4-dflash-test.log
```

## 11. 已知问题与坑

1. 启动日志里 fp8 kv cache has no scaling factors 告警正常，不影响 fp8_e4m3 KV。
2. libavutil 缺失只影响视频请求，文本和图不受影响。
3. 首条请求约 30 tok/s（内核懒加载预热），第 2 条起满速，不是故障。
4. 单并发设计（--max-running-requests 1），多并发加长文本会挤 KV 池，易 OOM。
5. HiCache 不要上 24G/rank（合计 48G，实测 OOM），12G/rank 是安全值。
6. 20G 卡 mem-fraction-static 上限 0.95（剩 0.65G 空闲），再高大概率启动 OOM。
7. --language-only 与 --language-model-only 易混：前者是只加载 LM 权重省显存（配置 B/C 用）；后者是 EPD 拆分里的进程角色标志，不省显存。用错会导致视觉塔仍在内存里。
8. WSL 保活没做，服务会随机消失（第 8 节），新机器先做这个。
9. 本仓库单元 ExecStart 指向 /mnt/d/qwen38-dual3080/ 路径，仓库换位置要同步改。
10. 本机 transient systemd 单元偶发 stop 报 device or resource busy，换新单元名即可，属机器特性，不是服务问题。

## 快速还原（新机器 / 重装后，5 步）

1. Windows 11 + WSL2 Ubuntu + NVIDIA 驱动，nvidia-smi 在 WSL 里确认两张 3080 20G。
2. WSL 保活：config/wslconfig.txt + wsl-keepalive.vbs + wsl --shutdown（第 8 节）。
3. 环境：conda 环境（3.1）+ SGLang 打补丁源码树（3.2）+ FlashInfer 运行目录（3.3）；下载两个模型（第 2 节）。
4. 脚本与服务：替换 sh 脚本里 4 个硬编码路径，安装 systemd 单元（第 4 节）。
5. 选配置 A/B/C 之一启动，按第 10 节验证，跑 --kind text 和 --kind long 测速与第 9 节对账；速度明显低于 85 tok/s 先查 3.2/3.3（SGLang 树和 FlashInfer）与第 8 节（保活）。
