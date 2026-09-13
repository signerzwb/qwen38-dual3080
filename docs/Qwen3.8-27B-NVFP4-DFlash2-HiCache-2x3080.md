# Qwen3.8-27B NVFP4 + DFlash2 + HiCache 双 3080 方案

更新时间：2026-09-10

## 1. 结论

当前已经跑通一套可用于本地 Codex/Qwen 推理的方案：

- 模型：`Qwen3.8-27B-Uncensored-NVFP4`
- 推理框架：SGLang
- 投机解码：DFlash2
- 视觉：开启
- GPU：物理 GPU 1、2，两块 RTX 3080 20GB
- TP：2
- 监听：`127.0.0.1:8788`
- 模型名：`Qwen3.8-27B-FP4-Dflash-test`
- HiCache：开启，使用系统内存作为 KV 二级缓存

当前设备的实际活跃上下文上限：

- `context-length=262144`
- 实际 GPU KV 池：`226816 tokens`
- 推荐单请求输入：不超过约 `220K`
- 完整 256K 单请求仍然无法稳定运行

HiCache 可以把历史 KV 和 Mamba 状态备份到内存，减少后续重复预填充，但活跃请求仍必须放进 GPU KV 池。因此 HiCache 不能把单个 256K 请求变成 GPU 可同时计算。

## 2. 硬件和系统

### GPU

本方案只使用：

- 物理 GPU 1：RTX 3080 20GB
- 物理 GPU 2：RTX 3080 20GB

不使用 GPU 0 和 GPU 3。

### 内存

WSL2 报告：

```text
Total: 108 GiB
HiCache 实际 pinned memory: 约 31.3 GiB
系统剩余可用: 约 68 GiB
```

### 软件

SGLang 源码：

```text
/home/dministrator/sglang-qwen38-fp4-test/sglang-src/python
```

SGLang Python 环境：

```text
/home/dministrator/miniconda3/envs/sglang-awq-test
```

已验证版本：

```text
sglang 0.5.12.post1
torch 2.11.0+cu130
FlashInfer Python 0.6.17
系统 nvcc 12.0
```

模型：

```text
/home/dministrator/models/Qwen3.8-27B-Uncensored-NVFP4
```

DFlash2 草稿模型：

```text
/home/dministrator/models/Qwen3.8-27B-DFlash2
```

## 3. 当前链路

```text
Codex / 客户端
  -> http://127.0.0.1:8788/v1
  -> SGLang
  -> NVFP4 target model
  + DFlash2 draft model
  + GPU KV
  + HiCache host KV
```

当前没有配置 FRP，也没有公网暴露。服务只监听本机回环地址。

## 4. 当前稳定启动参数

启动脚本：

```text
/mnt/d/codexproj/chat/scripts/start-sglang-qwen38-fp4-dflash-test.sh
```

脚本关键默认值：

```bash
HICACHE_SIZE_GB=12
MEM_FRACTION_STATIC=0.93
```

当前实际启动命令的核心参数：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_DISABLE_HICACHE_MHA_STAGED_WRITE_BACK=1
export SGLANG_USE_HICACHE_SAFE_PAGE_FIRST_WRITE_BACK=1

python -m sglang.launch_server \
  --model-path /home/dministrator/models/Qwen3.8-27B-Uncensored-NVFP4 \
  --served-model-name Qwen3.8-27B-FP4-Dflash-test \
  --host 127.0.0.1 \
  --port 8788 \
  --tp-size 2 \
  --trust-remote-code \
  --enable-multimodal \
  --context-length 262144 \
  --max-total-tokens 262144 \
  --max-running-requests 1 \
  --mem-fraction-static 0.93 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-mixed-chunk \
  --chunked-prefill-size 2048 \
  --disable-prefill-cuda-graph \
  --attention-backend flashinfer \
  --mamba-backend flashinfer \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-ssm-dtype bfloat16 \
  --max-mamba-cache-size 8 \
  --mamba-max-states-per-path 6 \
  --mamba-track-interval 512 \
  --page-size 64 \
  --radix-eviction-policy lru \
  --enable-hierarchical-cache \
  --hicache-size 12 \
  --hicache-io-backend kernel \
  --hicache-mem-layout page_first \
  --hicache-write-policy write_back \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /home/dministrator/models/Qwen3.8-27B-DFlash2 \
  --speculative-draft-model-quantization unquant \
  --speculative-num-draft-tokens 8 \
  --speculative-draft-window-size 2048 \
  --speculative-draft-attention-backend flashinfer \
  --speculative-draft-kv-cache-dtype fp8_e4m3 \
  --enable-linear-replayssm-spec
```

注意：

- `--context-length` 可以写 `262144`，但实际可用 KV 池由显存决定。
- `--max-total-tokens=262144` 会被 SGLang 降到 profiled 值。
- 当前 profiled 值为 `226816`。
- 不要为了追 256K 把 `mem-fraction-static` 提到 `0.94` 或 `0.95`。

## 5. 显存配置

`mem-fraction-static` 对比：

| 配置 | GPU KV 池 |
|---|---:|
| `0.90` | `198848 tokens` |
| `0.93` | `226816 tokens` |

当前使用 `0.93`。

220K 长文请求时：

```text
GPU1 peak: 19312 MiB
GPU2 peak: 19040 MiB
最低空闲显存: 约 0.49 GiB
```

这已经是贴近上限运行。继续升高 fraction 的收益有限，并且 OOM 风险明显增加。

## 6. HiCache 配置

当前 `--hicache-size 12` 是每个 TP rank 的配置，TP=2 下实际主 HiCache 约 24GB，另有 DFlash/Mamba sidecar。

实际分配：

| 池 | 容量 | 内存 |
|---|---:|---:|
| 主 KV host pool | 651392 tokens | 10.67 GB/rank |
| Mamba host pool | 同主池规模 | 1.65 GB/rank |
| DFlash draft host pool | 651456 tokens | 3.34 GB/rank |

TP=2 合计 pinned memory 约 `31.3 GB`。

HiCache 的作用：

- GPU KV 满时，把可淘汰节点备份到 host memory
- 后续相同或相似前缀请求可以优先复用 host KV
- Mamba 状态也支持 host backup
- 不会扩大单请求的 active GPU KV 上限

日志中可以观察到：

```text
MAMBA-EVICT target=EvictLayer.HOST
MAMBA-EVICT-BACKUP
HiCache attached=True
```

## 7. 实测性能

### DFlash 基础性能

以下数据来自 DFlash 131K 配置：

| 输入 | Decode |
|---|---:|
| 50K | 122.5 tok/s |
| 100K | 124.1 tok/s |
| 120K | 150.5 tok/s |
| 128K | 146.5 tok/s |

DFlash 接受率：

- 50K：接受长度约 `6.83/8`，接受率约 `83%`
- 120K/128K：接受长度 `8/8`，接受率 `100%`

### 220K + HiCache + mem-fraction 0.93

冷启动：

```text
输入: 220000 tokens
TTFT: 280.419 s
输出: 256 tokens
Decode: 86.557 tok/s
GPU1 peak: 19312 MiB
GPU2 peak: 19040 MiB
```

连续三次相同前缀复用：

| 次数 | TTFT | Decode |
|---|---:|---:|
| 1 | 1.866 s | 97.034 tok/s |
| 2 | 1.903 s | 75.571 tok/s |
| 3 | 1.859 s | 94.984 tok/s |

三次请求显存峰值一致，没有增长，服务保持健康。

### 视觉

图片测试正常识别：

```text
圆形/蓝色/42
```

视觉请求峰值：

```text
GPU1: 18906 MiB
GPU2: 18634 MiB
```

## 8. 必要修复记录

### FlashInfer 版本

系统环境原先安装：

```text
flashinfer-python 0.6.11.post1
flashinfer-cubin 0.6.11.post1
```

DFlash 与 SGLang 需要较新的 FlashInfer Python API。

当前使用隔离目录：

```text
/home/dministrator/flashinfer-0.6.17-test
```

启动时通过 PYTHONPATH 优先加载：

```bash
export PYTHONPATH="/home/dministrator/sglang-qwen38-fp4-test/sglang-src/python:/home/dministrator/flashinfer-0.6.17-test"
```

由于官方没有发布匹配的 `flashinfer-cubin 0.6.17`，当前使用：

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

### CUDA 12.0 CUB 兼容

FlashInfer 0.6.17 的默认判断要求 CUDA `>=12.1` 才启用 `SubtractLeft`，但本机系统 nvcc 是 12.0，且捆绑的 CUB 已经包含该接口。

补丁文件：

```text
/mnt/d/codexproj/chat/scripts/flashinfer-0617-cub-120-subtractleft.patch
```

修改内容：

```diff
-#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120100)
+#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120000)
 #define FLASHINFER_CUB_SUBTRACTLEFT_DEFINED
 #endif
```

### WSL HiCache 写入路径

为避免 WSL 下 MHA staged write-back 和 pinned memory 路径异常，使用：

```bash
export SGLANG_DISABLE_HICACHE_MHA_STAGED_WRITE_BACK=1
export SGLANG_USE_HICACHE_SAFE_PAGE_FIRST_WRITE_BACK=1
```

## 9. 启动和验证

### 启动

```powershell
wsl -d Ubuntu -- bash -lc "bash /mnt/d/codexproj/chat/scripts/start-sglang-qwen38-fp4-dflash-test.sh 262144 262144 8788"
```

### 健康检查

```powershell
wsl -d Ubuntu -- bash -lc "curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8788/health"
```

预期：

```text
200
```

### 查看模型

```powershell
wsl -d Ubuntu -- bash -lc "curl -sS http://127.0.0.1:8788/v1/models"
```

### 查看日志

```powershell
wsl -d Ubuntu -- bash -lc "tail -n 200 /home/dministrator/logs/sglang-qwen38-fp4-dflash-test.log"
```

### 测试 220K

```powershell
wsl -d Ubuntu -- bash -lc "export PYTHONPATH=/home/dministrator/sglang-qwen38-fp4-test/sglang-src/python:/home/dministrator/flashinfer-0.6.17-test; /home/dministrator/miniconda3/envs/sglang-awq-test/bin/python /mnt/d/codexproj/chat/scripts/sglang_fp4_test_client.py --base-url http://127.0.0.1:8788/v1 --model Qwen3.8-27B-FP4-Dflash-test --kind long --context 262144 --tokens 220000 --max-tokens 256 --prompt-suffix '请从1数到200，用逗号分隔，只输出数字。' --log-file /home/dministrator/logs/sglang-qwen38-fp4-dflash-test.log --timeout 1800"
```

## 10. 已知问题

1. 完整 256K 活跃上下文不可用。
2. 220K 时显存余量很薄，最低约 0.49GB。
3. `0.94` 及以上没有足够安全收益，不建议使用。
4. FlashInfer cubin 版本检查目前被关闭，因为官方缺少 0.6.17 cubin 包。
5. `libtorchcodec`/FFmpeg 加载失败会在日志出现，但图片输入可用，只影响视频解码。
6. 首次启动时 FlashInfer/Marlin 等 CUDA JIT 可能非常慢，缓存完成后后续启动会明显加快。
7. HiCache pinned memory 在 WSL 下不能无限增加；24GB/rank 会碰到 pinned/GPU 侧 OOM。
8. 当前配置为单并发，未做多用户并发压测。

## 11. 推荐运行边界

建议：

```text
普通使用上下文: 64K-160K
长文稳定档位: 196K-220K
视觉请求: 保持开启
主动上下文: 1
mem-fraction-static: 0.93
hicache-size: 12
```

不建议：

```text
单请求 256K
mem-fraction-static 0.94+
hicache-size 24/rank
关闭视觉来硬追上下文
```

## 12. 2026-09-13 更新：WSL 保活 + 无视觉 256K 选项

### 12.1 WSL "反复重启" 根因与修复

现象：服务每隔约 45-90 秒被重置一次，模型永远加载不完。

根因：不是系统周期任务（计划任务/systemd 定时器/cron 均已排查），而是 WSL 2.5 的
空闲停机：发行版没有活动会话时，超过 vmIdleTimeout（默认 60s，.wslconfig 未配置）
就杀掉该发行版的 init 和全部用户空间进程；底层 VM 内核可以保持存活（内核 uptime /
boot_id 不变），表现为"用户空间反复重新初始化"。journalctl 只有 "Started" 没有
"exit"、PID 反复复用都是该现象的指纹。

修复（全部已部署并验证）：

```text
1. 常驻保活会话:  wsl -d Ubuntu --exec tail -f /dev/null  (隐藏窗口后台运行)
   有活动会话 => 永不被判空闲 => 永不被停
2. .wslconfig 增加 vmIdleTimeout=86400000 (双保险, 下次 VM 启动生效)
3. 开机自启保活:  启动目录 3080双卡qwen38-wsl保活.vbs
4. 桌面快捷方式 bat 更新: 先确保保活会话存在, 再检查/启动服务
```

验证：冷启动（wsl --shutdown 后直接跑 bat）约 2.5 分钟到 ready；
有保活会话后服务连续运行不再被重置。

### 12.2 无视觉 256K 满上下文选项

目标：放弃视觉输入，把 KV 池做到 262144。

实测（TP=2 双 3080 20G，NVFP4 + DFlash + HiCache12G）：

```text
视觉版(原生产配置, 0.93):  KV 池 226816 tokens (~222K), 主模型权重 10.09GB/rank
无视觉 @0.93 (--language-only): KV 池 252288 tokens (~246K), 权重 9.60GB/rank
无视觉 @0.95 (--language-only): KV 池 262144 tokens (满 256K!), 权重 9.60GB/rank
                                  终态空闲显存 0.65GB/rank
```

关键点：
- 正确的标志是 --language-only（VLM 只载语言模型权重）；
  --language-model-only 是 EPD 独立进程模式，不是省显存用的，别用错。
- 视觉塔只占约 0.5GB/rank，单靠它不够补 256K 缺口，需要配合 0.95。
- 启动方式: NO_VISION=1 + MEM_FRACTION_STATIC=0.95 跑
  start-sglang-qwen38-fp4-dflash-test.sh（脚本已内置 NO_VISION 开关）。

端到端验证：250K token 输入请求 200 OK，TTFT 325.8s（prefill 约 770 tok/s），
双卡峰值 19.5G/20.5G，输出正常。

取舍：
- 换来满 256K；代价是图片输入不可用、显存余量更薄（0.65GB/rank）。
- 250K prefill 约 5.4 分钟，长文场景 TTFT 体验差，属正常物理限制。
- 若需要视觉，恢复生产配置: 停掉 nov 单元, systemctl start sglang-qwen38。

### 12.3 启动脚本改动

start-sglang-qwen38-fp4-dflash-test.sh:
- 新增 NO_VISION 环境变量开关（1 => --language-only, 0 => --enable-multimodal）
- 原参数（CTX MAX_TOTAL PORT）不变

start-sglang-3080x2-qwen38.bat（桌面快捷方式）:
- 新增 [1/4] 保活会话检查/拉起
- CIM 查询不用 -Filter 参数（引号经 WSL/cmd 转义易错），改 PS 端 Where-Object
