import argparse
import base64
import io
import json
import os
import subprocess
import threading
import time

import httpx
from PIL import Image, ImageDraw
from transformers import AutoTokenizer


def sample_gpu_memory(stop_event, peak):
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            for line in out.splitlines():
                index, used = [part.strip() for part in line.split(",")]
                if index in ("1", "2"):
                    peak[index] = max(peak.get(index, 0), int(used))
        except Exception:
            pass
        time.sleep(0.5)


def make_test_image():
    image = Image.new("RGB", (192, 192), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((24, 24, 168, 168), fill=(20, 90, 220))
    draw.text((80, 76), "42", fill="white", stroke_width=1, stroke_fill="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def build_messages(args):
    if args.kind == "text":
        return [
            {
                "role": "user",
                "content": "只回复：文本正常",
            }
        ]
    if args.kind == "image":
        image_b64 = make_test_image()
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "图中主要是什么形状和颜色？图中的数字是多少？"
                            "只回答：形状/颜色/数字。"
                        ),
                    },
                ],
            }
        ]
    if args.kind == "long":
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True
        )
        text = open(args.doc, encoding="utf-8").read()
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids = ids[: args.tokens]
        prompt = tokenizer.decode(ids, skip_special_tokens=True)
        return [
            {
                "role": "user",
                "content": prompt + "\n\n" + args.prompt_suffix,
            }
        ]
    raise ValueError(args.kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8788/v1")
    parser.add_argument("--model", default="Qwen3.8-27B-FP4-test")
    parser.add_argument("--kind", choices=["text", "image", "long"], required=True)
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=50_000)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-suffix", default="请只回复：OK")
    parser.add_argument("--doc", default="/home/dministrator/xiyouji-all.txt")
    parser.add_argument(
        "--tokenizer",
        default="/home/dministrator/models/Qwen3.8-27B-Uncensored-NVFP4",
    )
    parser.add_argument(
        "--log-file", default="/home/dministrator/logs/sglang-qwen38-fp4-test.log"
    )
    parser.add_argument(
        "--result-file",
        default="/home/dministrator/logs/sglang-fp4-test-results.jsonl",
    )
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()

    messages = build_messages(args)
    prompt_text = json.dumps(messages, ensure_ascii=False)
    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    log_start = 0
    if os.path.exists(args.log_file):
        log_start = os.path.getsize(args.log_file)

    peak = {}
    stop = threading.Event()
    sampler = threading.Thread(
        target=sample_gpu_memory, args=(stop, peak), daemon=True
    )
    sampler.start()

    t0 = time.perf_counter()
    first = None
    chunks = []
    usage = None
    stream_done = False
    status = None
    error = None
    try:
        with httpx.stream(
            "POST",
            args.base_url.rstrip("/") + "/chat/completions",
            json=payload,
            timeout=args.timeout,
        ) as response:
            status = response.status_code
            if status != 200:
                error = response.read().decode("utf-8", errors="replace")[:1000]
            else:
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    if first is None:
                        first = time.perf_counter()
                    data = line[5:].strip()
                    if data == "[DONE]":
                        stream_done = True
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for choice in obj.get("choices", []):
                        delta = choice.get("delta", {})
                        if delta.get("content"):
                            chunks.append(delta["content"])
                        if delta.get("reasoning_content"):
                            chunks.append(delta["reasoning_content"])
    except Exception as exc:
        error = repr(exc)
    finally:
        stop.set()
        sampler.join(timeout=2)

    t1 = time.perf_counter()
    total_s = t1 - t0
    ttft_s = (first - t0) if first else None
    completion_tokens = (usage or {}).get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = len(chunks)
    decode_s = (total_s - ttft_s) if ttft_s is not None else None
    decode_tps = (
        completion_tokens / decode_s
        if completion_tokens and decode_s and decode_s > 0
        else None
    )

    log_scan = ""
    if os.path.exists(args.log_file):
        with open(args.log_file, "rb") as fh:
            fh.seek(log_start)
            log_scan = fh.read().decode("utf-8", errors="replace")
    log_errors = [
        line
        for line in log_scan.splitlines()
        if any(
            needle in line.lower()
            for needle in (
                "out of memory",
                "cuda error",
                "illegal memory",
                "fatal python error",
                "oom",
            )
        )
    ]

    result = {
        "kind": args.kind,
        "context": args.context or None,
        "requested_tokens": args.tokens if args.kind == "long" else None,
        "status": status,
        "prompt_chars": len(prompt_text),
        "ttft_s": round(ttft_s, 3) if ttft_s is not None else None,
        "total_s": round(total_s, 3),
        "completion_tokens": completion_tokens,
        "decode_tokens_per_s": round(decode_tps, 3) if decode_tps else None,
        "stream_done": stream_done,
        "peak_gpu_mem_mib": peak,
        "answer": "".join(chunks)[:500],
        "error": error,
        "log_errors": log_errors[-10:],
    }
    text = json.dumps(result, ensure_ascii=False)
    print(text, flush=True)
    with open(args.result_file, "a", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
