# GROOT N1.7 融合 PR — 完整交接文档

> **用途**：此文档让你（或任何机器上的 Claude Code 会话）在新设备上拉取
> 仓库后，无需访问原开发机即可：理解全部工作、继续未竟事项（CUDA 13 下
> 的 fp8 真 GEMM 验证）、提交 PR、撰写报告。
>
> **写作时间**：2026-09-30。开发机（RTX 4090 D）路径
> `/home/zhangaoxiang/code/flashrt_github/FlashRT`；验证机为 AutoDL 租用
> 的 RTX 5090 容器（已无法访问，本文档为其唯一存续记录）。

---

## 1. 一句话总结

为开源 FlashRT（github.com/flashrt-project/FlashRT）的 GROOT N1.7 模型
做了 elementwise kernel 融合 + DiT packed-QKV 结构性合并 + FP8 pipeline
接线，**12 个 commit 在分支 `feature/groot-n17-ele-fusion` 上**，双卡
（4090 + 5090）差分验证全部 bit-equal，性能 4090 e2e -7.8% / 5090
-10.4%，**尚未 push、尚未发 PR**。

## 2. 仓库与分支状态

- **开源仓库本地路径**（4090 开发机）：`/home/zhangaoxiang/code/flashrt_github/FlashRT`
  （若已不可访问：`git clone https://github.com/flashrt-project/FlashRT`
  后按 §3.1 重建，但 **12 个 commit 只存在于那台机器的本地分支**，
  push 是第一优先级——见 §8）
- **分支**：`feature/groot-n17-ele-fusion`，基于 `origin/main` =
  `839b1597`（"test(amd): cover temporal KV graph refresh"）
- **12 个 commit**（全部已 commit，工作树除 `docs/superpowers/` 外干净）：

```
877d0011 docs: new-machine (RTX 5090) environment setup + PR verification manual
2777c219 perf(groot_n17): fuse the FP8-pipeline (SM120 tier) backbone elementwise chains
04323cd6 fix(test): qk-norm differential gate — fp64 cosine
3c8d13b0 scripts: one-shot RTX verification script for the groot_n17 fusion PRs
567d6fd3 perf(groot_n17): merged packed-QKV GEMM for the DiT self-attn (bf16 tier)
76e83aeb bench(groot_n17): synthetic end-to-end backbone+DiT latency bench
c01b2716 fix(groot_n17): restore the DiT FFN residual + harden the differential tests
ad972cc0 perf(groot_n17): fuse the DiT-head elementwise chains (strict, bit-identical)
ab294319 bench(groot_n17): SM89 elementwise-fusion latency benchmark + kernel docs
94123b26 perf(groot_n17): fuse the SM89 backbone elementwise chains
08c6bc2c feat(csrc): fp16 instances of the fused bias+GELU kernels
3f17ddc9 feat(csrc): qk_norm_rope_rotate_half_fp16 — template the fused QK-RMSNorm+RoPE kernel
```

- **配套文档（已 commit 进分支）**：
  - `docs/setup-new-machine-rtx5090.md` — 新机环境配置手册（驱动/CUDA/
    python/依赖踩坑全集）
  - `docs/superpowers/pr-body-groot-n17-ele-fusion.md` — PR 描述终稿
    （发 PR 时直接粘贴；最后更新含 5090 数字与两条上游发现）
  - `docs/superpowers/specs/2026-09-29-...-design.md` — 设计 spec（含
    PR-1/PR-2/PR-3 三段 addendum）
  - `docs/superpowers/plans/2026-09-29-...-fusion.md` — 实施计划全文
  - 注：`docs/superpowers/` 在 gitignore 之外（`internal-docs/` 才被
    ignore）——**检查 877d0011 之后的提交里是否包含了 pr-body 与 spec**；
    若未包含，push 前先 `git add docs/superpowers/pr-body-*.md
    docs/superpowers/specs docs/superpowers/plans`（HANDOVER 本身
    可选，建议 add）。

## 3. 核心设计决策（新会话必读）

### 3.1 改动拓扑（三个逻辑部分合成一个 PR）

| 部分 | 内容 | 关键契约 |
|---|---|---|
| A. elementwise 融合 | `qk_norm_rope_rotate_half_fp16`（模板化 Thor bf16 kernel，解除 `FLASHRT_HAVE_THOR_VLA_KERNELS` 门控）；`bias_gelu_inplace[_strict]_fp16`（G7.11 模板实例化）；`bias_residual_strict_bf16`（fp16 strict 的模板孪生）；接线 `pipeline_rtx_sm89.py` 7 位点 + 共享 `dit_forward` 2 位点（o-tail 全分支 / FFN-up bf16 分支） | strict = 两次 round 与旧链逐位一致（torch.equal 硬门槛） |
| B. packed-QKV 合并 | DiT self-attn bf16 层：3 GEMM+3 add_bias → 1 个 (Sa,4608,1536) GEMM + `qkv_split_bias_bf16` scatter | **opt-in 键控**：`weights['qkv_w'/'qkv_b']` + `bufs['qkv_buf']` 三键齐备才激活，否则回退 3-GEMM；前端 `groot_n17_rtx.py` 加载时 concat（镜像 Thor fp8 先例 `groot_n17_thor.py:678`） |
| C. FP8 pipeline 接线 | `pipeline_rtx_fp8.py`（README 5090 生产路径的 backbone）7 个同型位点 | 与 sm89 pipeline 位点逐字同型，kernel 复用 A 部分 |

### 3.2 dedup 纪律（用户强制要求，贯穿全程）

每个新 kernel 写码前全 `csrc/` 扫同类。实际结果：**12 个融合位点只有
1 个需要碰 kernel 数学**（qk_norm 的 fp16 模板化）；`qkv_split_bias_bf16`
与 `residual_add_bias_bf16` 是从 qwen3_vl 模块**迁移绑定**到主模块（kernel
原样，新增 `csrc/kernels/bias_epilogue_bf16.cuh`）。

### 3.3 测试方法学（PR 的验证骨架）

- **kernel 级差分**：融合 vs 旧链，strict 位点 `torch.equal`；qk-norm 融合
  的 reduce 重结合差用"每腿距自身 fp64 rope 参照 ≤1 fp16 ulp"判据；
- **stage 级 A/B**：同 GEMM runner、同 seed 权重（**每腿手动 reseed**，
  曾经因两腿权重不同产出假差异 cos=0.9999999——教训：fixture 的 mk_*
  必须内部 reseed）；
- **golden 冻结链测试**：1 层 DiT vs origin/main 链的逐字复制（torch
  matmul 不能当参照——bf16 split-K GEMM 归约序差 ~0.3 绝对值）。**这个
  测试抓到过真 bug**：review 阶段发现 DiT FFN 公共 residual 被误删，
  A/B 双腿共享 pipeline 体所以 bit-equal 掩盖了它（修复 commit
  `c01b2716`）；
- **路径选择守卫**：opt-in 测试必须断言新路径真的被执行（merged 测试
  检查 `qkv_buf` 非零），否则多余键被忽略、测试虚假通过（真踩过）；
- **fp64 cosine 教训**：`assert cos > 0.9999999` 这种门槛落在 fp32 的
  表示空档里（1-2⁻²⁴ 与 1-2⁻²³ 之间无 fp32 值），退化成掷硬币——4090
  过、5090 挂。修复 `04323cd6`：cosine 用 fp64 + 合理阈值。

## 4. 验证结果（已完成部分）

### 4.1 双卡数字（写进 PR body 的终值）

| bench | RTX 4090 D (torch 2.13+cu130) | RTX 5090 (torch 2.11+cu128) |
|---|---|---|
| qk-norm+RoPE pair S=257/1024/4096 | 1.46×/2.20×/3.49× | 1.30×/1.48×/**4.20×** |
| LLM stage (16L, S=1024) | -3.3% | **-6.7%** |
| DiT stage (32L, Sa=41, 含 QKV 合并) | -7.3% | — |
| FP8-pipeline LLM stage (shimmed) | -3.1% | — |
| **e2e (Se=768)** | **-7.8%** (27.66→25.51ms) | **-10.36%** (22.34→20.02ms，两次复现) |

正确性：三段差分（ele / QKV merge / fp8-pipeline wiring）**双卡全部
ALL PASS**；所有 stage 门 torch.equal。

### 4.2 e2e bench 说明

`benchmarks/groot_n17_sm89_e2e_bench.py`：合成 e2e（5 stage + DiT×4，
真实前端尺寸，stub attention，随机权重）。绝对 ms 不是重点，**delta
（相对值）才是可移植数字**。PR body 已注明。构造 fixture 的已知坑：
deepstack 的 norm_w 是 (Dmid=4096) 不是 Din；fc1_w 是 (Dmid,Dmid)；
各 stage 的 dims 是位置参数不是关键字。

## 5. 5090 验证的踩坑全集（新机器跑之前必读）

按时间序，全部在 AutoDL RTX 5090 容器实测：

| # | 症状 | 根因 | 处置 |
|---|---|---|---|
| 1 | `nvcc: command not found` | toolkit 装了但不在 PATH | `export PATH=/usr/local/cuda/bin:$PATH` + `CUDACXX` |
| 2 | `CMake 3.24 or higher required`（系统 3.22） | 系统包老 | `pip install cmake`（pip 落在 env bin，PATH 优先） |
| 3 | ptxas: `uses too much shared data (0x18400, 0xc000 max)` @ `qwen3_prefill_sm120_obj` | **上游 bug**：`fmha_fp8_causal_gqa_sm120.cu` 97KB **静态** smem > 48KB 硬限（>48KB 必须 dynamic + `cudaFuncSetAttribute`，其 sage2 兄弟 TU 是正确范例） | 临时 CMake 补丁 A：从目标摘除该 TU |
| 4 | `.so` undefined symbol `fmha_fp8_causal_gqa...` | 绑定宏 `ENABLE_QWEN3_FP8_PREFILL_ATTN` 仍定义，引用被摘 TU 的符号 | 补丁 B：同时移除宏（保留 `ENABLE_SAGE2_F8_RAW`） |
| 5 | ptxas 同类错误 @ `motus_und_ffn_sm120_obj`（66KB） | `tinyfp8_kernels_sm120.cu` 同上游 bug | 官方开关：`-DFLASHRT_ENABLE_MOTUS=OFF`（GROOT N1.7 不需要 motus） |
| 6 | qk-norm 差分 `cos=0.99999988 < 0.9999999` 断言挂 | **测试 bug**：fp32 表示空档（见 §3.3） | 已修（04323cd6），分支里是好的 |
| 7 | PIL/pandas ModuleNotFoundError（3 个测试文件收集错误） | 环境缺包，与 PR 无关 | `pip install pillow pandas` 或无视（脚本已 try 排除） |
| 8 | `fp8_descale_fp16` 全形状 code 15 | **上游 bug**：col-major **NN** 布局组合在 CUDA 12.8 cuBLASLt 不支持（同机 `torch._scaled_mm` TN 组合正常）。上游 5090 环境推测是 CUDA 13 | 见 §6 待办 |

**补丁 A+B 幂等脚本**（5090 本地构建必跑，两处 anchor 见
CMakeLists.txt `qwen3_prefill_sm120_obj` 源列表与
`ENABLE_QWEN3_FP8_PREFILL_ATTN` 宏定义；**这两个补丁不属于 PR**，
本地构建产物用，`git checkout CMakeLists.txt` 即可还原）：

```python
# python3 - << 'EOF' 形式，或并入 setup 脚本
p = 'CMakeLists.txt'; s = open(p).read()
a_old = """  add_library(qwen3_prefill_sm120_obj OBJECT
    csrc/attention/sage2/sage2_attn_f8_raw.cu
    csrc/attention/fmha_fp8_causal_gqa_sm120.cu)"""
a_new = """  add_library(qwen3_prefill_sm120_obj OBJECT
    csrc/attention/sage2/sage2_attn_f8_raw.cu)"""
b_old = """  target_compile_definitions(flash_rt_kernels PRIVATE
    ENABLE_SAGE2_F8_RAW=1
    ENABLE_QWEN3_FP8_PREFILL_ATTN=1)"""
b_new = """  target_compile_definitions(flash_rt_kernels PRIVATE
    ENABLE_SAGE2_F8_RAW=1)"""
for old, new, tag in [(a_old, a_new, 'A'), (b_old, b_new, 'B')]:
    if old in s: s = s.replace(old, new); print(tag, 'applied')
    else: print(tag, 'already/absent')
open(p, 'w').write(s)
```

configure 命令（5090）：`cmake -B build -S . -DGPU_ARCH=120
-DFA2_ARCH_NATIVE_ONLY=ON -DFLASHRT_ENABLE_MOTUS=OFF
-DCMAKE_CUDA_COMPILER=$(which nvcc) -DPython3_EXECUTABLE=$(which python)`

## 6. 未竟事项（新会话的 TODO，按优先级）

### 6.1 [P0] push 分支并发 PR

```
git remote add fork https://github.com/<你的用户名>/FlashRT.git
git push fork feature/groot-n17-ele-fusion
# 网页: base=flashrt-project/FlashRT:main ← head=<fork>:feature/groot-n17-ele-fusion
# 描述 = docs/superpowers/pr-body-groot-n17-ele-fusion.md 全文（确认已含 §4.1 数字）
```

### 6.2 [P1] CUDA 13 环境下补 fp8 真 GEMM 验证（可选加分项）

- **背景**：PR-3 的 wiring 差分两腿共享 GEMM shim，与真 GEMM 无关，
  正确性已闭环。真 `fp8_descale_fp16` 在 CUDA 12.8 不可执行（§5 #8），
  **若新机器是 CUDA 13 + 5090**，跑 `verify` 脚本 + 下面的真 GEMM 探针
  即可补上：
  ```bash
  python -c "
  import sys; sys.path.insert(0, '.')
  import torch, flash_rt.flash_rt_kernels as fvk
  g = fvk.GemmRunner(); M,N,K = 1024,2048,2048
  X=(torch.randn(M,K,device='cuda:0')*0.5).to(torch.float8_e4m3fn)
  W=(torch.randn(N,K,device='cuda:0')*0.02).to(torch.float8_e4m3fn)
  out=torch.empty(M,N,device='cuda:0',dtype=torch.float16)
  s=torch.ones(2,device='cuda:0')
  g.fp8_descale_fp16(int(X.data_ptr()),int(W.data_ptr()),int(out.data_ptr()),M,N,K,int(s.data_ptr()),int(s.data_ptr()))
  torch.cuda.synchronize(); print('fp8_descale_fp16 OK:', out[0,:3].tolist())"
  ```
  通过 → PR 评论补一行 "confirmed on CUDA 13 + 5090"；不通过 → 上游
  descale 布局需修（提 issue，素材在 §5 #8）。
- **fp8 生产路径延迟数字**（README 16.6ms 那条的 PR 后 followup）需要
  N1.7 真权重（HF: nvidia/GR00T-N1.7）+ CUDA 13 才能测，不阻塞 PR。

### 6.3 [P2] 上游 issue 素材（两条，独立于本 PR 提）

1. **sm120 构建**：`fmha_fp8_causal_gqa_sm120.cu`（97KB）/ `tinyfp8_kernels_sm120.cu`（66KB）静态 smem 超 48KB ptxas 硬限，裸 5090 + CUDA 12.8 无法构建，需 dynamic-smem + `cudaFuncSetAttribute` 改造（sage2_attn_f8_raw.cu:117 是正确范例）；
2. **fp8_descale_fp16**：col-major NN 组合在 CUDA 12.8 cuBLASLt 全形状 NOT_SUPPORTED（torch._scaled_mm 的 TN 同机可用）；"SM120-safe" 路径在 12.8 上并不 safe，上游 5090 测试环境疑为 CUDA 13。

### 6.4 后续 PR 候选（与 maintainer 反馈对齐后）

- SigLIP ViT QKV 后处理融合（add_bias+全维 rope，LayerNorm 风格——
  需新 kernel 变体，PR body Deferred 段已提）；
- Thor 前端自身的 wiring；
- DiT fp8 分支内部融合（上游自带，仅在 maintainer 要求时动）。

## 7. 报告撰写素材（"撰写完整报告"用）

- **改动统计**：12 commits，20 文件，+1655/-97 行（`git diff --stat
  origin/main..HEAD`）；
- **kernel 账目**：新写 device 代码 1 个（qk_norm fp16 模板化）；主模块
  新增绑定 6 个；接线位点 7（sm89 pipeline）+2（DiT）+7（fp8 pipeline）；
- **验证账目**：差分测试文件
  `tests/test_groot_n17_sm89_ele_fusion.py`（863 行，13 个测试），加
  一次性脚本 `scripts/verify_groot_n17_fusion_rtx.sh`；
- **叙事线**（报告可按此写）：动机（Thor 有融合、RTX 没有，git 历史
  佐证）→ dedup-first 方法论 → strict bit-parity 契约 → 三层验证
  （kernel/stage/golden）→ review 抓到 residual 删除 bug 的故事
  （体现测试方法学价值）→ 双卡验证 → 两条上游发现。
- 所有 commit message 自含完整数据（GPU/命令/数字），可直接引用。

## 8. 给新机器上的 Claude Code 会话的指令模板

在新机器上 clone/pull 后，把下面这段（连同本文件路径）交给 Claude：

> 你在继续 FlashRT GROOT N1.7 融合 PR 的工作。请先读
> `docs/superpowers/HANDOVER.md`（本文档）了解全貌，再读
> `docs/superpowers/pr-body-groot-n17-ele-fusion.md`（PR 描述终稿）。
> 当前分支 `feature/groot-n17-ele-fusion` 应包含 12 个 commit（§2 列
> 表核对 `git log origin/main..HEAD`）。你的任务：
> 1. 若分支不完整（commit 缺失），先从 fork/备份恢复；
> 2. 按 `docs/setup-new-machine-rtx5090.md` 配环境（§5 踩坑表已并入）；
> 3. 跑 `PY=$(which python) ./scripts/verify_groot_n17_fusion_rtx.sh`，
>    期望三段 ALL PASS + e2e delta 为正（4090 基准 -7.8%，5090 基准
>    -10.4%，见 §4.1）；
> 4. 若是 CUDA 13 + 5090：跑 §6.2 的真 GEMM 探针补 PR-3 终验；
> 5. 帮我完成 §6.1 的 push + 发 PR（描述用 PR body 终稿，署名行已含）；
> 6. 后续按 maintainer 反馈迭代，注意 §3.3 的测试纪律与 §5 的已知坑。

---

*本文件由开发会话生成于 2026-09-30；ledger 全文在原开发机
`/home/zhangaoxiang/code/flashrt/.superpowers/sdd/2026-09-29-groot-n17-sm89-ele-fusion/progress.md`（若不可访问，本文件的 §3/§5 即其精编）。*
