# 新机器环境配置手册（RTX 5090 / sm120，PR 验证用）

适用于在新机器上从零配置 FlashRT 开源仓库并运行
`scripts/verify_groot_n17_fusion_rtx.sh`（GROOT N1.7 融合 PR 的
一次性验证）。每一步都标注了本机（4090 开发机）实际踩过或规避的坑。

上游总安装文档：`docs/INSTALL.md`（Docker / 原生两条路径）。本手册是
其中"原生 Linux"路径的 5090 具体化 + PR 验证专项补充。

---

## 1. 硬件 / 驱动前置

| 项 | 要求 | 检查命令 |
|---|---|---|
| GPU | RTX 5090 (sm_120) | `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader` 应输出 `RTX 5090, 12.0` |
| 驱动 | **550+**（5090 必须；对应 CUDA 12.8 runtime） | `nvidia-smi | head -3` |
| 磁盘 | ~15 GB（源码 + cutlass + torch + build 产物） | — |

5090 上 CUDA Toolkit 需 **12.8+**（sm_120 的 nvcc 支持）。
没有 sudo 也能装（见 §2 的用户级 toolkit 安装）。

## 2. CUDA Toolkit（12.8+，确认 nvcc 可用）

```bash
nvcc --version    # 理想情况系统已有
```

没有的话（免 sudo，用户级安装到 ~/local）：

```bash
CUDA_VER=12.8.1   # 或仓库镜像可用的最新 12.8.x
wget https://developer.download.nvidia.com/compute/cuda/${CUDA_VER}/local_installers/cuda_${CUDA_VER}_560.35.05_linux.run
sh cuda_${CUDA_VER}_560.35.05_linux.run --toolkit --silent --override \
    --defaultroot=$HOME/local/cuda
# 写入 ~/.bashrc：
export PATH=$HOME/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/cuda/lib64:$LD_LIBRARY_PATH
```

> 本机坑（4090）：系统 nvcc 在 `/usr/local/cuda/bin` 但不在 PATH，
> cmake 报 "The CUDA compiler identification is unknown"。环境变量
> `CUDACXX=/usr/local/cuda/bin/nvcc` 或 `-DCMAKE_CUDA_COMPILER=...`
> 二选一显式指定即可。验证脚本会自动透传 `CUDACXX`。

## 3. Python 环境（单独 env，版本自选但全程唯一）

```bash
conda create -n flashrt python=3.11 -y    # 或 python3.12 -m venv .venv
conda activate flashrt
```

**核心坑**：构建时运行 cmake 的 python 与之后 import flash_rt 的
python **必须是同一个解释器**（.so 的 ABI tag 绑定解释器小版本，
如 `flash_rt_kernels.cpython-311.so`）。混用系统 python 构建 +
conda python 运行是 #1 失败模式。

```bash
# torch（按 CUDA 版本选 wheel；5090 用 cu128）
pip install torch --index-url https://download.pytorch.org/whl/cu128
# 仓库本体（editable，必须 -e：.so 构建后直接落在 flash_rt/ 源码树内）
cd /path/to/FlashRT
pip install -e ".[torch]"
# 本仓库构建/测试还需要、但不在依赖声明里的两个包（本机实测缺过）：
pip install pybind11 pytest
```

> 本机坑（4090）：`pybind11` 与 `pytest` 不在 pyproject 依赖里，
> CMake 在 `execute_process(python -m pybind11 --cmakedir)` 处直接
> 失败；跑差分测试的机器还需要 pytest。一次装齐：`pip install
> pybind11 pytest`。

## 4. CUTLASS（v4.4.2，仓库不打包）

```bash
git clone --depth 1 --branch v4.4.2 \
    https://github.com/NVIDIA/cutlass.git third_party/cutlass
```

> 国内网络可加 `-c http.proxy=...` 或用镜像；仅需 headers，depth 1
> 约 300MB。clone 完成后 `ls third_party/cutlass/include/cutlass/cutlass.h`
> 必须存在（CMake 以此做存在性检查）。
> 已有其他机器的 clone 时可直接 `rsync`/复制该目录，无需重新下载。

## 5. 获取代码（含 PR 分支）

```bash
# 直接用 fork 后的仓库（见 §8 发 PR 一节）：
git clone https://github.com/<你的用户名>/FlashRT.git
cd FlashRT
git checkout feature/groot-n17-ele-fusion
# （cutlass 在 clone 之后、cmake 之前放入 third_party/）
```

## 6. 构建 + 全量验证（一条脚本）

```bash
PY=$(command -v python)   # conda env 激活后的 python 的绝对路径
./scripts/verify_groot_n17_fusion_rtx.sh
```

脚本内部依次执行（也可手动分步重跑）：
1. **构建**：`cmake -B build -S . -DGPU_ARCH=120 -DFA2_ARCH_NATIVE_ONLY=ON ...`
   （GPU_ARCH 从 nvidia-smi 自动探测；5090 会得到 120）
2. **差分正确性**：`tests/test_groot_n17_sm89_ele_fusion.py` ——
   全部融合 bit-equal（torch.equal）断言 + golden 冻结链。
   **任何一行不是 PASS / ALL PASS 都是阻断项**。
3. **仓库测试**：groot_n17 子集，预期 108 passed / 少量 skip；
   若出现 FAIL，先对照 4090 基线记录区分"环境差异"与"真回归"。
4. **延迟基准**：两个 bench，与 4090 参考对照（见下）。

### 结果对照表（4090 基线，2026-09-29）

| 指标 | 4090 D 基线 | 5090 上看什么 |
|---|---|---|
| 差分测试 | ALL PASS（12 tests） | 必须 ALL PASS（bit-equal 是架构无关断言） |
| 仓库 groot_n17 套件 | 108 passed / 10 failed（Thor 专属，已在脚本中排除）/ 113 skipped | passed 数应一致；skips 可能因 fixture 存在与否不同 |
| qk-norm+RoPE pair（S=257/1024/4096） | 1.3–3.5× | fused 应稳定快于 legacy（幅度可不同） |
| LLM stage | -3.3% | delta 为正即符合预期 |
| **e2e（Se=768）** | **-7.8%**（27.66→25.51 ms） | **delta 应在 -4% ~ -8% 区间**；若 DiT 份额反向（merged GEMM 在 sm120 调度劣化），e2e 会退到 -4% 左右——此时按 PR 描述的 opt-in 键做架构门控即可，精度不受影响 |

绝对 ms 在 5090 上必然不同（更快），**对照的是 delta（相对值）**。

### 已知可排除的失败模式

| 症状 | 原因 | 处置 |
|---|---|---|
| `CMake 3.24 or higher is required` | 系统 cmake 3.22 | `pip install cmake` 后把 pip bin 目录放 PATH 前面 |
| `The CUDA compiler identification is unknown` | nvcc 不在 PATH | `export CUDACXX=$(which nvcc)` 再跑脚本 |
| `CUTLASS headers not found` | §4 没做 | 按 CMake 报错提示 clone v4.4.2 |
| `ModuleNotFoundError: flash_rt_kernels` | pip 没 `-e` 安装 / python 与构建时不一致 | 同一解释器 `pip install -e ".[torch]"` |
| pytest 收集错误（fp4 相关 4 个文件） | sm120 默认构建里 fp4 模块名不同 | 脚本已 `--ignore` 排除，无需处理 |
| `test_groot_n17_thor_fp4_kernels.py` 失败 | Thor sm110 专属 kernel，非 Thor 构建必失败 | 脚本已排除 |

## 7. 全新 sm120 构建需额外留意的两件事

1. **NVFP4 模块（`flash_rt_fp4`）**：4090 (sm89) 构建不含、我们的 PR
   也不触碰它；但 sm120 默认构建会编它（CMake 的
   `ENABLE_NVFP4` 组）。若新机器编译 NVFP4 相关 TU 失败，可加
   `-DENABLE_NVFP4=OFF` 重试——不影响本 PR 验证（`flash_rt_kernels`
   目标独立）。
2. **构建时长**：首次 sm120 全量构建比 4090 略长（多了 fp4 组）；
   本机用 `-j6`（多用户机器过载约束），独占机器可 `-j$(nproc)`。

## 8. 发 PR（fork 流程，与验证机无关）

```bash
# GitHub 网页: fork flashrt-project/FlashRT 到你的账号
git remote add fork https://github.com/<你的用户名>/FlashRT.git
git push fork feature/groot-n17-ele-fusion
# 网页发 PR: base = flashrt-project/FlashRT:main
#           head = <你的fork>:feature/groot-n17-ele-fusion
# PR 描述 = .superpowers/sdd/.../pr-body-draft.md 全文粘贴
```

PR 合并前若 5090 验证有任何与本手册 §6 对照表不符的项，
在 PR 里回复结果，不要合并。
