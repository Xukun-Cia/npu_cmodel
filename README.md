# npu_cmodel

NPU 行为级 **C/C++ 仿真模型**：在主机上模拟 SRAM、DMA、CPU、RVV、TPU 等路径，并附带 SILU、MatMul、GELU、Conv3D 等算子与可执行用例，用于验证数据类型与维度下的功能与统计信息。

## 依赖

- CMake ≥ 3.10
- 支持 **C++17** 的编译器（GCC/Clang 等）

## 构建

```bash
mkdir -p build && cd build
cmake ..
make -j
```

可执行文件：`build/npu_top`（或当前生成目录下的 `npu_top`）。

## 运行

```bash
./npu_top -t <用例> [选项]
```

| 用例 | 说明 |
|------|------|
| `t000` | SILU（FP32） |
| `t001` | MatMul（FP16 / BF16 / FP8） |
| `t002` | GELU（FP32） |
| `t003` | Conv3D（FP16 / BF16），默认场景含 ViT Patch Embed 相关维度 |
| `t004` | Conv3D patched（BF16 输入等），从 `data/` 目录加载权重与数据 |

常用示例：

```bash
./npu_top -t t000
./npu_top -t t001 --M 512 --N 128 --K 80 --type fp16
./npu_top -t t003 --type bf16
./npu_top -t t004 --data_dir ../data
```

未带 `-t <用例>` 或参数不合法时，程序会打印完整用法说明。

## 数据文件

`t004` 默认从 `../data`（相对 `build` 运行时）读取 BF16 等文本数据；请将 `data/` 置于合适路径或通过 `--data_dir` 指定。

## 仓库说明

- `operator/`：算子实现  
- `usecase/`：各测试入口与参数  
- `data/`：部分用例所需输入/参考输出（体积较大，克隆可能较慢）

## 许可证

（若需开源许可证，请在本仓库补充 `LICENSE` 文件。）
