# 复现与数据边界

## 本地依赖

已验证 Windows x64、Python 3.13.9、FastAPI、httpx、pydantic、pytest 与 Uvicorn。Demo 以源码检出运行，使用 `scripts/p10_2_fault_harness.py` 提供合成场景。Wheel 的包数据包含前端与合成草稿，单独安装 Wheel 尚未作为 Console Demo 的验收入口。另用本机已有 setuptools 环境离线构建 Wheel，验证包内只有 `public_fixtures.json`，未包含原文文件名。

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[agent,dev]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B scripts/run_console_demo.py
```

[旧 RC1 Windows/Python 3.13 固定清单](../requirements/agent-v01-win-py313.txt)供复查 RC1。V1 的适用范围见 `pyproject.toml`。在无训练集、无历史数据、无两份历史原文副本的源码集合上验证了 Demo。首装依赖由可信 Python 包索引或本地 wheelhouse 提供，Demo 运行不需访问外网。

## 快照离线测试

在快照根目录运行：

```powershell
$env:PYTHONPATH = 'src;.'
.\.venv\Scripts\python.exe -B -m pytest -q
```

本快照包含配置、BM25 检索以及 Mock Console 端到端的独立离线测试。评测摘要中的完整工作树回归、设备实验和真实模型复验需要受控历史证据。

## M1 与公开来源

M1 使用 `MARLIN_BASE_URL` 的通用客户端。Console 安全设备适配器固定到开发端口 8100。公开 Demo 使用 MockTransport 和全新历史库，数据结构、读数来源与正常运行的限制见 [架构](ARCHITECTURE.md) 与 [演示](DEMO_GUIDE.md)。保留数据、GPU 预算和实验归档位于受控运行目录，统一排除于公开材料。
