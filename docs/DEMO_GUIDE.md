# Demo 操作指南

## 模式选择

| 模式 | 设备来源 | 报告来源 | 启动 |
| --- | --- | --- | --- |
| 离线展示 | 合成数据，页面标记“设备未连接” | 固定 Mock 草稿和合成回放。相同输入可重复演示。 | `scripts/run_console_demo.py --port 8010` |
| 模拟器只读 | 既有 Marlin Simulator `127.0.0.1:8100` | 状态查询、合成回放，模型保持关闭 | 默认 Web 应用 |
| 可选真实模型 | 单独选择 Simulator 或经审查的 Fixture | 通过授权的 Base / LoRA 服务真实生成 | 单独部署云端及本地 Console |

## A–G 离线演示

1. 参考 [安装命令](../README.md#五分钟离线演示) 启动 Demo。打开 [诊断页](http://127.0.0.1:8010/console/diagnosis)。终端打印本次 Demo 的临时 SQLite 路径。
2. 查看 **设备总览**。示例状态可浏览，来源明确标为 Mock。展示 8100 的实时读数时使用下节的独立命令。
3. 保持 V2 开启，输入「打印外壁明显偏细，相邻走线出现规则空隙，材料仍持续流出。」。
4. 查看“疑似挤出不足 / 证据不足”、维护知识和依次列出的检查。Demo 仅为这一固定症状返回挤出不足草稿，其他故障文本使用保守的证据不足草稿。Base / LoRA 按钮用于演示选择和来源显示。
5. 点击 **Demo 回放 → Safety · 危险建议拦截**。核对拦截提示和安全提醒。合成的危险建议经实际发布管线过滤。
6. 打开 **诊断历史**，查看两个条目，打开详情并选择 **导出 Markdown**。
7. 展开 **技术详情**，查看 V2、Safety、来源与检索引用。讲解 RAG 与设备回执的区别。

[诊断截图](assets/console/v1-diagnosis.png) · [历史截图](assets/console/v1-history.png)

## 单独展示真实模拟器读数

先由操作人员启动自己的 Marlin Simulator，并确认 HTTP API 支持 `/ping`、`/state`、`/idle` 和限位组件读取。使用另一空闲端口，无需停止 Demo。以下在新 PowerShell 执行。

```powershell
$env:PRINTDIAG_ALLOW_REMOTE_INFERENCE = '0'
$env:PRINTDIAG_SIM_CONTROL_ENABLED = '0'
$env:PRINTDIAG_LIVE_GPU_BUDGET = '0'
Remove-Item Env:PRINTDIAG_LIVE_BUDGET_STATE, Env:PRINTDIAG_LIVE_AUDIT_DIR -ErrorAction SilentlyContinue
$env:PRINTDIAG_HISTORY_DB = Join-Path ([IO.Path]::GetTempPath()) "printdiag-simulator-$([guid]::NewGuid()).sqlite3"
Invoke-RestMethod 'http://127.0.0.1:8100/ping' -TimeoutSec 3
.\.venv\Scripts\python.exe -B -m uvicorn printdiag.web.app:app --host 127.0.0.1 --port 8011
```

打开 [设备总览](http://127.0.0.1:8011/console/overview)，或 GET `/api/device/status` 查看 `source`、`provenance`、`device_connected`。8100 可读时显示“模拟器在线”。8100 不可达时显示“设备离线”，清除不可用读数。 `online` 可表示合成读数可展示，`device_connected` 表示连接证据。上述两个启动器均禁用仿真控制。默认 Console 的回放 B/C 仍是明确标注的合成场景。

## 可选云端真实推理

此步骤需要另行授权、模型与算力。Demo 和离线测试不需要这一步。

1. 在隔离 GPU 环境提供固定版本的 Qwen3.5-4B 完整 Base 快照、经批准的 LoRA Adapter 与运行依赖。核对两片权重、Adapter 配置和权重的 SHA-256。两种模式用同一 Base 快照。用于真实推理的 Prompt 模板依赖归档的训练格式，由维护者提供经审核的材料；公开 Demo 自带独立合成模板。
2. 使用 `cloud_inference/requirements-cloud.txt` 审核依赖。设置 `PRINTDIAG_CLOUD_BASE`、`PRINTDIAG_CLOUD_ADAPTER`，在进程环境注入 `PRINTDIAG_CLOUD_TOKEN`，仅绑定云主机回环后运行 `bash cloud_inference/start_server.sh`。通过 `ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18008:127.0.0.1:8008 <approved-ssh-alias>` 建立隧道。
3. 先以只读接口核对 `/healthz`、`/v1/model-info`、`/v1/models`。获批请求范围后，新建独立 History、Budget、Audit 路径，使用 `scripts/start_printdiag_live.ps1` 的显式 `-Python`、`-StatePath`、`-Budget` 和 `-MaxConcurrency 1`。启动器的环境说明应与实际部署一致，密钥不进入命令历史或文档。
4. 仅执行授权样本，保留请求、预算和失败证据。历史 P10-9B 的两次预算已耗尽，演示不重置该记录。

公开资料仅展示合成 Demo 与真实联调的汇总结果。
