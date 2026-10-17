# `serve/server.py` 参数参考：config.json、args、环境变量

本文整理 `serve/server.py` 读取的三类配置，全部行号以当前 checkout 为准：

1. 运行配置（`--config` 指向的 `strata-<model>.json`，由 `setup.py` 写出）里的所有键；
2. 配置里 `"args"` 列表能出现的引擎命令行参数；
3. `serve/` 读取的所有环境变量。

## 优先级规则

- 同一项配置，**命令行参数 > config 键 > 内置常量**。例：`--host` 未给时用 `"host"`，都没有才是 `127.0.0.1`（`server.py:5657`）。
- config 里键缺失或为 `null` = 不注入任何东西，由引擎或 server 的默认值生效。
- 表中标 **opt-in** 的键默认关闭，写了才生效。
- `"args"` 是 `--engine strata` 的必填键；`exe`、`model`、`mmproj` 同为必填。相对路径一律按配置的 `"cwd"` 解析。
- 读取配置使用 `encoding="utf-8-sig"`，带 BOM 的文件可以。`runconfig.save` 写回前保留 `<name>.bak`。
- `runconfig.EDITABLE`（`runconfig.py:19-38`）只列出 Web Settings 页面可改的 16 项；其余键手工编辑，保存时原样保留。

## 一、config.json（`--config`）的键

### 1.1 启动与身份

| 名称 | 类型与取值范围 | 默认值 | 示例值 | 作用 |
|---|---|---|---|---|
| `exe` | 字符串，必填 | 无（必填） | `"engine/strata.exe"` | 引擎可执行文件；相对路径按 `cwd` 解析（`server.py:5711`） |
| `args` | 字符串数组，必填 | 无（必填） | `["--native","pack/full","--max-context","8192"]` | 传给引擎的参数列表，见第二节（`2162/2169`） |
| `model` | 字符串，必填 | 无（必填） | `"pack/full/model.gguf"` | 主模型文件，仅出现在 vision 编码器自己的 `--model`（`1845`） |
| `mmproj` | 字符串 | 无 | `"pack/full/mmproj.sve"` | 视觉投影文件；有 `vision` 段时必填 |
| `cwd` | 字符串 | `null`（进程当前目录） | `"D:/models/qwen3.8"` | 引擎与编码器的进程目录；相对 `exe`/`args` 路径的基准 |
| `log` | 字符串（追加打开） | `null`（无日志文件） | `"engine/engine.log"` | 引擎 stdout/stderr 落盘位置；也是 `tmp` 回退目录的基准（`2395`） |
| `tokenizer` | 字符串（目录） | `--tokenizer` 的默认 `ROOT/pack/full/tokenizer` | `"pack/full/tokenizer"` | 有值时覆盖 `--tokenizer`（`5662`）；缺 `vocab.json` 时 `--engine strata` 直接报错 |
| `model_name` | 字符串 | `"qwen3.8-flash-next"` | `"qwen3.8-flash-next"` | `/v1/models` 与 `/models` 列出的模型名（`5735`） |
| `aliases` | 字符串（逗号分隔）或数组 | `[]` | `["qwen3.8-flash", "next"]` | 模型应答的其它名字，同样被列出（`5742`） |
| `host` | 字符串 | `127.0.0.1`（`--host` 与键都缺时） | `"0.0.0.0"` | 监听地址；`0.0.0.0` 表示局域网可访问，此时应配 API key |
| `port`（无此键） | — | `8095` | — | 端口只能来自 `--port`，config 里没有对应键 |
| `api_key` | 字符串（逗号分隔多 key）或数组 | `""`（不校验） | `["k1", "k2"]` | `/v1/*` 的 Bearer / `x-api-key` 校验（`5754`）；`--api-key` 与 `STRATA_API_KEY` 优先 |
| `cors_origins` | 字符串或 `scheme://host[:port]` 数组 | `[]` | `["https://chat.example.com"]` | `/v1/*` 的 CORS 白名单；允许 `"*"`（`5761`） |
| `trusted_origins` | 同上，不允许 `*` | `[]` | `["https://app.example.com"]` | 需要精确来源的跨站判断（`5762`） |
| `allowed_hosts` | 字符串或主机名数组，支持 `*`、`.example.com` | `[]` | `["mybox.local", ".lan.example.com"]` | `Host` 头白名单；与 `STRATA_ALLOWED_HOSTS` 合并（`5764`） |
| `backend` | `"hip"` 或其它非空值 | `null`（按 NVIDIA/CUDA 处理） | `"hip"` | AMD：读数走 sysfs（`#301`），设备编号改用 HIP 序号（`2341/5824`） |

### 1.2 GPU 与分层

| 名称 | 类型与取值范围 | 默认值 | 示例值 | 作用 |
|---|---|---|---|---|
| `gpu` | int、数组或逗号串；编号同 `nvidia-smi` | `[]`（不指定） | `[0, 2]` 或 `"0,2"` | 单卡或分层切分的卡列表（`2083`）；同时决定 Monitor 读哪张卡（`5822`） |
| `hip_ordinal` | 整数（字符串亦可） | `null` | `1` | AMD 单卡配置下覆盖 `gpu`：`strata-device --list-devices` 给出的 HIP 序号（`#325`，`2267`） |
| `layer_split` | `"auto"`／字符串／整数数组 | `"auto"` | `"24,36,42"` | **每张后续 GPU 的第一层**（不是每卡层数），升序、首值 ≥2；仅当 ≥2 卡且 `args` 无 `--peer-device` 时注入 `--layer-split`（`2131`） |
| `gpu_order` | `"as_given"` 或其它 | 按速度排序（快卡放最后） | `"as_given"` | 阻止自动重排卡序（`2336`）；手写了 `layer_split` 时同样不重排 |
| `split_skip_if_fits` | 真值 | 关（opt-in） | `true` | 自动切分时，若首卡装得下全部 profiled expert 就只用首卡（`2185`） |
| `vram_elastic` | `true` | 关（opt-in） | `true` | expert cache 分段，`POST /v1/vram` 可让出/收回显存（`#533`，`2189`） |
| `vram_segment_mib` | 正整数 | 引擎默认 `512` | `1024` | 上项的分段大小（`2191`） |
| `parallel` | 整数 ≥2（`PARALLEL_MAX=8`） | `null`（一次一个请求） | `4` | 只注入 `--batch N`；`args` 里已有 `--batch`/`--slots` 时整个忽略；>8 会提示（`2217`）。分组不在这里管：引擎自己按 auto 规则选（`generate.cpp:4022`，0.1.41 起的默认），跑的是几组由 `INFO batch_groups=G` 说回来，server 读回来用（`710-723`）；要改就在 `args` 里写 `--batch-groups` |

### 1.3 超时、卸载与恢复

| 名称 | 类型与取值范围 | 默认值 | 示例值 | 作用 |
|---|---|---|---|---|
| `engine_silence_s` | 数字 ≥0（0=一直等） | `ENGINE_SILENCE_S=300.0` | `600` | 引擎在请求期间静默多久即被结束（`#481`，`2118`） |
| `idle_unload_s` | 数字 ≥0（0/空=永不） | `0` | `900` | 无请求这么多秒后卸载模型（`5779`） |
| `min_free_vram_mib` | 整数 ≥0 | `0`（总是加载） | `24576` | 空闲显存不足则回 503；等待上限 15 s（`5781`） |
| `before_load` | 字符串或数组 | `null` | `["python","-m","other","unload"]` | 重新加载模型前执行的命令，子进程超时 120 s（`5782`） |
| `slot_save_path` | 目录字符串（创建为 0700） | `null` | `"engine/slots"` | 打开 `POST /slots/0?action=save\|restore`；会话文件约 1.2 GB / 63K token（`5785`） |
| `lazy_load` | `true` | 关（opt-in） | `true` | 启动不加载模型，首个请求才加载（`5687`） |
| `tool_call_recovery` | 布尔 | `false`（opt-in） | `true` | 工具调用被截断时的恢复（`5797`） |
| `reasoning_loop_recovery` | `false`／`"stop"`／`"recover"`（`true` 视作 `"recover"`） | `false`（opt-in） | `"stop"` | 推理循环卡住时停止或恢复（`#728`，`5815`） |
| `reasoning_close_retry` | `true` | 关（opt-in） | `true` | 回复停在 `<think>` 内时补一次收尾（`#1053`，`5738`） |
| `codex_thread_titles` | `true` | 关（opt-in） | `true` | Codex 的标题轮次不经引擎作答（`#923`，`5739`） |
| `codex_compaction_cache` | `true` | 关（opt-in） | `true` | compaction 按会话自身工具渲染（`#924`，`5740`） |
| `repeat_stop_tokens` | 整数 ≥0（0=关） | `REPEAT_STOP_TOKENS=256` | `64` | 同一 token 连续重复这么多次即按 `length` 结束（`#606`，`5801`） |

### 1.4 采样与推理

| 名称 | 类型与取值范围 | 默认值 | 示例值 | 作用 |
|---|---|---|---|---|
| `sampling.temperature` | 数字 ≥0 | 无块=贪心 | `0.7` | 请求未带 temperature 时的默认（`5553`） |
| `sampling.top_p` | `0 < x ≤ 1` | 未设 | `0.95` | 同上 |
| `sampling.top_k` | 整数 `1..64` | 未设 | `40` | 同上 |
| `sampling.min_p` | `0 ≤ x ≤ 1` | 未设 | `0.05` | 同上 |
| `sampling.presence_penalty` | 数字 ≥0 | 未设 | `1.2` | 同上 |
| `sampling.frequency_penalty` | 数字 ≥0 | 未设 | `1.5` | 同上 |
| `sampling.repetition_penalty` | 数字 >0（1=关） | 未设 | `1.1` | 同上 |
| `sampling.penalty_last_n` | 整数 ≥0 | 未设 | `64` | 同上 |
| `sampling.seed` | 正整数 | 未设 | `42` | 同上 |
| `sampling.experimental_speed_projection` | 布尔 | 未设 | `true` | 引擎带该向量时的默认 |
| `reasoning_budget_tokens` | 整数 ≥0（0=不设预算） | 未设 | `8192` | 每个请求的思考 token 上限（`5806`） |
| `reasoning_effort` | `none`/`low`/`medium`/`high` | 未设 | `"medium"` | 仅出现在 `<stem>.shared-settings.json`（`SHARED_KEYS`，`5508`） |
| `max_tokens`（shared） | 正整数 | 未设 | `4096` | 同上；`sampling` 块里没有这个键 |
| `fit_max_tokens` | `true` | 关（opt-in） | `true` | `max_tokens` 装不下时截短而不是回 400（`5737`） |
| `anthropic_thinking` | `"model"`／`"on_request"` | `"model"` | `"on_request"` | Anthropic 请求未要求思考时是否思考（`5793`） |
| `effort_position` | `"start"`／`"end"` | `"start"` | `"end"` | `"end"` 时注入 `--tail-role-token <id>`，需引擎 0.1.39+（`2095`） |
| `open_browser` | 布尔 | 未设（`--open` 决定） | `false` | `false` 压过 `--open`（`5865`） |
| `api_monitor` | `true` | 关（opt-in） | `true` | `/api-monitor` 内存保留最近 100 次请求（`5770`） |

### 1.5 vision 段（`cfg["vision"]`，`5688-5695`）

| 名称 | 类型与取值范围 | 默认值 | 示例值 | 作用 |
|---|---|---|---|---|
| `vision.exe` | 字符串（按 `cwd` 解析） | 无（必填） | `"engine/strata-vision.exe"` | 图像编码器可执行文件 |
| `vision.model` / `vision.mmproj` | 字符串（按 `cwd` 解析） | 无（必填） | `"pack/full/mmproj.sve"` | 编码器读的模型/投影文件 |
| `vision.gpu` | 整数 | 未设 | `2` | 编码器所在卡；有它且 `args` 无 `--vram-reserve-mib` 时自动补 `700`（`2174`） |
| `vision.cuda_device` | 整数（编号同 `nvidia-smi`） | 未设（沿用引擎环境） | `1` | 单独指定编码器的卡（`#408`，`2417`）；`backend=="hip"` 时改写 `HIP_VISIBLE_DEVICES` |
| `vision.threads` | 整数 | 未设 | `8` | 传给编码器的 `--threads`（`1848`） |
| `vision.max_tokens` / `vision.min_tokens` | 整数 | 未设 | `1024` / `4` | 传给编码器的 `--max-tokens` / `--min-tokens`（`#767`，`1850/1852`） |

### 1.6 子进程环境

| 名称 | 类型与取值范围 | 默认值 | 示例值 | 作用 |
|---|---|---|---|---|
| `env` | 对象（值转成字符串） | `{}` | `{"HIPBLASLT_TUNE_TABLE":"..."}` | 追加到引擎环境（AMD 的 GEMM 调优表即在此）（`2361`） |
| `lib_dirs` | 目录数组（仅保留存在的） | `[]` | `["engine/lib"]` | 前置到 `PATH`（Windows）或 `LD_LIBRARY_PATH`（`2363`） |

### 1.7 MCP 相关键

| 名称 | 类型与取值范围 | 默认值 | 示例值 | 作用 |
|---|---|---|---|---|
| `mcp` | 对象，仅 4 个已知键 | 见下 | `{"timeout_s": 30}` | 工具调用参数（`mcp.py:631`） |
| `mcp.timeout_s` | 数字 >0 | `60.0` | `30` | 单次工具调用超时 |
| `mcp.max_result_chars` | 正整数 | `20000` | `40000` | 结果截短长度 |
| `mcp.max_rounds` | 正整数 | `8` | `4` | 一轮对话最多几轮工具往返 |
| `mcp.start_timeout_s` | 数字 >0 | `120.0` | `60` | 服务器启动超时 |
| `mcp_servers` / `mcpServers` | `{"name": {...}}` | `{}` | `{"time": {"command": "uvx", "args": ["mcp-time"]}}` | 工具服务器；`"type":"sse"` 被拒；`"disabled": true` 跳过；`--mcp-config` 文件里的同名条目胜出（`mcp.py:648`） |
| 单条 server 的键 | `command`+`args`(数组)+`env`(对象)+`cwd`，或 `url`+`headers`(对象) | `cwd` 未设 | `{"url":"http://127.0.0.1:9000/sse"}` | 至少要有 `command` 或 `url`（`mcp.py:601`） |

### 1.8 Web Settings 可编辑的 16 项

`runconfig.EDITABLE`（`runconfig.py:19-38`）：`sampling.temperature`、`sampling.top_p`、`sampling.top_k`、`sampling.min_p`、`reasoning_budget_tokens`、`fit_max_tokens`、`anthropic_thinking`、`effort_position`、`aliases`、`idle_unload_s`、`lazy_load`、`engine_silence_s`、`api_monitor`、`open_browser`、`vram_reserve_mib`（映射到 `args` 里的 `--vram-reserve-mib`，引擎默认 700），以及 `sampling.experimental_speed_projection` 一类采样项。保存时其余键原样保留。

## 二、`"args"` 里能出现的引擎参数

`args` 是原样传给 `strata --serve` 的参数列表。server 只在三种情况下改动它：有 `vision` 段但缺 `--vision` 时补 `--vision`（连带 `--vram-reserve-mib 700`）；分层切分时补 `--layer-split`；`parallel` / `vram_elastic` / `expert_profile_save` / `effort_position` 各自补对应 flag。已在 `args` 里的同名 flag 永不被覆盖。

### 2.1 server 自己会读出的

| 参数 | 取值 | 默认值 | 示例值 | 作用 |
|---|---|---|---|---|
| `--native` / `--pack` | 模型路径 | `model_path` 回退 `"pack/full"`（`607`） | `--native pack/full` | 模型位置；也决定启动时那句 experts 加载描述 |
| `--max-context` | 整数 | server 侧 `4096`（`611`） | `--max-context 8192` | 上下文窗口；决定请求的 `max_tokens` 上限 |
| `--batch` / `--slots` | 整数（两者同义） | `0`（一次一个请求） | `--batch 4` | 批处理行数；引擎实开数由 `INFO batch_slots` 报（`705`） |
| `--batch-groups` | `"default"`／`"auto"`／整数 | `"default"`（`710`）＝不写，由引擎选 | `--batch-groups 2` | 分层切分时把 slot 分成 G 组流水线过卡；不写＝引擎 auto：每组一 stage，取整除 `--batch` 的最大 d（`generate.cpp:4022`，0.1.41 起），实跑数由 `INFO batch_groups` 报 |
| `--batch-mtp` | 无值 | 关 | `--batch-mtp` | 批处理走 MTP 波次；有它时 `parallel>8` 不再提示 |
| `--peer-device` | 整数 | 未设 | `--peer-device 1` | 后卡作 expert 缓存层；有它即不做 `--layer-split`（`#665`） |
| `--tail-role-token` | token id | 未设 | `--tail-role-token 248045` | 由 `effort_position:"end"` 自动写入 |
| `--expert-profile` | 文件 | 未设 | `--expert-profile prof.bin` | 自适应层读的 profile；与 learned profile 形状一致时被替换 |
| `--expert-profile-save` / `--expert-profile-save-every` | 文件 / 分钟 | 每 10 分钟（0=仅 QUIT） | `--expert-profile-save-every 30` | 由 `expert_profile_save` 注入（`2243/2247`） |
| `--resident-budget-gib` | 浮点 GiB | 未设 | `--resident-budget-gib 24` | RAM 预算；大于本机空闲/总内存时告警（`#1080`，`416/5260`） |
| `--resident-experts` / `--mmap-experts` | 无值 | 关 | `--mmap-experts` | experts 的加载方式（`423/426`） |
| `--eos-ids` | id 列表 | `{248044, 248046}`（`562`） | `--eos-ids 248044,248046` | 结束 token |
| `--gpu` | 编号 | 由 `gpu` 键注入 | `--gpu` | 编码器的 `--gpu` 由 `vision.gpu` 决定（`1846`） |
| `--threads` | 整数 | 未设 | `--threads 8` | 仅编码器（`1848`） |
| `--max-tokens` / `--min-tokens` | 整数 | 未设 | `--max-tokens 1024` | 仅编码器（`1850/1852`） |

### 2.2 引擎侧其余常用参数（原样透传）

| 参数 | 取值 | 默认值 | 示例值 | 作用 |
|---|---|---|---|---|
| `--serve` | 无值 | 由 server 追加 | `--serve` | 常驻模式（`stdin` 读 `GEN`） |
| `--temperature` / `--top-k` / `--top-p` / `--seed` | 数字/整数 | 贪心 | `--temperature 0.7` | 引擎侧采样默认；请求字段总是优先 |
| `--max-new` | 整数 | 引擎默认 | `--max-new 512` | 单请求生成上限 |
| `--vram-reserve-mib` / `--vram-reserve-later-mib` | 整数 MiB | `700` | `--vram-reserve-mib 1200` | 为其它程序留出的显存 |
| `--vram-elastic` / `--vram-segment-mib` | 无值 / 整数 | 关 / `512` | `--vram-elastic` | 分段显存（`#533`） |
| `--layer-split` | `"auto"`／升序层号 | `"auto"` | `--layer-split 24,36,42` | 每张后续卡的首层 |
| `--split-skip-if-fits` | 无值 | 关 | — | 首卡装得下即不切分 |
| `--vision` | 无值 | 关 | `--vision` | 打开图像路径 |
| `--prefill` / `--prefill-until` | 名字/整数 | 引擎默认 | `--prefill auto` | 预填充路径与阈值 |
| `--conversation-cache-mib` / `-slots` / `-min-free-mib` | 整数 | 引擎默认 | `--conversation-cache-mib 512` | 会话 checkpoint 缓存 |
| `--prompt-cache` / `--prompt-cache-root` | 无值/路径 | 关 | — | 提示缓存 |
| `--expert-cache` / `--expert-cache-device` | 整数/编号 | 引擎默认 | `--expert-cache 4096` | expert 缓存大小与所在卡 |
| `--kv` / `--kv-grow` / `--kv-resident` | 整数/无值 | 引擎默认 | — | KV 池 |
| `--rope-scaling` / `--yarn-attn-factor` 等 | 字符串/浮点 | 引擎默认 | `--rope-scaling yarn` | 长上下文外推 |
| `--greedy` | 无值 | — | `--greedy` | 强制贪心 |

## 三、环境变量

`_timeout_env`（`227`）对超时类变量：值 ≤0 时返回 `None`，即“一直等/关闭”，不是 0 秒。

| 变量 | 默认值 | 示例值 | 作用 |
|---|---|---|---|
| `STRATA_API_KEY` | `""`（`5631`） | `k1,k2` | `--api-key` 的默认值；已设但为空串会直接退出（`5747`） |
| `STRATA_ALLOWED_HOSTS` | `""`（`5764`） | `mybox.local,.lan.example.com` | 追加到 `allowed_hosts` |
| `STRATA_MAX_BODY_MIB` | `"0"` → `CHUNKED_BODY_MAX=256 MiB`（`4187`） | `512` | 请求体上限；超出先回 413 |
| `STRATA_HTTP_BACKLOG` | `"256"`（下限 5，`5220`） | `512` | `request_queue_size` |
| `STRATA_DEBUG` | 未设（`3815`） | `1` | 每请求一行诊断 + 原始 token 转储（`3733`） |
| `STRATA_REQUEST_LINES` | 未设（`636`） | `1` | 把引擎日志行同时打到 stdout |
| `STRATA_BATCH_TRACE` | 未设（`554`） | `1` | stderr 上打 `[batch-trace]` |
| `STRATA_BATCH_MTP` | `"0"`；仅 `""`/`"0"` 起作用（`2216`） | `1` | 抑制 `parallel>PARALLEL_MAX(8)` 的告警 |
| `STRATA_PARALLEL_SOLO` | 仅 `"0"` 起作用（`1166`） | `0` | 批处理时不把 slot cache 留在 slot 里 |
| `STRATA_KEEP_EMPTY_TURNS` | 仅 `"1"` 起作用（`frontend.py:60`） | `1` | 空轮次照原样渲染（`#843`） |
| `STRATA_TOKENIZER` | 未设（仅 `test_detok.py:21`） | `pack/full/tokenizer` | 测试选 tokenizer 目录 |
| `STRATA_ENGINE_STALL_S` | `90.0`（`239`） | `0`（关） | 静默且 CPU/IO 也无动作即判冻结；需 psutil |
| `STRATA_ENGINE_READY_S` | `900.0`（`251`） | `1800` | 等引擎 `READY` 的上限；0=一直等 |
| `STRATA_VISION_READY_S` | `300.0`（`249`） | `600` | 等编码器 `READY` |
| `STRATA_VISION_ENCODE_S` | `300.0`（`250`） | `600` | 单次编码的等待上限 |
| `CUDA_DEVICE_ORDER` | 由 server 设为 `PCI_BUS_ID`（`2357`） | — | 让卡编号与 `nvidia-smi` 一致 |
| `CUDA_VISIBLE_DEVICES` | 由 `gpu`/`ordered_gpus` 写入（`2359`） | `0,2` | 引擎可见的 CUDA 卡 |
| `HIP_VISIBLE_DEVICES` | 由 `hip_ordinal`/`gpu` 写入（`2356`） | `1` | AMD 侧同上 |
| `HIP_PATH` / `HIP_DEVICE_LIB_PATH` / `LLVM_PATH` | 指向不存在的目录时被剔除（`2380`） | — | 避免 HIP 运行时在打印任何一行前崩溃（`#654`） |
| `TEMP` / `TMP` / `TMPDIR` | 不可写时换成 `<log 同级>/tmp`（`2395`） | — | comgr JIT 需要可写临时目录 |
| `DISPLAY` / `WAYLAND_DISPLAY` | 未设（`5282`） | — | 判定 Linux 桌面会话，触发显存提示 |
| `SystemDrive` | `C:`（`1830`） | — | 编码器临时目录的回退位置 |
| `ProgramFiles` | `C:\Program Files`（`telemetry.py:34`） | — | 找 `nvml.dll` 的回退位置 |

注：`STRATA_FAIL_ONCE` 不是环境变量，它只是 `test_server.py:3644/3709` 里的 fixture `FAKE_STRATA_FAIL_ONCE`。

## 四、其它内置常量（无对应 config 键）

| 常量 | 值 | 位置 | 作用 |
|---|---|---|---|
| `SESSION_WAIT_MAX_S` | `3600` | `210` | 引擎 `SWAIT` 可宣布的单步上限 |
| `PP_CHUNK_MAX` / `PP_FLOOR_TOK_S` / `PP_SLACK` | `32768` / `50.0` / `3.0` | `214-216` | 读 prompt 分块时的等待估算 |
| `STALL_CPU_EPS_S` / `STALL_IO_EPS_B` | `0.5` / `1<<20` | `240-241` | 判定“冻结”而非“慢”的门限 |
| `PARALLEL_MAX` | `8` | `2201` | 批处理窗口最多 8 行（`kVerifyMaxT`） |
| `CHUNKED_BODY_MAX` | `256<<20` | `4179` | 见 `STRATA_MAX_BODY_MIB` |
| `EOS_IDS` | `{248044, 248046}` | `562` | `<\|endoftext\|>`、`<\|im_end\|>` |
