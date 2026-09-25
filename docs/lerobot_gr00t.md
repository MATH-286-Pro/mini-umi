# UMI LeRobot 数据与 GR00T 接入

## 存储格式

LeRobot v2.1 和 v3.0 使用相同的位姿字段：

- `observation.state.robotN_eef_tf`：float32，每帧 `[4,4]`，由 Zarr 的 pos + axis-angle 合成。
- 方向为 `tf_world_eef`，使用列向量：`p_world = tf_world_eef @ p_eef`。平移单位为米，最后一行是 `[0,0,0,1]`。
- 不再保存该末端的独立 pos、rotation 表示或 xyz+rotvec pose 别名。夹爪等其他状态及视频保留。

转换器固定写入 shape `[1]` 的零值 action 占位。它不是监督目标；训练时由未来状态生成动作。
UMI 加载器始终忽略磁盘 action 列，直接使用未来状态，不再需要动作来源或占位标记。

## 导出

```bash
uv run python -m scripts.convert_umi_zarr_to_lerobot_v21 \
  /path/to/dataset.zarr.zip /path/to/umi_lerobot_v21 \
  --repo-id local/umi_state_v21 --fps 60

uv run python -m scripts.convert_umi_zarr_to_lerobot_v3 \
  /path/to/dataset.zarr.zip /path/to/umi_lerobot_v3 \
  --repo-id local/umi_state_v3 --fps 60
```

两个版本必须使用不同输出目录。旧数据需要重新转换才能更新字段。
不读取或导出 Zarr 中的 action，不提供 `--action-source` 选项。观测位姿存 tf。

## UMI 加载与训练

两个版本共用 tf 加载、时间采样、相对变换与模型表示编码逻辑。
例如 action 字段可配置为：

```yaml
basic:
  horizon: 16
  latency_steps: 0
  down_sample_steps: 3
details:
  - source: robot0_eef_tf
    type: pos3d_xyz
    relative_to: robot0_eef_at_obs_end
  - source: robot0_eef_tf
    type: rot6d_col
    relative_to: robot0_eef_at_obs_end
  - source: robot0_gripper_width
    type: width1d
```

已有 pos + axis-angle 数据仍可读取：加载时先转为内存中的 tf。
直接保存 tf 后，加载时读取并校验矩阵；每次采样仍执行相同的相对变换与表示编码。
6D 编码提取矩阵的前两行或前两列；四元数、轴角编码使用数值转换。
因此存储改为 tf 不会增加采样阶段的表示转换步骤，但读取体积和初始化耗时可能改变。
每帧每机器人未压缩 float32 位姿从 6 个数（24 字节）变为 16 个数（64 字节），实际 Parquet 大小还受压缩影响。

## GR00T 接入限制

新导出不再生成依赖 xyz+rotvec 列的 `meta/modality.json` 和 `meta/gr00t_config.py`。
旧 GR00T 单列 EEF reader/config 不能直接将 `[4,4]` 当成 `XYZ_ROTVEC`。
要使用新数据，需要在 GR00T 的加载边界把 tf 转为其所需的位姿表示，并使统计路径采用相同转换。
该外部适配尚未在本仓库实现；v2.1 的文件布局本身不保证位姿字段兼容。

## 验证

```bash
uv run python -m unittest discover -s tests -p 'test_converter_state_export.py'
uv run python -m unittest discover -s tests -p 'test_dataloader_lerobot.py'
```

转换测试实际写出并读回两种版本的 Parquet，验证 tf、占位 action、无来源标记时的加载和从未来状态生成的训练目标，并确认输入 action 不会被导出。
这些测试不等同于完整模型训练或 GR00T 训练验证。
