# LeRobot Remote

WebSocket layer for LeRobot to enable cloud-edge architecture.

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    云端 4090 Server                         │
│              LeRobot Policy + WebSocket Server              │
│                 监听 8000 端口                              │
└────────────────────────────┬────────────────────────────────┘
                             │ WebSocket (obs → action)
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    边缘端 192.168.3.164                      │
│           LeRobot Robot + WebSocket Client                  │
│    收集 obs → 发送到云端 → 接收 action → 发送给机械臂        │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      SO-101 机械臂                          │
└─────────────────────────────────────────────────────────────┘
```

## 目录结构

```
lerobot_remote/
├── src/lerobot_remote/
│   ├── __init__.py
│   ├── msgpack_numpy.py          # Msgpack with numpy support
│   ├── policy_loader.py          # LeRobot policy loader
│   ├── websocket_policy_server.py # 云端 Policy Server
│   └── websocket_robot_client.py # 边缘端 Robot Client
└── scripts/
    ├── run_server.py            # 启动云端 Server
    └── run_client.py            # 启动边缘端 Client
```

## 快速开始

### 云端（4090 Server）

```bash
# 启动 Policy Server
python scripts/run_server.py \
  --checkpoint /path/to/checkpoint/pretrained_model \
  --port 8000
```

### 边缘端（Robot Machine）

```bash
# 启动 Robot Client
python scripts/run_client.py \
  --server-url ws://100.89.143.11:8000 \
  --robot-host 127.0.0.1 \
  --robot-port 8765
```

## 依赖

- Python 3.8+
- websockets
- msgpack
- numpy
- torch
- lerobot (from source)

## License

MIT
