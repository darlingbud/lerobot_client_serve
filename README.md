# LeRobot Remote

通用云边端机械臂控制框架。边缘端获取机器人状态发送到云端，云端推理后返回动作。

## 架构

```
边缘端 Client                    云端 Server
┌──────────────┐                ┌──────────────┐
│ get_obs()    │── WebSocket ──→│ infer(obs)  │
│ execute(act) │←── action ─────│              │
└──────────────┘                └──────────────┘
```

云端 Server 可以是任意算法：ACT、YOLO+逆运动学、硬编码轨迹等。

## 目录结构

```
lerobot_remote/
├── src/lerobot_remote/
│   ├── __init__.py
│   ├── protocol.py          # 通用通信协议
│   ├── msgpack_numpy.py     # msgpack 序列化
│   ├── robot_client.py      # 通用机器人客户端
│   └── policy_server.py     # 通用策略服务器基类
├── configs/
│   └── config.example.yaml  # 配置示例
├── scripts/
│   ├── run_server.py
│   └── run_client.py
└── tests/
```

## 快速开始

### 1. 安装依赖

```bash
pip install websockets msgpack numpy torch
pip install -e .
```

### 2. 配置

复制配置示例文件并修改：

```bash
cp configs/config.example.yaml configs/config.yaml
# 编辑 config.yaml
```

### 3. 启动云端 Server

```bash
python scripts/run_server.py --config configs/config.yaml
```

### 4. 启动边缘端 Client

```bash
python scripts/run_client.py --config configs/config.yaml
```

## 实现自己的算法

云端只需实现 `PolicyServer` 接口：

```python
from lerobot_remote import PolicyServer, Observation, Action

class MyAlgorithm(PolicyServer):
    def infer(self, obs: Observation) -> Action:
        # 实现你的算法
        action = self.compute_action(obs)
        return Action(action=action)

    def get_metadata(self) -> ServerMetadata:
        return ServerMetadata(
            name="my_algorithm",
            algorithm="custom",
            description="My custom algorithm"
        )
```

## 配置说明

| 配置项 | 说明 |
|--------|------|
| `server.host` | 云端监听地址 |
| `server.port` | 云端监听端口 |
| `client.server_url` | 边缘端连接的服务器地址 |
| `robot.host` | 机器人控制服务地址 |
| `robot.port` | 机器人控制服务端口 |

具体值写在 `configs/config.yaml` 中（不上传）。

## License

MIT
