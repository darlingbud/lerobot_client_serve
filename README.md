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
边缘端 Robot 可以是任意 LeRobot 兼容机器人：SO-101、SO-100 等。

## 目录结构

```
lerobot_remote/
├── src/lerobot_remote/
│   ├── __init__.py
│   ├── protocol.py          # 通用通信协议
│   ├── robot_client.py      # 抽象机器人客户端接口
│   ├── lerobot_robot.py     # LeRobot 机器人实现
│   ├── policy_server.py     # 抽象策略服务器基类
│   ├── policies.py          # 具体策略实现 (ACT, YOLO+IK, 硬编码)
│   └── remote_client.py     # 边缘端 WebSocket 客户端
├── configs/
│   └── config.example.yaml  # 配置示例
├── scripts/
│   ├── run_server.py        # 启动云端 Server
│   └── run_client.py        # 启动边缘端 Client
└── tests/
```

## 快速开始

### 1. 安装依赖

```bash
pip install websockets msgpack numpy torch opencv-python pyyaml
```

### 2. 配置

复制配置示例文件并修改：

```bash
cp configs/config.example.yaml configs/config.yaml
# 编辑 config.yaml
```

### 3. 启动云端 Server（4090）

```bash
python scripts/run_server.py --config configs/config.yaml
```

### 4. 启动边缘端 Client

```bash
python scripts/run_client.py --config configs/config.yaml
```

## 使用示例

### 命令行参数

```bash
# 启动边缘端
python scripts/run_client.py \
  --server-url ws://100.89.143.11:8000 \
  --robot-type so101 \
  --robot-port /dev/ttyUSB0 \
  --camera-port 0

# 干运行（只推理不执行）
python scripts/run_client.py --dry-run --verbose
```

### Python 代码

```python
from lerobot_remote import SO101Robot, RemoteRobotClient

# 创建机器人
robot = SO101Robot(port="/dev/ttyUSB0", camera_port=0)
robot.connect()

# 创建远程客户端
client = RemoteRobotClient(robot=robot, server_url="ws://100.89.143.11:8000")
client.connect()

# 控制循环
while True:
    obs = robot.get_observation()
    action = client.infer(obs)
    robot.execute(action.action)
```

## 实现自己的算法

云端只需实现 `PolicyServer` 接口：

```python
from lerobot_remote import PolicyServer, Observation, Action, ServerMetadata

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
| `robot.type` | 机器人类型 (so101, etc.) |
| `robot.port` | 机器人串口 |
| `robot.camera_port` | 相机端口 |

具体值写在 `configs/config.yaml` 中（不上传）。

## License

MIT
