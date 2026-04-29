# lerobot-record 录制日志

## 2026-04-28 录制 Grab the black cube 数据

### 成功运行的命令

```bash
/home/donquixote/miniconda3/envs/lerobot/bin/lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras='{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"} }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=Donquihote168077/grab-test \
    --dataset.num_episodes=10 \
    --dataset.single_task="Grab the black cube" \
    --dataset.push_to_hub=true \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=30
```

### 注意事项

1. **本地缓存冲突**：如果遇到 `FileExistsError`，删除缓存目录：
   ```bash
   rm -rf ~/.cache/huggingface/lerobot/<repo_id>
   ```

2. **遥操作录制**：需要两个串口（follower + leader）

3. **上传 HuggingFace**：`push_to_hub=true` 会自动上传数据集

4. **相机设备**：用 `index_or_path: 4`（`/dev/video4`）

5. **使用完整路径**：避免 `.local/bin/lerobot-record` 的环境问题
