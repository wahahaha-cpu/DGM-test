# Q-Adapt 执行说明

## 代码位置

- scale_tta/qadapt.py：行为商类、后验更新、主动划分动作、可信类鲁棒动作
- scale_tta/core.py：SCALE-AP baseline
- scale_tta/bridge.py：冻结 T2-z/LeWM callback bridge
- scale_tta/tests/test_scale.py：SCALE-AP 和 Q-Adapt 单元测试
- Q_ADAPT_PROPOSAL.md：方法定义和论文主线

## 1. 安装和 CPU 检查

~~~bash
python3 -m pip install -e .
python3 -m unittest discover -s scale_tta/tests -v
~~~

预期通过 6 个测试。

## 2. Q-Adapt 最小调用

冻结模型需要提供一个 callback：

~~~python
def predict_event(z, action):
    # z: 一个 T2-z environment latent
    # action: 当前 planner 的一个 action chunk
    # 返回 [0, 1] 中的 task-progress probability
    predicted_video = frozen_t2z_rollout(obs, action, z)
    return geometric_progress_probability(predicted_video, goal)
~~~

使用 T2-z 训练阶段保存的 environment latent table：

~~~python
import numpy as np
from scale_tta import QAdaptConfig, QAdaptFilter

latent_table = np.load("train_environment_latents.npy")
actions = native_planner_candidates(obs, goal)

config = QAdaptConfig(
    signature_threshold=0.10,
    progress_threshold=0.30,
    safety_threshold=0.80,
    credible_mass=0.90,
    beta=4.0,
)

qadapt = QAdaptFilter.from_latent_table(
    latent_table=latent_table,
    actions=actions,
    predict_event=predict_event,
    config=config,
)

probe_action, probe_log = qadapt.choose_probe_action(
    actions=actions,
    predict_event=predict_event,
    predict_safe=predict_safe,
)

execute(probe_action)
event = event_detector(obs_before, obs_after, goal)
qadapt.update(event, probe_action, predict_event)

robust_action, robust_log = qadapt.choose_robust_action(
    actions=native_planner_candidates(obs_after, goal),
    predict_event=predict_event,
)
execute(robust_action)
~~~

## 3. 和 T2-z/LeWM 对接时必须保持一致

以下接口不能修改：

- checkpoint；
- action dimension；
- action normalization；
- frameskip；
- action chunk horizon；
- observation frame count；
- planner candidate generation；
- camera calibration；
- task success threshold。

Q-Adapt 只接受一个冻结模型 callback，不反向传播，不更新 T2-z latent table，不训练新的 event network。

## 4. 8GB GPU 最小实验

先只运行一个初始 observation：

1. 从训练 latent table 取 8 到 16 个 latent；
2. 取 native planner 的 8 到 32 个 action candidates；
3. 计算行为签名矩阵，形状为 [latent_count, action_count]；
4. 在签名空间聚类；
5. 选择一次 probe；
6. 执行一次真实动作；
7. 更新 quotient-class posterior；
8. 选择一次 robust action。

记录：

~~~json
{
  "latent_count": 16,
  "action_count": 16,
  "quotient_class_count": 4,
  "signature_time_ms": 0,
  "probe_index": 3,
  "observed_event": 1,
  "posterior_weights": [],
  "credible_classes": [],
  "robust_action_index": 2,
  "peak_memory_mb": 0,
  "success": 0
}
~~~

显存不足时，依次降低：

1. latent prototype 数量；
2. action candidate 数量；
3. diffusion inference steps。

不能改变 latent table、动作编码和任务事件定义。

## 5. 最小论文实验

先只跑一个 unseen friction-payload 条件：

~~~text
Frozen
Mean-z T2-z
SCALE-AP
Q-Adapt without refinement
Q-Adapt
~~~

每个方法：

- K=0、1、2；
- 5 个 query episodes；
- 相同 episode seeds；
- 相同动作候选；
- 相同事件检测器。

只有在 Q-Adapt 在 K=1 或 K=2 的成功率超过 Mean-z、SCALE-AP 和 Q-Adapt without refinement 后，才扩大到完整条件矩阵。

## 6. 必须保存的中间结果

每个 episode 保存：

- raw latent table；
- behavioral signature matrix；
- quotient labels；
- quotient class centroids；
- candidate action list；
- each-action class progress；
- selected probe；
- observed event；
- posterior weights；
- credible class indices；
- robust action；
- task success；
- failure type；
- wall time；
- peak GPU memory。

## 7. 结果解释

- raw latent distance 和 behavior distance 高相关：Q-Adapt 的 quotient 必要性不足；
- quotient classes稳定，但 refinement 无收益：第一个点有效，第二个点失败；
- refinement 提高 success，但 unconstrained probe 增加碰撞：progress/safety constraint 有必要；
- K=1/2 success 提高：支持少交互适应成立；
- MSE 下降但 success 不升：不得作为正面结论。

