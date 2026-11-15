# SCALE-AP：可直接执行的实验协议

本协议验证两个相互耦合但可单独拆开的贡献：

1. **SCALE：任务事件后验滤波。** 把 T2-z 的单一环境 latent 改成一组候选 latent，使用真实的任务进展/失败事件更新候选权重。
2. **AP：Active Probing，任务约束主动探测。** 在支持阶段优先执行能最大区分候选 latent、同时仍有足够任务成功概率和安全裕量的动作。

两点都只调用冻结的 LeWM/T2-z，不更新网络参数、不训练 residual、不访问 query 结果。AP 不是独立的探索网络，而是使用已有动作候选库计算一个闭式选择分数。

## 1. 固定数学定义

对当前观测和候选动作块 \(a\)，冻结世界模型在第 \(i\) 个 latent 粒子下给出任务事件概率 \(p_i(a)\)。粒子权重为 \(w_i\)。

任务收益和风险：

\[
\mu(a)=\sum_iw_ip_i(a),\qquad
\sigma^2(a)=\sum_iw_i(p_i(a)-\mu(a))^2.
\]

二元事件的预测熵下降：

\[
\mathcal I(a)=H(w)-\sum_{y\in\{0,1\}}
\bar p_y(a)H(w^{a,y}),
\]

其中 \(\bar p_y(a)=\sum_iw_i[p_i(a)]^y[1-p_i(a)]^{1-y}\)，\(w^{a,y}\) 是观察到事件 \(y\) 后的归一化后验。

支持阶段的主动探测动作：

\[
a_{probe}=\arg\max_{a\in\mathcal A}
\mu(a)-\lambda\sigma^2(a)+\eta\mathcal I(a),
\]

并约束 \(\sum_iw_is_i(a)\geq s_{min}\)，其中 \(s_i(a)\) 是冻结模型估计的安全概率。没有安全头时，先用几何碰撞/力矩规则给出确定性的 \(s_i(a)\)。

执行动作后只读取一个实时事件 \(y\in\{0,1\}\)，按下式更新：

\[
w_i'\propto w_i\exp\{-\beta[-y\log p_i(a)-(1-y)\log(1-p_i(a))]\}.
\]

AP 只在前 \(K\) 个支持动作中启用；后续动作使用 \(\mu-\lambda\sigma^2\) 做利用。这样可以分别回答“结果反馈是否有用”和“主动选动作是否进一步减少交互”。

## 2. 必须固定的任务和事件

首个实验使用单臂平面推物：桌面摩擦和物体内部 payload 独立变化，物体外观、目标区域和相机位置不变。每个 episode 使用一个固定随机种子，动作块长度、控制频率、目标阈值和 episode 最大步数全程不变。

事件检测必须在开始采集前冻结：

- event=1：当前动作块结束时，物体到目标的几何距离相对动作前下降至少 \(\delta\)，或达到终止成功阈值；
- event=0：距离未下降、发生滑脱、越界或超过安全力矩阈值；
- 使用 AprilTag/颜色质心/深度点云的确定性规则；不训练一个新的事件分类器；
- 事件检测器单独在 20 个不参与论文统计的校准片段上测一次漏检率和误报率，阈值随后冻结。

## 3. 条件划分

先固定 3 个摩擦水平 \(F=\{low,mid,high\}\) 和 3 个 payload 水平 \(M=\{light,mid,heavy\}\)，形成 9 个物理条件。训练冻结基础模型时只使用预先指定的条件；测试条件在采集前写入配置文件，不允许看 query 成败后调整。

| split | 条件 | 用途 |
|---|---|---|
| calibration/ID | 训练范围内的 3 个组合 | 冻结 T2-z/LeWM；估计 latent 探测基 |
| seen-ID | 另外 2 个训练范围组合 | 检查常规迁移 |
| unseen-combination | 摩擦和 payload 各自见过、组合未见过 | 主结果 |
| extrapolation | 至少一维超出训练范围 | 泛化压力测试 |

每个物理条件采集 15 个独立 episode：5 个 support、5 个 query、5 个复现实验。复现实验必须使用新的随机顺序，最好在第二天重复；不能把同一个 episode 的不同片段算成独立样本。

## 4. 方法和消融矩阵

主表只保留下面 7 个方法，避免把多个已有模块堆在一起：

| 编号 | 方法 | latent | 支持动作 | 目的 |
|---|---|---|---|---|
| B0 | Frozen | 固定 \(z_0\) | 原生 planner | 无适应基线 |
| B1 | Mean-z T2-z | 原项目 mean/TTT 方案 | 原生 planner | 对比现有适应 |
| B2 | Random latent | 随机粒子但不更新权重 | 原生 planner | 排除 ensemble 偶然收益 |
| B3 | Passive SCALE | 粒子 + 事件后验 | 原生固定顺序动作 | 验证第一个点 |
| B4 | Information-only | 粒子 + \(\mathcal I(a)\) | 只最大化信息增益 | 负对照，证明信息本身不够 |
| B5 | SCALE-AP | 粒子 + 事件后验 | \(\mu-\lambda\sigma^2+\eta\mathcal I\) | 完整两点方法 |
| B6 | Oracle-z | 测试真实物理条件对应 latent | 原生 planner | 仅作上限诊断 |

B1 必须使用你当前项目已经认可的 T2-z 适应设置；不要为了让 B1 变差而修改学习率、步数或初始化。B6 不能作为可部署方法，也不能计入主平均。

## 5. 交互预算和运行顺序

对每个 query episode 预先生成完全相同的候选动作库 \(\mathcal A_t\)，并运行 \(K\in\{0,1,2,5\}\)：

1. K=0：不读取支持事件，直接执行 query。
2. K=1：只执行一次支持动作；B5 使用 AP，B3 使用固定顺序动作；然后冻结后验并执行 query。
3. K=2：执行两次支持动作，每次都用最新后验；第二次可根据第一次事件重新选 probe。
4. K=5：模拟更充足的交互；只作为曲线，不得用来掩盖 K=1/2 的失败。

每个方法、条件、K、随机种子使用相同的 episode 顺序和动作候选库。support 和 query 的视频、动作和结果分目录保存；query 的真实结果在适应结束前禁止被程序读取。

## 6. 8-GB GPU 最小验证顺序

先在 RTX 5070 8-GB 上只做以下三步，不要直接启动全矩阵：

### Stage A：代码和事件接口

~~~bash
python3 -m unittest discover -s scale_tta/tests -v
python3 scripts/run_toy.py --episodes 2000 --seed 7
~~~

要求：4 个单元测试通过；检查 predictive_information_gain 在所有粒子事件相同的动作上接近 0，在事件分裂动作上为正。

### Stage B：冻结 T2-z 单块 rollout

固定一个 sim_pushbox_friction 初始帧、一个动作块、3 个粒子（中心和第一探测方向的正负方向），只渲染 3 个预测视频和 3 个事件概率。记录：显存峰值、单动作块耗时、事件概率、随机种子。不要在这一步报告成功率提升。

建议：

~~~bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
~~~

先沿用已有 T2-z inference 配置；若 OOM，只降低 diffusion inference steps 或粒子数，不能改变方法定义和 action chunk 接口。T2-z 的 context_dim=32、动作维度、视频帧数和 frameskip 必须从当前 checkpoint 配置读取，不要手写成另一套接口。

### Stage C：小规模成功率门槛

只在 1 个未见条件上运行 B0、B1、B3、B5，各 5 个 query episode，K=0/1。进入完整实验必须同时满足：无 OOM、生成视频有限、后验熵在至少一次事件后下降、所有方法的动作维度和控制频率一致。该阶段不是最终统计结果。

## 7. 完整实验清单

### P0：采集前冻结

- [ ] 写入摩擦/payload 条件表、训练/测试 split 和 episode seed。
- [ ] 冻结 camera intrinsics、机器人控制频率、动作块长度、最大步数。
- [ ] 冻结事件阈值 \(\delta\)、成功距离阈值、安全力矩阈值。
- [ ] 导出训练条件的 latent 表，用于固定探测基 \(P_r\)；不能从 query 条件计算 PCA。
- [ ] 记录 T2-z/LeWM checkpoint、git commit、Python/CUDA 版本。

### P1：接口审计

- [ ] 用同一个 obs/action 调用所有方法，确认动作编码、归一化、frameskip、chunk horizon 完全相同。
- [ ] 用 3 个粒子跑一次，确认不同 \(z_i\) 真正进入冻结模型，而不是被 pipeline 的缓存 context 覆盖。
- [ ] 保存每个候选动作的 particle_event、mean_event、variance、information_gain、safe_probability。
- [ ] 检查事件 0/1 的更新都能改变权重，且权重和为 1。

### P2：主成功率实验

- [ ] 运行 B0/B1/B3/B4/B5/B6，所有 seen-ID、unseen-combination、extrapolation 条件都跑 K=0/1/2/5。
- [ ] 每个 cell 至少 5 个 query；正式 RSS 统计建议 15 个 query，若资源不足先完成 5 个 gate 再扩充。
- [ ] 记录 task success、每次事件、失败类型、动作块数量、总交互时长、碰撞/滑脱。
- [ ] 不把 MSE 当作 early stopping、主比较或方法有效性证据。

### P3：最小消融

- [ ] 去掉 AP：B3 与 B5 同条件、同动作库比较。
- [ ] 去掉风险项：\(\lambda=0\)，验证 AP 是否只追求不安全的高信息动作。
- [ ] 去掉安全门：\(s_{min}=0\)，报告接触失败是否增加。
- [ ] 固定 \(\eta\in\{0,0.1,0.25,0.5\}\)，只在 validation 条件选一次；测试条件不得调参。
- [ ] 固定粒子数 \(N\in\{3,9,17\}\)，只在一个 unseen 条件上做效率曲线。

### P4：统计和复现

- [ ] 主指标为同一 episode 顺序下的 paired success difference。
- [ ] 对 episode 做 paired bootstrap 95% CI；同时报告按物理条件聚合的均值，避免某个条件样本过多。
- [ ] 报告 interaction-to-success 曲线、成功率-交互次数曲线和每成功一次的平均动作块数。
- [ ] 用新的 episode 顺序或第二天重复 B0/B3/B5 的 K=1/2。
- [ ] 保存完整 JSON/CSV、视频、配置和 git commit；论文表格从脚本自动生成。

## 8. 论文结论的硬门槛

只有同时满足以下条件，才能写“训练免费适应提升”：

1. B5 在 K=1 或 K=2 的 unseen-combination 上超过 B1 和 B3；
2. paired bootstrap 95% CI 的下界大于 0，且第二次 episode 顺序/第二天方向一致；
3. B5 的 catastrophic contact failure 不高于 B3；
4. B4（只看信息增益）不能与 B5 等价，否则第二个点没有必要；
5. 所有提升来自任务成功率，MSE 只能作为诊断图；
6. 没有读 query 结果、没有更新世界模型参数、没有新增训练数据。

若门槛 1 或 2 不成立，论文应改写为“任务事件后验在何种隐藏动力学偏移下失效”，不能把 MSE 下降包装成控制提升。

## 9. 建议的日志字段

~~~json
{
  "method": "scale_ap",
  "condition": {"friction": "high", "payload": "mid"},
  "episode_seed": 17,
  "K": 2,
  "step": 1,
  "selected_action_index": 3,
  "observed_event": 0,
  "posterior_entropy": 1.73,
  "effective_particles": 5.8,
  "mean_event": 0.61,
  "event_variance": 0.09,
  "information_gain": 0.34,
  "safe_probability": 0.97,
  "success": 1,
  "catastrophic_contact": 0,
  "wall_time_ms": 412
}
~~~
