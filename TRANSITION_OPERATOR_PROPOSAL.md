# Transition-Law Memory and Outcome Flow

状态：研究假设，尚未宣称已完成创新性证明。

## 1. 重新定义问题

隐藏摩擦、质量、接触刚度和负载变化的影响，最终通过“同一动作会造成什么状态转移”表现出来。现有方法通常把未知物理压缩成一个静态上下文向量，或者把历史片段存入记忆，再由策略自己解释历史。

这会留下三个结构性问题：

1. 上下文向量的坐标没有物理规范，欧氏距离不代表控制上的相似性。
2. 预测损失较低不能保证动作会带来正确的任务后果。
3. ODE policy 保证动作轨迹连续，却没有保证动作造成的物理结果连续。

因此主对象改为动作干预的响应规律，而不是隐藏物理参数本身。

## 2. 核心概念

### 2.1 Transition-Law Memory

给定冻结的视觉编码器和 LeWM，令：

\[
h_t=E(o_{\leq t}),\qquad
\Delta h_t=h_{t+1}-h_t.
\]

冻结世界模型给出基础响应：

\[
\Delta \hat h^{0}_t=F_\theta(h_t,a_t).
\]

实际交互形成响应残差：

\[
r_t=\Delta h_t-\Delta \hat h^{0}_t.
\]

Transition-Law Memory 不保存一个待解释的 z，而保存一个从动作干预到转移修正的算子：

\[
\mathcal K_t:\psi(a)\mapsto r.
\]

在局部动作空间中：

\[
\hat{\Delta h}(h,a)
=
F_\theta(h,a)+\mathcal K_t(h)\psi(a).
\]

其中 \(\psi(a)\) 是固定的动作基函数，\(\mathcal K_t\) 可以用低秩形式表示：

\[
\mathcal K_t=\sum_{j=1}^{r}u_{t,j}v_{t,j}^{\top}.
\]

一次真实转移即可进行递归更新：

\[
e_t=r_t-\mathcal K_t\psi(a_t),
\]

\[
\mathcal K_{t+1}
=
\mathcal K_t+
\eta_t
\frac{e_t\psi(a_t)^\top}
{\lambda+\|\psi(a_t)\|^2}.
\]

更稳定的实现使用递归最小二乘，并限制算子秩和谱范数。这个记忆可以直接回答反事实问题：

\[
a'\mapsto \hat{\Delta h}(h,a').
\]

这正是静态 z 或 episode retrieval 无法直接提供的对象。

### 2.2 Outcome Flow

令任务目标在 latent transition 空间中的期望速度为 \(v_g(h,\tau)\)。使用算子得到动作到转移的局部 Jacobian：

\[
J_t(a)=
\frac{\partial \hat{\Delta h}(h,a)}
{\partial a}.
\]

动作轨迹不再仅在原始 action space 中积分，而是要求其产生期望的 outcome flow：

\[
J_t(a_\tau)\dot a_\tau
\approx v_g(h,\tau).
\]

最小范数解为：

\[
\dot a_\tau
=
(J_t^\top WJ_t+\epsilon I)^{-1}J_t^\top Wv_g(h,\tau).
\]

若保留已有 ODE policy 的行为先验，可把它作为正则项：

\[
\dot a_\tau =
\arg\min_u
\|J_tu-v_g\|_W^2+
\beta\|u-v_{\mathrm{ODE}}(a_\tau,\tau)\|^2.
\]

因此物理记忆改变的是动作到结果的映射，ODE 先验负责动作连续性，二者在同一个响应方程中结合。

## 3. 两个论文级贡献点

### Contribution A

提出 operator-valued physical memory。隐藏物理不再被定义为需要恢复的数值 latent，而被定义为作用在动作干预上的 transition law。记忆的可组合性和可迁移性由动作基函数与转移残差直接约束。

### Contribution B

提出 outcome-flow control。策略在任务结果空间中生成期望转移，再通过在线估计的 response operator 拉回动作空间。这样动作连续性和物理结果连续性成为同一个求解问题。

这两个点共同改变了三个基本对象：

| 层面 | 常见做法 | 本方法 |
| --- | --- | --- |
| 记忆对象 | episode、检索向量、静态 z | action-to-transition operator |
| 测试时目标 | 预测误差或 latent 一致性 | 反事实转移和任务后果 |
| 动作生成空间 | action space | outcome space，再通过响应算子求动作 |

## 4. 与已有工作的边界

T2-z 证明环境级 latent 可以从多轨迹中承载隐藏物理上下文。这里进一步改变适应对象。相关论文仍然必须逐篇核对，当前不能直接把本文方法称为已证明的新范式。

ODE Policy 学习动作向量场，并从上一个动作积分以获得连续轨迹。它的核心对象仍是 action dynamics。这里的假设是，策略应该在 outcome space 中生成流，再由物理响应算子反解动作。genui{"citation":{"refs":["turn559413364academia0","turn559413364academia3"]}}

SCOUT 和 few-shot residual adaptation 都从 action-outcome mismatch 或 latent prediction residual 更新适应状态。它们说明测试时反馈确实有价值，但主要仍然更新 belief latent 或 residual correction。Transition-Law Memory 需要验证算子结构是否比等参数量的 latent/residual 更能跨动作方向迁移。genui{"citation":{"refs":["turn359304612academia1","turn359304612academia2"]}}

近期诊断还表明，动作的写法本身会改变世界模型的行为，说明 action interface 是世界模型的结构性问题。算子记忆必须在规范化动作基上定义，并加入 action re-parameterization 测试。genui{"citation":{"ref":"turn5206761academia1"}}

## 5. 最小可证伪实验

使用现有 PhysCalWorld FetchSlide 和冻结 LeWM/T2-z，不重新训练视觉主干。

### 实验一：算子是否可迁移

- support 在动作方向集合 A1 上采集 1、2、5 次转移。
- query 使用未出现在 support 中的动作方向集合 A2。
- 对比静态 T2-z 更新、普通 latent residual、Transition-Law Memory。
- 主要指标只用任务成功率和成功所需交互数。
- MSE、Jacobian 误差和算子谱范数作为诊断指标。

若 operator memory 只在 support 动作附近有效，且不能改善 A2 上的成功率，核心假设失败。

### 实验二：结果空间流是否有效

固定同一 frozen world model、候选动作预算和交互次数，对比：

1. native planner；
2. T2-z test-time adaptation；
3. operator memory；
4. operator memory + action-space ODE；
5. operator memory + outcome flow。

主要指标为任务成功率、首次成功所需交互数、动作平滑度和失败接触次数。MSE 不作为模型效果主指标。

若 outcome flow 的增益在等候选动作预算下消失，说明它只是搜索预算增加造成的收益。

### 实验三：动作表示重参数化

对同一真实末端轨迹使用绝对动作、delta action 和归一化 action 三种写法。检查：

- frozen world model 的响应是否改变；
- operator memory 是否在规范化后保持一致；
- outcome flow 的成功率是否稳定。

这个实验直接检验方法是否解决 action interface 的结构问题。

## 6. 8GB 实现边界

第一版不训练新大模型：

- 冻结 LeWM/T2-z；
- 缓存 \(h_t\)、\(F_\theta(h_t,a_t)\) 和 action basis；
- 只在线更新低秩 \(\mathcal K\)；
- rank 取 4、8、16；
- 使用 FP32 递归更新和 batch size 1；
- 只运行 K=0、1、2、5；
- 先在 5 个 test dynamics 上完成机制验证，再决定是否扩大。

第一版代码不应把该方法与 PARE、Q-Adapt、SCALE-AP 同时叠加。它们只保留为行为层基线和负结果诊断。

## 7. 论文成立条件

以下现象至少需要同时出现：

1. raw z distance 不能稳定预测动作后果相似性。
2. Transition-Law Memory 在未观测动作方向上优于同等参数量的 latent residual。
3. outcome flow 在固定候选动作预算下提高任务成功率，而不是只增加搜索次数。
4. adaptation 交互次数减少，成功率提升仍然存在。
5. action re-parameterization 不会显著破坏响应算子。

若这些条件不成立，应停止包装概念，回到 T2-z 的表示学习问题或承认该方法只是残差适应。

## 8. 当前判断

这是目前比 PARE 和 Q-Adapt 更接近“从本质改对象”的候选。它的论文价值来自：

\[
\text{context vector}
\rightarrow
\text{transition law operator}
\]

以及：

\[
\text{action-space generation}
\rightarrow
\text{outcome-space flow + response pullback}.
\]

创新性仍需通过完整文献碰撞检查和上述三组实验确认。当前仓库里的 PARE、Q-Adapt 和 SCALE-AP 代码不能被称作该方法的实现。

