# SE-JEPA: Self Evolution Latent World Models

CVPR 2027 投稿摘要（2026-10-30 版，英文 202 词）

## Abstract

Latent world models let agents plan by imagining the outcomes of their actions, but they stop learning once deployed: when objects or appearances shift, prediction errors become planning failures. Test-time adaptation such as AdaJEPA repairs these errors within an episode but then resets, meeting every recurring shift as if for the first time. We therefore seek a world model that remembers what it learns. However, we find that simply keeping the adapted weights fails: updates across episodes are nearly orthogonal, and accumulating them degrades planning. To overcome this, we present SE-JEPA, a self-evolving latent world model. Inspired by complementary learning systems, it pairs fast within-episode adaptation with slow cross-episode consolidation, replaying experience into slow learners, one per self-discovered domain. Supervised only by its own prediction errors, SE-JEPA decides what to consolidate, where to store it, and when to trust it for planning. Because its pretrained model is never overwritten, nothing it already knows can be forgotten. It requires no shift labels or task boundaries and benefits both gradient-based and sampling-based planners. Across unseen object shapes, visual shifts, and deformable rope and granular manipulation, SE-JEPA grows stronger the longer it is deployed, setting a new state of the art by a wide margin.

## 摘要

潜空间世界模型让智能体通过想象动作的后果来规划，但一经部署便停止学习：一旦物体或外观发生变化，预测误差就会演变为规划失败。AdaJEPA 等测试时适应方法能在单个回合内修正这些误差，但回合结束即重置，每次遇到反复出现的偏移都如同初见。因此，我们希望实现能够持久保留学习记忆的潜空间世界模型。但我们发现，简单保留适应后的权重行不通：不同回合的更新几乎相互正交，累积起来反而损害规划。为解决这一问题，我们提出自我进化的潜空间世界模型 SE-JEPA。受互补学习系统启发，SE-JEPA 将回合内的快速适应与跨回合的缓慢巩固相结合，通过回放把经验沉淀到慢学习器中，每个自主发现的领域对应一个慢学习器。SE-JEPA 仅以自身的预测误差为监督，自主决定巩固什么、存放在哪里、何时信任其用于规划。由于预训练模型从不被改写，模型已掌握的知识不会被遗忘。SE-JEPA 无需偏移标签或任务边界，并对基于梯度和基于采样的规划器均有提升。在未见过的物体形状、视觉偏移以及绳索与颗粒等可变形物体操作上，SE-JEPA 部署越久越强大，以显著优势刷新了当前最优水平。
