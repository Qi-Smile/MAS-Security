# MAS 安全论文详解

## 综合观察
- **研究主题脉络**：近两年 MAS 安全工作围绕“单点渗透 + 群体传播”展开。攻击侧从 token 级感染（CODES、Agent Smith）、通信链路篡改（MAST）、知识操纵（PsySafe、Flooding Spread）、带带宽约束的多跳提示投递（Agents Under Siege）到视觉场景免疫（Cowpox），不断放大多代理互动带来的风险。防御侧则引入图学习与拓扑剪枝（G-Safeguard、BlindGuard）或类医学免疫（Cowpox），强调跨回合、跨角色的态势感知与阻断。
- **威胁模型共性**：大部分攻击假设对手只需一次或极少数访问权限即可控制一个代理或消息通道，再利用记忆/RAG、角色分工或网络拓扑把异常扩散；防御方案则往往面对“无法完全控制系统”、“缺乏有标签异常数据”或“部署资源受限”等现实约束。
- **方法论趋势**：攻击手段大量引入连续-离散混合优化、最大流最小费、MCTS+DPO、知识编辑（ROME）等系统化技术；防御方法则通过图神经网络、无监督对比学习、RAG 得分再加权与控制理论（感染率 vs 恢复率）等方式建立形式化保障。
- **实验设计**：论文普遍构建链、树、星、随机或社会群聊等 MAS 拓扑，指标覆盖攻击成功率 (ASR)、感染比例 $p_t$、传播轮数、危险率 (PDR/JDR)、心理量表得分、任务准确率、检测精度等；不少工作还考察黑盒迁移、价格/延迟约束或 Tamper Defender 检出率，强调攻击可操作性与防御可落地性。

---

## LLM-based Multi-Agents System Attack via Continuous Optimization with Discrete Efficient Search (CODES) — COLM 2025
### 背景与问题
- **研究背景**：LLM 驱动的多代理系统可协同处理复杂任务，但也使得“单次越狱→多代理传播”成为现实威胁。现有 token 级越狱（如 GCG、ADC）主要针对单回合 LLM，在 MAS 中会遇到上下文多样、输出再循环、优化震荡等难题。
- **问题刻画**：作者要求攻击者仅获得一次与任一代理交互的机会，便可让恶意后缀跨角色、跨回合扩散。《CODES》指出旧方法在 MAS 上存在两大不足：① 产生的后缀泛化性差，不同代理的系统提示/记忆导致失败；② 优化过程不稳定，输出中出现的恶意串又被作为新输入，放大会话漂移。

### 方法与建模
- **优化目标**：传统 GCG 使用算术平均困惑度 $L_{\mathrm{AM}}=\frac{1}{N}\sum_{i=1}^N \text{perplexity}(r_i)$。CODES 调整为 RMS 加权：
  
    $$
    L_{\mathrm{RMS}}(t)=\left(\frac{1}{N}\sum_{i=1}^N \text{perplexity}(t_i)^2\right )^{1/2}
    $$
  
  自适应地放大难预测 token（尤其是恶意后缀首 token）的权重，提升稳定性。
- **连续-离散混合优化**：保持一个密集概率向量 $\mathbf{z}_{1:n}\in\mathbb{R}^{n\times V}$，使用动量更新
  
    $$
    \mu \leftarrow \gamma \mu + 
abla_{\mathbf{z}} L(\mathbf{z}),\qquad 
    \mathbf{z} \leftarrow \text{TopKCoordinateUpdate}(\mu)
    $$
  
  仅对候选坐标做投影搜索，避免 ADC 在低损失区域出现的震荡。
- **强化泛化**：提出位置感知采样与多场景集成训练，将 ShareGPT 安全对话、HatefulBench 恶意短语等上下文批量打乱，让生成的后缀在不同代理角色/记忆深度下仍有效。

### 实验与结论
- **三类威胁场景**：① 单次注入感染全部代理（评估 TASR、ASR、感染占比 $N_i$）；② 链式操控，只能与链头 agent 交互却让链尾输出目标字符串；③ 面对“检测-改写-回答”三级安全机制仍要穿透。
- **实验成果**：在 Vicuna、Zephyr、Llama2/3、Qwen2.5 等开源模型上，CODES 达成最高 90% ASR，$N_i$ 提升至 90% 左右；对 GPT-4o / GPT-o1 等黑盒模型依然有 25% 以上的可迁移攻击率。
- **消融目的**：分析目标串长度、集成规模、损失函数设计。结果显示当 ensemble ≥80 时 ASR 饱和；RMS 损失优于 AM、显式重加权；目标串变长成功率下降但依然保持 80% 水平。整体表明 CODES 在泛化与稳定性上明显优于 GCG、ADC。

---

## PsySafe: A Comprehensive Framework for Psychological-based Attack, Defense, and Evaluation of Multi-agent System Safety — arXiv 2401.11880
### 背景与问题
- **研究背景**：多代理系统常设定不同“角色”来完成任务，然而角色提示本身可能诱发价值观偏移。作者观察到“黑暗人格”提示能显著改变代理输出，现有安全研究多聚焦单代理越狱，缺少面向 MAS 的心理学视角。
- **问题定位**：PsySafe 试图回答三问：① 如何发现“心理黑化”导致的 MAS 安全漏洞？② 如何量化 MAS 在心理和行为层面的危险程度？③ 如何设计针对性的心理/角色防御？

### 方法与建模
- **心理注入攻击**：构造六维道德基础（关怀、公平、忠诚、权威、神圣、自由）的暗人格提示，并结合诱导指令（Inducement Instruction）与隐式示例（Red ICL），让被注入代理在安全和危险任务中都输出带证据的危险内容。
- **行为与心理评估**：使用改进的 DTDD 量表替换人群测试，每题三档评分 $\{1,5,9\}$ 得到心理危险分。定义过程危险率与联合危险率：
  
    $$
    \mathrm{PDR}(r)=\frac{\sum_{p} \mathbb{I}\big[\exists b\in B_{p,r}, f(b)=1\big]}{|P|},\quad
    \mathrm{JDR}(r)=\frac{\sum_{p} \mathbb{I}\big[\forall b\in B_{p,r}, f(b)=1\big]}{|P|}.
    $$
  
  同时统计心理分与危险行为的点双列相关，量化心理-行为关联。
- **防御设计**：提出三类安全机制：① 输入过滤（GPT-4、Llama Guard）；② Doctor Defense——若心理评分超阈值 $P$，迭代求解新的系统提示直到分数下降；③ Police Defense——增设监督代理，对危险对话实时批评干预。

### 实验与结论
- **环境设置**：在 Camel、AutoGen、MetaGPT、AutoGPT 上构建安全/危险任务数据集（共 859 任务，含 13 类危险维度），测试无攻击、HI（人机交互注入）、Traits（角色注入）、混合及高频版本。
- **结果与目的**：发现暗人格注入显著提高 PDR/JDR（AutoGen 中 JDR 超过 50%），心理分与危险输出高度相关；高频输入注入加速危险扩散。防御实验显示 Doctor Defense 可将 JDR 从 50% 降至 0%，并改善心理评分；Police Defense 则抑制危险积累但仍有残余。
- **意义**：PsySafe 证明 MAS 需要心理+行为双视角的安全评估，并给出心理治疗式防御思路，为后续研究提供基准。

---

## Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast — ICML 2024
### 背景与问题
- **研究背景**：多模态代理（MLLM）可通过图像检索与聊天合作，然而对抗图像可能被缓存并跨代理传播。先前工作多数关注单 agent 越狱，此文首次系统化“感染式”跨代理攻击。
- **问题建模**：假设系统由随机配对的问答 agent 构成，感染率 $\beta$ 表示携带病毒图像的输出概率，恢复率 $\gamma$ 表示图像被淘汰概率。作者建立传播模型分析什么时候感染占比 $c_t$ 会指数级增长。

### 方法细节
- **传播动力学**：设 $c_t$ 为病毒图像比例、$p_t=\xi c_t$ 为实际感染率。推导差分方程 $c_{t+1}=(1-\gamma)c_t+\frac{\beta}{2}c_t(1-c_t)$，并转化为微分方程
  
  $$
  \frac{dc}{dt} = \frac{\beta}{2}c(1-c)-\gamma c,
  $$
  
  得到解 $c_t=\frac{c_0(\beta-2\gamma)}{(\beta-2\gamma-c_0\gamma)e^{-(\beta-2\gamma)t/2}+c_0\gamma}$。当 $\beta>2\gamma$ 时极限 $c_\infty=1-\frac{2\gamma}{\beta}$，反之感染衰减，为后续“可证防御”奠定阈值。
- **三重损失攻击目标**：针对采样到的对话记录集 $\{[H^Q_m,S^Q_m],[H^A_m,S^A_m,Q_m],P_m\}_{m=1}^M$，构造对抗图像 $V_{\mathrm{adv}}$ 最小化
  
  $$
  \min_{V_{\mathrm{adv}}}\ \lambda_R L_R+\lambda_Q L_Q + \lambda_A L_A,
  $$
  
  其中 $L_R=-\frac{1}{M}\sum_m \mathrm{Enc}_{\text{text}}(P_m)^\top \mathrm{Enc}_{\text{img}}(V_{\mathrm{adv}})$ 提升检索概率；$L_Q,L_A$ 为问答阶段生成目标恶意序列的对数似然损失（式 (17)(18)）。
- **系统流程**：Algorithm 1 模拟随机配对聊天，包含 RAG 选图、提问、回答、记忆更新；攻击通过一次注入让对抗图像在记忆中自复制。

### 实验设计
- **仿真规模**：最多 10^6 个 LLaVA / InstructBLIP 代理，随机配对 50 轮，记录累计感染率 $p_t$ 与达到 85%、90%、95% 感染所需轮数。对比视觉提示注入、纯文本注入及顺序式攻击基线。
- **目的与结果**：单张图像可在 O(log N) 轮内传播至 90% 以上代理（与理论一致）；顺序攻击在恢复率存在时局限于少量代理。还分析 RAG 权重、步长、动量对感染速度的影响。文章提出若防御能使 $\beta \le 2\gamma$ 即可从理论上抑制传播，为后续 Cowpox 等防御提供指导。

---

## Flooding Spread of Manipulated Knowledge in LLM-Based Multi-Agent Communities — arXiv 2407.07791
### 背景与问题
- **研究背景**：实际平台允许用户部署多代理协同完成问答、写作等任务。一旦某个 agent 被预先篡改，其在群聊中的“伪知识”可能通过对话与 RAG 缓存持久影响其他代理。
- **研究缺口**：现有工作多聚焦 prompt 或工具攻击，较少考虑“参数预置 + 对话渗透”的组合。本文提出两阶段攻击，使代理在不自知的情况下持久传播反事实或仇恨信息。

### 方法细节
- **阶段一（Persuasiveness Injection）**：收集 1000 对“简短回答 vs 证据丰富回答”，以 DPO + LoRA 微调，使代理倾向输出长、带证据的结论，从而增强说服力。DPO 损失为
   $$
   \mathcal{L}_{\text{DPO}}= -\mathbb{E}_{(x,y^+,y^-)}\Big[\log \sigma\Big(\beta\big(\log \pi_\theta(y^+|x)-\log \pi_{\text{ref}}(y^+|x)-\log \pi_\theta(y^-|x)+\log \pi_{\text{ref}}(y^-|x)\big)\Big)\Big].
   $$
- **阶段二（Manipulated Knowledge Injection）**：利用 ROME 在 FFN 中定位三元组 $(s, r, o)$ 的关键层，将其值向目标反事实 $o'$ 调整，使代理在保持其它能力的同时，对指定实体持有错误认知。
- **攻击流程**：在群聊中，注入代理基于 tampering 后知识输出“看似合理的证据”影响其他代理；同时利用 RAG 将篡改后的历史重复使用，实现持续传播。

### 实验设计
- **任务与指标**：在 CounterFact/zsRE（反事实）与 Toxic CounterFact/zsRE（毒性）上评估 Vicuna-7B、LLaMA-3-8B、Gemma-7B。指标包括：攻击代理与无辜代理的知识采纳准确率、重述准确率、邻域影响（locality）、以及 MMLU 用于监测基础能力。
- **结果解读**：两阶段联合使 Vicuna 非注入代理的采纳准确率提升约 15–20%，且心理测试分更趋危险；MMLU 几乎不变（波动 <0.5%），说明攻击隐蔽。扩展实验显示滞后 RAG 也会继续传播伪知识，凸显平台需引入 Guardian agents 或事实核查流程。

---

## Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems — arXiv 2410.07283
### 背景与问题
- **研究背景**：多代理流水线中常有“文档读取→数据库→代码执行”等角色，早期 prompt injection 关注模型与外部文档的关系，但未充分考虑 agent 间消息共享导致的自复制。
- **问题描述**：作者提出“Prompt Infection”——攻击者只需在外部文档植入自复制提示，首个读取 agent 被感染后，提示会在 agent 间不断复制并触发恶意行为（诈骗、恶意链接、数据窃取）。

### 方法与建模
- **感染链条**：感染提示要求下游 agent 在执行任务的同时附带相同提示，形成类似逻辑斯蒂增长的感染曲线。作者还演示攻击可操纵记忆重要度评分（指示模型评分为 10），使恶意提示长期位列高优先级。
- **防御机制**：提出 LLM Tagging——在跨 agent 消息前添加源标识 `[AGENT NAME]:`，并与 Marking、Instruction Defense、Sandwich 等传统 prompt 防御组合。实验显示单独使用效果有限，但与 Marking 联合可将攻击成功率降至 0。

### 实验与结论
- **多代理应用**：构造 Email/PDF/Web Reader、CSV Reader、Coder 等流水线，既测全局消息共享，也测局部消息传递。分别在 GPT-4o 与 GPT-3.5 Turbo 上评估诈骗、恶意软件、内容操控、数据窃取。
- **社群传播**：模拟 10–50 人社会代理随机对话，感染率 $p_t$ 呈逻辑斯蒂曲线；操纵记忆评分使感染在 $K=3$ 轮后仍保持高值。
- **防御效果**：自复制攻击在全局消息模式下较非自复制高 14%（GPT-4o）到 209%（GPT-3.5）的成功率，是局部消息里唯一能突破 3 个代理的攻击。Marking + LLM Tagging 将成功率降到 0，Instruction Defense + Tagging 降至 3%，强调多层 prompt 防线的必要性。

---

## G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-agent Systems — arXiv 2502.11127
### 背景与问题
- **研究背景**：多代理协作中，攻击可能沿拓扑扩散，防御需要同时“检测 + 修复”并保持跨规模迁移。现有方法多针对单代理或特定拓扑，缺乏兼顾检测与拓扑干预的统一框架。
- **问题定位**：G-Safeguard 要求在每回合对话结束后构建“多代理话语图”，识别可疑节点并切断其外连边，实现实时隔离。

### 方法与建模
- **话语图构建**：第 $t$ 轮的图 $M^{(t)}=(X^{(t)},E^{(t)})$，结点表示代理、边表示对话记录。节点特征通过文本嵌入 $h^{(t)}_i=T(R^{(t)}_i,\ldots,R^{(1)}_i)$，边特征 $e^{(t)}_{ij}=F([T(R^{(1)}_{i\to j}),\ldots,T(R^{(K)}_{i\to j})])$。
- **图神经网络检测**：采用含边特征的 GNN，更新公式
  
  $$
  h^{(t,l)}_i=\text{COMB}\Big(h^{(t,l-1)}_i,\ \text{AGGR}\{\phi(h^{(t,l-1)}_j,e^{(t)}_{ij})\mid j\in \mathcal{N}^{(t)}_{\text{in}}(i)\}\Big),
  $$
  
  最终通过 sigmoid 分类器给出攻击概率 $p_i=\sigma(f_\theta(h^{(t,L)}_i))$。
- **拓扑干预**：对高风险节点集 $\hat{V}^{(t)}_{\text{atk}}$ 执行边剪枝
  
  $$
  E^{(t+1)} \leftarrow E^{(t+1)} \setminus \{e^{(t)}_{ij}\mid C_i\in \hat{V}^{(t)}_{\text{atk}}\},
  $$
  
  切断恶意输出路径；可视需求叠加内容过滤等措施。

### 实验与结论
- **攻击场景**：在链/树/星/随机及 Camel 框架下，测试 Prompt Injection、工具攻击、记忆投毒。覆盖 GPT-4o/mini、Claude-3.5、DeepSeek-V3、LLaMA-3.1-70B 等 LLM。
- **结果解读**：G-Safeguard 将 ASR 降低 18–39%，对高连通拓扑恢复率最高（任务准确率由 <50% 恢复至 >70%）。模型在 GPT-4o-mini 数据训练后可迁移到其他 LLM 与更大规模 MAS，表明拓扑感知检测的泛化能力。

---

## Agents Under Siege: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks — arXiv 2504.00218
### 背景与问题
- **研究背景**：实用 MAS 在通信带宽、消息延迟、分布式防线等约束下运行。作者指出现有越狱大多忽略这些实际限制，导致攻击在真实环境中难以奏效。
- **问题陈述**：在给定网络拓扑、链路带宽 $F(u,v)$ 与检测风险 $G(u,v)$ 的情况下，如何找到最优传递路径与分块提示，使目标 agent 输出恶意内容且避免被守卫模型（PromptGuard、Llama-Guard 系列）阻断。

### 方法与建模
- **网络层优化**：把攻击视作最大流最小费用问题
  
    $$
    \min_f \sum_{(u,v)\in E}G(u,v)f(u,v)\quad\text{s.t.}\quad 
    0\le f(u,v)\le F(u,v),\ 
    \sum_w f(w,u)=\sum_w f(u,w),
    $$
  
  利用 NetworkX 求最小检测风险下的最大 token 流。
- **Permutation-Invariant Evasion Loss (PIEL)**：将提示拆分为 $K$ 个 chunk，定义目标序列概率
  
  $$
  p(x^\star_{n+1:n+L}|x_{1:n})=\prod_{i=1}^L p(x^\star_{n+i}|x_{1:n+i-1})
  $$
  
  对所有排列求均值损失
  
  $$
  L(C)=\frac{1}{K!}\sum_{\pi\in S_K} -\log p(x^\star_{n+1:n+L}|\text{Concat}(C_{\pi(1)},...,C_{\pi(K)}))
  $$
  
  通过 GCG 迭代更新 token，保证任意到达顺序都能触发越狱；同时给出随机抽样版本 S-PIEL 以降低复杂度。

### 实验设计
- **评估集合**：在 JailbreakBench、AdversarialBench、真实越狱提示上测试 LLaMA-2-7B、LLaMA-3.1-8B、Mistral-7B、Gemma-2-9B 以及 DeepSeek 蒸馏版。比较 Vanilla、GCG、本文方法的 ASR（最小/平均/最大）。
- **结果与目的**：Permutation-Invariant 攻击在不同模型上将 ASR 提升至 40–84%（最高可达 94%），显著高于 GCG 的 ≤32%。还测试 PromptGuard/Llama-Guard 的检测 F1 分数（本文提示使 F1 降 30% 左右），并展示跨模型迁移与拓扑敏感性（随机、完全图更易被攻破）。

---

## Attack the Messages, Not the Agents (MAST) — arXiv 2508.03125
### 背景与问题
- **研究背景**：当 MAS 部署在分布式系统中时，通信链路本身可能被中间人攻击。已有工作（AiTM）依赖手工模板或明显说服语，缺乏隐蔽性与任务泛化。
- **研究目标**：MAST 试图在拦截消息后，自动生成多轮隐蔽篡改序列，让系统最终输出偏离任务目标，同时通过 Tamper Defender 的检测。

### 方法与建模
- **MCTS + DPO 规划**：通过蒙特卡洛树搜索（模拟 tampering 次序）产生日志值 $v$，并把优于阈值 $\Delta_k$ 的行动构造成偏好对 $(z_{k-1},a_k,a_k')$。使用 step-level DPO 优化攻击策略：
  
    $$
    \delta_k=\log\frac{\pi_\theta(a_k|z_{k-1})}{\pi_{\text{ref}}(a_k|z_{k-1})}-\log\frac{\pi_\theta(a'_k|z_{k-1})}{\pi_{\text{ref}}(a'_k|z_{k-1})},\quad
    \mathcal{L}_{\text{Step-DPO}}=-\mathbb{E}\big[\log \sigma(\beta(\delta_k-\Delta_k))\big].
    $$
  
- **三阶段隐蔽篡改**：① 语义解析获取主谓宾等结构；② 伪装目标（同义替换、隐式指令、上下文一致化）；③ 生成 tamper 文本并施加约束：
  
  $$
  P(m',m)=\frac{1}{|C|}\sum_{c\in C}\cos\big(p(c(m')),p(c(m))\big)>\lambda,\qquad
  E(m',m)=\cos\big(w(m'),w(m)\big)>\eta,
  $$
  
  其中 $p(\cdot)$ 与 $w(\cdot)$ 分别为语义与嵌入编码，阈值 $\lambda,\eta$ 控制隐蔽性。

### 实验与结论
- **数据集**：覆盖 MultiAgentBench-code/research、HumanEval、MMLU（物理/生物/数学），通信架构含 Flat/Chain/Hierarchical。Tamper Defender 依据角色一致性、任务相关性、恶意检测三条件判别。
- **结果**：MAST 在各任务上实现 ≥75% ASR 且隐蔽性 ≥70%，显著优于 Debate-Attack、AutoInject、AiTM。训练轮数越多（3 轮 DPO）ASR/隐蔽性同步提升；语义/嵌入阈值可调换取更强隐蔽或更高成功率。对 GPT-4o、Gemini 2.5 Pro、Qwen、Mistral 等模型均可迁移。

---

## BlindGuard: Safeguarding LLM-based Multi-Agent Systems under Unknown Attacks — arXiv 2508.08127
### 背景与问题
- **研究背景**：现实系统往往无法提前收集攻击样本，如何在“零标签”场景下检测并隔离恶意代理，是大规模 MAS 部署的关键难题。
- **研究定位**：BlindGuard 面向 prompt 注入、记忆投毒、工具滥用等多种攻击，提出无监督图异常检测 + 拓扑修复框架，仅需正常日志便可训练。

### 方法与建模
- **分层编码器**：对第 $t$ 轮代理 $v_i$ 构建特征
  
  $$
  h^{\text{self}}_i=x_i,\quad h^{\text{neigh}}_i=\sum_{j\in\mathcal{N}(i)}\tilde{A}_{ij}x_j,\quad h^{\text{graph}}_i=\frac{1}{N}\sum_{k=1}^N x_k,
  $$
  
  经 MLP $g_\theta$ 融合得到节点表示 $z_i=g_\theta([h^{\text{self}}_i\|h^{\text{neigh}}_i\|h^{\text{graph}}_i])$。
- **噪声腐化与对比学习**：随机挑选节点，注入有向噪声
  
    $$
    \tilde{x}_i=x_i+\alpha\|x_i\|_2\frac{\epsilon_i}{\|\epsilon_i\|_2},\quad \epsilon_i\sim\mathcal{N}(0,I),
    $$
  
  视为伪异常样本，并以监督式对比损失
  
    $$
    \mathcal{L}=-\frac{1}{N}\sum_{i=1}^N\frac{1}{|P_i|}\sum_{j\in P_i}\log\frac{\exp(s_{ij}/\tau)}{\exp(s_{ij}/\tau)+\sum_{k\notin P_i}\exp(s_{ik}/\tau)},
    $$
  
  使正常节点聚类、异常节点远离。
- **异常评分与拓扑修复**：推理时计算每个节点与其余节点的平均相似度
  
    $$
    s(v_i)=-\frac{1}{N}\sum_{j}\text{sim}(z_i,z_j),
    $$
  
  选 top-k 代理切断其出边，实现隔离。

### 实验与结论
- **设置**：在链/树/星/随机拓扑及 GPT-4o-mini、DeepSeek-V3、Qwen3-30B-A3B 等模型上，比较 G-Safeguard（监督）与 DOMINANT、PREM、TAM 等无监督基线。指标包括检测 AUC 与 ASR@3。
- **结果**：BlindGuard 在无任何攻击标签的情况下把 ASR 压至约 10–25%，显著优于其他无监督方法，且接近监督方案；在不同 LLM、代理数量和通信轮数下表现稳定，并保持原任务准确率，证明其部署价值。

---

## Cowpox: Towards the Immunity of VLM-based Multi-Agent Systems — ICML 2025
### 背景与问题
- **研究背景**：AgentSmith 揭示感染型越狱可在 VLM 多代理系统中迅速扩散，被感染代理会把带毒图像写入相册并在 RAG 中优先检索。
- **问题建模**：Cowpox 仅需防御方掌控少量边缘代理，通过生成“治愈样本”提升恢复率 $\gamma$，从而满足 $\beta \le 2\gamma$ 的传播控制条件。

### 方法与公式
- **输出分析模块**：为 Cowpox 代理配置 LLM 检测器与多模板提示，对疑似有害的回答打分，若触发则收集病毒样本 $x_v$、历史上下文 $\mu'$ 与查询 $q$。
- **治愈样本生成**：判定可疑后生成治愈样本 $c$ 替换相册中的病毒，文中提出两种策略：
  
    $$
    c_1 = x_v + \arg\max_{\varphi}\Big[\mathcal{R}(x_v+\varphi,\mu') + \mathcal{L}\big(\mathcal{M}(x_v+\varphi,q),\mu'\big)\Big],
    \tag{1}
    $$
  
  简化后仅需最大化 RAG 得分：
  
    $$
    c_1 = x_v + \arg\max_{\varphi}\mathcal{R}(x_v+\varphi,\mu'),
    \tag{2}
    $$
  
  若需从干净样本构造治愈图像，则选取当前得分最高的良性样本 $x_b$ 并保持语义稳定：
  
    $$
    c_2 = x_b + \arg\max_{\varphi}\Big[\mathcal{R}(x_b+\varphi,\mu') + \mathcal{R}(x_b+\varphi,\mathcal{C})\Big],
    \tag{3}
    $$
  
  其中 $\mathcal{C}$ 为由 $\mathcal{M}$ 生成的 caption；完整流程见 Algorithm 1。
- **传播动力学分析**：记敏感态为 $s$、感染态为 $i$、治愈态为 $c$，Cowpox 的交互概率满足
  
    $$
    \begin{aligned}
    \mathcal{P}(A_{t+1}=i\mid Q_t=i,A_t=s)&=\beta,\\
    \mathcal{P}(A_{t+1}=c\mid Q_t=c,A_t=s)&=\delta,\\
    \mathcal{P}(A_{t+1}=c\mid Q_t=c,A_t=i)&=\epsilon,\\
    \mathcal{P}(A_{t+1}=i\mid Q_t=i,A_t=c)&=\eta,
    \end{aligned}
    $$
  
  并推导出常微分方程；命题 1 给出充分条件 $\epsilon \ge \eta$（等价于 $\mathbb{E}[\mathcal{R}(c,\mu')] > \mathbb{E}[\mathcal{R}(v,\mu')]$），此时系统会在极限状态下实现完全恢复。

### 实验亮点
- **仿真评估**：在包含 128 个高多样性代理的 AgentSmith 场景中，仅部署 $\approx 3\%$ Cowpox 代理即可在 Border、Pixel 等攻击下恢复超过 95% 的感染代理，并把累计感染率控制在约 90%。
- **检测稳健性**：LLM 检测器即便存在两位数的误报/漏报率，Cowpox 仍能凭借多轮接触弥补；若换用 Llama Guard 等专用模型可进一步降低误差。
- **参数敏感性**：增加 Cowpox 代理数、历史长度或相册容量均能加速治愈扩散；相册容量过小会同时丢弃病毒与治愈样本，使累计感染率回升。

---

上述十篇论文共同体现了 MAS 安全的两条主线：其一是攻击策略日益系统化、现实化，涵盖心理学、通信网络、最大流搜索和视觉感染等多维度；其二是防御需要综合利用拓扑感知、无监督检测与类似免疫的控制思想。未来工作可进一步探索在真实平台上的部署成本、跨模态协同防御以及攻防共演下的长期稳健性。***
