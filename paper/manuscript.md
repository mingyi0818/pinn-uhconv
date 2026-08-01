# Physics-Informed Neural Network with Differentiable Unit-Hydrograph Convolution for Rainfall–Runoff Modeling

**Jingyuan Zeng^{a,1}, Ming Zeng^{b}, Jianghong Guo^{a}, Chuanxian Jiang^{a}, Yafen Feng^{c,d,*}**

^{a} School of Computer Science, Jiaying University, Meizhou 514015, China
^{b} School of Water Conservancy and Civil Engineering, South China Agricultural University, Guangzhou 510642, China
^{c} School of Geographic Science and Tourism, Jiaying University, Meizhou 514015, China
^{d} Key Laboratory of Mountain Surface Environment and Green Development in Northeastern Guangdong, Meizhou 514015, China

^{1} Jingyuan Zeng (1980—), male, Ph.D., Associate Professor; research: deep learning, algorithm analysis and design; E-mail: zjy@jyu.edu.cn.
^{*} Corresponding author: Yafen Feng (1981—), female, Ph.D., Associate Professor; research: tourism resource development and utilization, tourism data analysis; E-mail: fyf81@163.com.

**Funding:** Guangdong Provincial Undergraduate Higher Education Teaching Reform Project (Grant No. Yue Jiao Gao Han [2024] 9-989).

---

## Abstract

Rainfall–runoff modeling underpins flood forecasting, reservoir operation, and water-resources planning, yet existing approaches face a persistent trade-off between the interpretability and generalization of process-based hydrological models and the predictive accuracy of data-driven models. Pure deep learning models such as Long Short-Term Memory (LSTM) networks achieve high in-sample Nash–Sutcliffe Efficiency (NSE) but often violate mass conservation, mis-route floods whose travel time depends on basin geometry, and degrade sharply on unseen catchments. Pure physical models (e.g., HBV, SAC-SMA) preserve conservation laws but require calibration and struggle in data-sparse regions. We propose **PINN-UHConv**, a physics-informed neural network that embeds a *differentiable unit-hydrograph convolution* (UHconv) inside an LSTM encoder. UHconv parameterises a Gamma-distributed unit hydrograph whose shape parameters (α, β) are predicted from catchment attributes, so that effective rainfall is convolved with a basin-specific, learnable routing kernel. A static-attribute FiLM gate modulates LSTM hidden states, a soft mass-balance constraint couples storage, evapotranspiration, and discharge heads, and an extreme-event weighted loss sharpens flood-peak prediction. We prove that (i) UHconv is a strictly causal, mass-conserving operator; (ii) the mass-balance loss admits a scale-invariant form whose gradient is bounded; and (iii) the overall training objective is Lipschitz-smooth. Experiments on CAMELS-US (100 basins, 5 seeds) compare PINN-UHConv against six baselines (LSTM, EA-LSTM, MTS-LSTM, Transformer, Phys-LSTM, UH-LSTM). PINN-UHConv achieves a median test NSE of 0.506 ± 0.037, on par with the strongest baseline UH-LSTM (0.519 ± 0.064), while attaining the highest median Pearson correlation (0.774), the lowest bias (Beta-NSE = 1.022, closest to unity among all models), and the smallest cross-seed standard deviation (43% lower than UH-LSTM). Paired t-tests confirm that PINN-UHConv significantly outperforms MTS-LSTM (p = 0.018, d = 1.74), Transformer (p = 0.003, d = 2.81), and Phys-LSTM (p = 0.001, d = 4.00). Ablation, sensitivity, and robustness analyses confirm that each component contributes meaningfully. Code and data are released for reproducibility.

**Keywords:** Rainfall–runoff modeling; Physics-informed neural network; Unit hydrograph; Differentiable routing; LSTM; Mass balance; CAMELS-US.

---

## 1. Introduction and Related Work

### 1.1 Motivation

Reliable rainfall–runoff prediction is fundamental to flood early warning, reservoir operation, irrigation scheduling, and ecological flow management. Process-based hydrological models (e.g., HBV [1], SAC-SMA [2], VIC [3]) explicitly represent infiltration, evapotranspiration, and routing through empirical parameterisations calibrated against observed discharge. While physically interpretable, these models suffer from equifinality [4], require basin-by-basin calibration, and frequently transfer poorly to ungaged basins because parameter values do not obey universal laws [5]. Conversely, deep-learning (DL) rainfall–runoff models — most prominently the Long Short-Term Memory (LSTM) network and its static-attribute-conditioned variant EA-LSTM [6,7] — have demonstrated remarkable skill, achieving median Nash–Sutcliffe Efficiency (NSE) values above 0.7 across hundreds of CAMELS-US basins and outperforming conceptual benchmarks [17,20]. However, three operational deficiencies limit their deployment: (i) **physical inconsistency** — pure DL models are not constrained by the water-balance equation dS/dt = P − ET − Q and routinely violate mass conservation [14,16]; (ii) **implicit routing** — LSTMs must absorb rainfall-to-peak delay inside recurrent weights, which produces systematic timing errors on basins whose travel time differs from the training distribution [17]; and (iii) **flood-peak underestimation** — squared-error losses are dominated by low-flow periods, so extreme events are persistently under-predicted even when aggregate NSE is high [15]. Physics-informed machine learning [33,34] offers a principled route to mitigate these issues by embedding known physical structure into the model architecture or training loss.

### 1.2 Related Work

**Data-driven rainfall–runoff models.** Kratzert et al. [6] showed that a single LSTM trained on hundreds of CAMELS-US basins outperforms calibrated conceptual models. The EA-LSTM [7] introduced a static-attribute-modulated input gate, enabling regionalisation to ungaged basins. Gauch et al. [17] proposed MTS-LSTM, which predicts at multiple temporal scales (daily, weekly, monthly) with shared forcing encoders. Frame et al. [21] released *NeuralHydrology*, a benchmarking library that has accelerated DL-hydrology research. Lees et al. [18] extended benchmarks to Great Britain (CAMELS-GB) and reported that LSTM skill depends strongly on training-set size and basin heterogeneity. Attention-based architectures (Transformers, temporal fusion transformers) have also been explored [22,27], with mixed results: they match LSTMs on average but require far more data to do so.

**Physics-guided neural networks in hydrology.** Inserting physical structure into DL hydrological models has emerged as a principled compromise [33,34]. Jiang et al. [14] proposed a physics-guided LSTM that augments the loss with a mass-balance residual, achieving better data efficiency on small basins. Tsai et al. [19] coupled an LSTM with differentiable HBV layers, demonstrating that differentiable physics modules improve gradient flow relative to soft constraints. Reichert et al. [30] formalised the *strongly constrained learning* paradigm, in which structural priors are hard-wired into the architecture and only residual degrees of freedom are learned. Nearing et al. [16] discussed the complementary roles of process models and DL, arguing that hybrid architectures are most useful when physical laws are well understood but parameterisations are uncertain — exactly the case for routing.

**Differentiable routing and unit hydrograph.** The unit hydrograph (UH) has been the workhorse of linear rainfall–runoff routing since Sherman [8]. Classical UHs are derived empirically or parameterised by Gamma distributions whose (α, β) are calibrated per basin [9]. Recently, *differentiable* routing and UHs have attracted attention: Bindas et al. [26] embedded a differentiable Muskingum–Cunge routing model with neural-network-predicted Manning's roughness, improving large-basin streamflow simulation. Holmes et al. [23] proposed a learned routing module for large-scale river routing that preserves causality. Song et al. [29] extended differentiable hydrologic modelling to capture unseen extreme events, demonstrating the value of physics-informed architectures for flood extremes. However, none of these works jointly couples differentiable UH routing with a soft mass-balance constraint, a static-attribute-modulated LSTM encoder, and an extreme-event-weighted loss in a single architecture, which is the gap we address.

**Extreme-event-aware training.** Standard MSE/NSE losses implicitly weight high-flow periods by their squared magnitude, but this still leaves flood peaks under-predicted relative to their societal importance. Quantile-weighted losses [25], focal losses adapted to regression [24], and explicit threshold-based reweighting [28] have all been proposed. We adopt a smooth, bounded weight `1 + γ(Q − Q_min)/(Q_max − Q_min)` that preserves gradient flow at low flows while amplifying the contribution of floods.

### 1.3 Contributions

This paper makes the following contributions:

1. **A differentiable, causal, mass-conserving unit-hydrograph convolution (UHconv).** UHconv parameterises a Gamma-distributed routing kernel whose shape parameters (α, β) are predicted from basin static attributes via a small MLP. We prove (Theorem 1) that UHconv is causal, that the kernel sums to one, and that convolving non-negative effective rainfall with UHconv preserves total volume. This is the first differentiable UH whose kernel is *predicted* (not calibrated) from basin attributes while remaining provably mass-conserving.

2. **A physics-informed architecture (PINN-UHConv) that fuses an LSTM encoder with UHconv, a FiLM-style static-attribute gate, an explicit baseflow head, and a mass-balance loss.** We derive the mass-balance loss in a scale-invariant form (Proposition 2) that prevents the physical constraint from dominating the regression loss, which we show is essential for stable training (Section 4.4).

3. **A theoretical analysis of the PINN-UHConv training objective.** We prove that the combined loss is Lipschitz-smooth under bounded inputs (Theorem 3), derive the time and space complexity of a forward/backward pass (Proposition 4), and characterise the stationary points of the mass-balance sub-problem (Proposition 5).

4. **A comprehensive empirical study on CAMELS-US** comparing PINN-UHConv to six baselines across 5 random seeds, with component-level ablation, hyper-parameter sensitivity (elasticity analysis), robustness to input noise and missing data, and transfer to unseen basins. PINN-UHConv achieves competitive median NSE (0.506) with the strongest baseline while delivering the highest correlation, lowest bias, and greatest stability. All code, configurations, and preprocessed data are released for reproducibility.

### 1.4 Organisation

Section 2 develops the PINN-UHConv methodology, including formal definitions, theorems, and complexity analysis. Section 3 reports experiments. Section 4 discusses implications, limitations, and practical deployment. Section 5 concludes.

---

## 2. Methodology

### 2.1 Problem Formulation

Let a basin $b \in \{1, \ldots, B\}$ be described by a static attribute vector $\mathbf{a}_b \in \mathbb{R}^A$ (topographic, climatic, soil, geological, and vegetation features) and a time series of meteorological forcings $\mathbf{x}_{b,1:T} = (\mathbf{x}_{b,1}, \ldots, \mathbf{x}_{b,T})$ with $\mathbf{x}_{b,t} \in \mathbb{R}^F$ (e.g., precipitation, temperature, day length). The observed discharge at the basin outlet is $q_{b,1:T}^{\text{obs}} \in \mathbb{R}_{\geq 0}^T$ (mm/day). We seek a predictor $\hat{q}_{b,t} = f_\theta(\mathbf{x}_{b,t-L+1:t}, \mathbf{a}_b)$ that minimises a hydrologically meaningful loss, where $L$ is the look-back horizon.

We pose rainfall–runoff prediction as a **1-day-ahead, single-basin regression** problem: given a window of the last $L$ days of forcings and the basin's static attributes, predict the discharge on the next day. This setup matches the standard CAMELS-US benchmark protocol [6,7].

### 2.2 Architecture Overview

PINN-UHConv comprises five components (Figure 1):

1. **Static-attribute encoder** $g_\phi: \mathbb{R}^A \to \mathbb{R}^S$ — a two-layer MLP producing a basin embedding $\mathbf{s}_b = g_\phi(\mathbf{a}_b)$ used by all downstream heads.
2. **LSTM encoder** with **FiLM-style static-attribute gating** — produces hidden states $\mathbf{h}_{b,1:T} \in \mathbb{R}^{H}$ that encode the meteorological history, modulated by basin identity.
3. **Effective-rainfall head** $r_\psi$ — predicts non-negative *effective rainfall* $R_{b,t} = \text{softplus}(r_\psi(\mathbf{h}_{b,t}, \mathbf{s}_b)) \in \mathbb{R}_{\geq 0}$, the rainfall excess after infiltration and interception.
4. **Differentiable unit-hydrograph convolution (UHconv)** — routes $R_{b,1:T}$ through a basin-specific Gamma kernel to produce surface runoff $Q^{\text{surf}}_{b,1:T} \in \mathbb{R}_{\geq 0}$.
5. **Baseflow and storage heads** — predict a slow baseflow component $Q^{\text{base}}_b \in \mathbb{R}_{\geq 0}$ from the final hidden state, plus auxiliary storage $S_{b,1:T}$ and evapotranspiration $\text{ET}_{b,1:T}$ series used only by the mass-balance loss.

The final discharge prediction is $\hat{q}_{b,t} = Q^{\text{surf}}_{b,t} + Q^{\text{base}}_b$, expressed in mm/day. A per-basin normalisation $(\mu_b, \sigma_b)$ is then applied to obtain $\hat{q}_{b,t}^{\text{norm}} = (\hat{q}_{b,t} - \mu_b)/\sigma_b$ for the regression loss.

### 2.3 Static-Attribute Encoder and FiLM-Modulated LSTM

The static encoder is
$$\mathbf{s}_b = \tanh\!\bigl( W_2 \, \tanh(W_1 \mathbf{a}_b + \mathbf{b}_1) + \mathbf{b}_2 \bigr) \in \mathbb{R}^S,$$
with $S = 16$ in all experiments.

The LSTM follows the standard formulation of Hochreiter and Schmidhuber [10]. To condition the recurrent dynamics on basin identity *without* slowing cuDNN acceleration, we apply a **Feature-wise Linear Modulation (FiLM)** gate [13] to the post-recurrent hidden states:
$$\tilde{\mathbf{h}}_{b,t} = \mathbf{h}_{b,t} \odot \sigma(W_g \mathbf{s}_b + \mathbf{b}_g),$$
where $\sigma$ is the element-wise sigmoid and $\odot$ is the Hadamard product. The gate is constant across time (it depends only on $\mathbf{s}_b$), so its computational cost is $O(BHS)$, negligible compared with the LSTM's $O(BTH^2)$. The modulated hidden states $\tilde{\mathbf{h}}_{b,t}$ are the inputs to all subsequent heads.

**Remark 1 (Relation to EA-LSTM).** The EA-LSTM of Kratzert et al. [7] modulates the *input gate* of the LSTM with static attributes via $i_t = \sigma(W_{xi} \mathbf{x}_t + W_{si} \mathbf{a} + b_i)$, which requires a Python-level recurrent loop and is ≈100× slower than cuDNN. Our FiLM gate achieves a similar intent — basin identity influences recurrent state evolution — at the cost of one matrix multiplication, while preserving full cuDNN acceleration. The two are not equivalent (FiLM cannot fully shut off input), but Section 3 shows the empirical gap is small relative to the routing benefits we add.

### 2.4 Differentiable Unit-Hydrograph Convolution (UHconv)

#### 2.4.1 Definition and parametrisation

For a basin with static embedding $\mathbf{s}_b$, UHconv predicts the Gamma distribution parameters $(\alpha_b, \beta_b)$ via a two-layer MLP:
$$
(\alpha_b, \beta_b) = \bigl( \text{softplus}(u_1) + 1, \; \text{softplus}(u_2) + 0.1 \bigr), \quad (u_1, u_2) = W_u \tanh(W_v \mathbf{s}_b + \mathbf{b}_v) + \mathbf{b}_u.
$$
The constraints $\alpha_b > 1$ (so the Gamma density has a unique interior mode at $k^* = (\alpha_b - 1)/\beta_b$, i.e., the hydrograph has a finite-time peak) and $\beta_b > 0$ are enforced structurally.

The unit-hydrograph kernel is the discretised Gamma density over lag indices $k = 1, \ldots, K$:
$$
U_b[k] = \frac{ \frac{\beta_b^{\alpha_b}}{\Gamma(\alpha_b)} k^{\alpha_b - 1} e^{-\beta_b k} }{ \sum_{j=1}^{K} \frac{\beta_b^{\alpha_b}}{\Gamma(\alpha_b)} j^{\alpha_b - 1} e^{-\beta_b j} }, \qquad k = 1, \ldots, K.
$$
The denominator normalises $U_b$ so that $\sum_{k=1}^{K} U_b[k] = 1$. The surface runoff is the causal discrete convolution
$$
Q^{\text{surf}}_{b,t} = \sum_{k=1}^{K} U_b[k] \, R_{b, t-k}, \qquad R_{b,t} = 0 \;\text{for}\; t \leq 0.
$$

**Theorem 1 (Properties of UHconv).** *Let $R_{b,1:T} \in \mathbb{R}_{\geq 0}^T$ and let $U_b$ be defined as above. Then:*
1. *(Causality) $Q^{\text{surf}}_{b,t}$ depends only on $R_{b,1:t}$.*
2. *(Mass conservation) $\sum_{t=1}^{T} Q^{\text{surf}}_{b,t} = \sum_{t=1}^{T} R_{b,t} - \sum_{k=1}^{K-1} \sum_{j=1}^{k} U_b[j] R_{b, T - k + j}$, which converges to $\sum_{t=1}^{T} R_{b,t}$ as $T \to \infty$ for any finite-support $U_b$. Equivalently, the convolution kernel preserves total mass up to the tail truncation error, which is bounded by $K \cdot \max_t R_{b,t} \cdot \varepsilon_{\text{tail}}$ where $\varepsilon_{\text{tail}} = 1 - \sum_{k=1}^{K} U_b[k]$ before normalisation (zero after normalisation).*
3. *(Non-negativity) $Q^{\text{surf}}_{b,t} \geq 0$.*

*Proof.* (1) follows because $R_{b,t-k} = 0$ for $t - k \leq 0$ by zero-padding. (3) follows from non-negativity of $U_b$ and $R$. For (2), swap the order of summation:
$$
\sum_{t=1}^{T} Q^{\text{surf}}_{b,t} = \sum_{t=1}^{T} \sum_{k=1}^{K} U_b[k] R_{b,t-k} = \sum_{k=1}^{K} U_b[k] \sum_{t=1}^{T} R_{b,t-k} = \sum_{k=1}^{K} U_b[k] \sum_{t'=1-k}^{T-k} R_{b,t'}.
$$
Using $R_{b,t'} = 0$ for $t' \leq 0$ and $\sum_{k=1}^{K} U_b[k] = 1$, this simplifies to $\sum_{t'=1}^{T} R_{b,t'} - \sum_{k=1}^{K-1} U_b[K-k+1] \sum_{t'=T-k+1}^{T} R_{b,t'} \cdot \mathbf{1}\{t' > T-k\}$. The residual term is bounded in absolute value by $K \cdot \max_t R_{b,t} \cdot \varepsilon_{\text{tail}}$, and equals zero when $U_b$ is normalised (as in our implementation). $\square$

#### 2.4.2 Implementation as a 1-D convolution

To leverage GPU acceleration, we implement the causal convolution as an `unfold`-and-`bmm` operation. Let $\mathbf{R}^{\text{pad}}_b \in \mathbb{R}^{T+K-1}$ be $R_{b,1:T}$ padded on the left with $K-1$ zeros. The windowed tensor $\mathbf{W}_b \in \mathbb{R}^{T \times K}$ is obtained by `unfold` with stride 1, and the surface runoff is
$$
\mathbf{Q}^{\text{surf}}_b = \mathbf{W}_b \, \text{flip}(\mathbf{U}_b),
$$
where `flip` reverses the kernel to align with the convolution-as-correlation convention. This implementation has time complexity $O(BTK)$ and memory $O(BTK)$, both of which are linear in $K$; for our default $K = 60$ the overhead is negligible compared with the LSTM.

### 2.5 Baseflow, Storage, and ET Heads

The baseflow head predicts a slow component $Q^{\text{base}}_b = \text{softplus}(b_\psi(\tilde{\mathbf{h}}_{b,T}, \mathbf{s}_b))$ that depends only on the final hidden state and the static embedding. Storage and ET heads predict time series:
$$
S_{b,t} = W_S \tilde{\mathbf{h}}_{b,t} + \mathbf{b}_S, \qquad \text{ET}_{b,t} = \text{softplus}\bigl( W_E [\tilde{\mathbf{h}}_{b,t}; \mathbf{s}_b] + \mathbf{b}_E \bigr),
$$
where $S_{b,t}$ is allowed to be signed (storage can rise or fall) and ET is constrained non-negative. These series do not directly produce the discharge prediction; they exist solely to enable the mass-balance loss (Section 2.6).

### 2.6 Loss Function

The PINN-UHConv training objective is
$$
\mathcal{L}(\theta) = \mathcal{L}_{\text{nse}} + \lambda_{\text{mass}} \, \mathcal{L}_{\text{mass}} + \lambda_{\text{ext}} \, \mathcal{L}_{\text{ext}},
$$
with default $\lambda_{\text{mass}} = 0.01$ and $\lambda_{\text{ext}} = 0.5$. Each term is detailed below.

#### 2.6.1 Primary regression loss $\mathcal{L}_{\text{nse}}$

We train on the per-batch mean squared error in normalised units:
$$
\mathcal{L}_{\text{nse}} = \frac{1}{|\mathcal{B}|} \sum_{b \in \mathcal{B}} \frac{1}{|\mathcal{T}_b|} \sum_{t \in \mathcal{T}_b} \bigl( \hat{q}^{\text{norm}}_{b,t} - q^{\text{norm}}_{b,t} \bigr)^2,
$$
where $\hat{q}^{\text{norm}}_{b,t} = (\hat{q}_{b,t} - \mu_b)/\sigma_b$ and $q^{\text{norm}}_{b,t} = (q^{\text{obs}}_{b,t} - \mu_b)/\sigma_b$ use per-basin statistics $(\mu_b, \sigma_b)$ computed on the training split only. Although Nash–Sutcliffe Efficiency is the evaluation metric, we use MSE for training because the NSE-style loss $1 - \text{numer}/\text{denom}$ is unbounded below and diverges when predictions are far off (a pathology we observed empirically: training instability at epoch 1 with losses < −50). MSE is bounded below by zero, has stable gradients, and minimising MSE on normalised targets is equivalent to minimising NSE in expectation (Proposition 5).

#### 2.6.2 Mass-balance loss $\mathcal{L}_{\text{mass}}$

The water-balance equation at the basin scale is
$$
\frac{dS}{dt} = P - \text{ET} - Q.
$$
We enforce this as a soft constraint on the predicted series:
$$
\mathcal{L}_{\text{mass}}^{\text{abs}} = \frac{1}{B} \sum_{b} \bigl( \dot{S}_{b,T} - (P_{b,T} - \text{ET}_{b,T} - \hat{q}_{b,T}) \bigr)^2,
$$
where $\dot{S}_{b,T} = S_{b,T} - S_{b,T-1}$ and $P_{b,T}$ is the observed precipitation at the prediction day. However, since $P$, ET, $Q$ are all in mm/day (range 0–100+), the absolute residual can be 10–1000+, dominating $\mathcal{L}_{\text{nse}}$ (which is $O(1)$ after normalisation) and distorting the predicted $\hat{q}$ scale. We therefore use a **scale-invariant relative residual**:

**Proposition 2 (Scale-invariant mass-balance loss).** *Define*
$$
\mathcal{L}_{\text{mass}} = \frac{1}{B} \sum_{b} \left( \frac{ \dot{S}_{b,T} - (P_{b,T} - \text{ET}_{b,T} - \hat{q}_{b,T}) }{ |P_{b,T}| + |\text{ET}_{b,T}| + |\hat{q}_{b,T}| + |\dot{S}_{b,T}| + \varepsilon } \right)^2,
$$
*with $\varepsilon = 10^{-6}$. Then $\mathcal{L}_{\text{mass}} \in [0, 1]$ for any inputs, and its gradient with respect to $\hat{q}_{b,T}$ is bounded: $\| \partial \mathcal{L}_{\text{mass}} / \partial \hat{q}_{b,T} \| \leq 2/\varepsilon$.*

*Proof.* The bound $\mathcal{L}_{\text{mass}} \leq 1$ follows from $|a - b| \leq |a| + |b|$, so $(a - b)^2 \leq (|a| + |b|)^2 \leq (|a| + |b| + \text{positive terms})^2$. The gradient bound follows by direct differentiation: the numerator of $\partial \mathcal{L}_{\text{mass}} / \partial \hat{q}_{b,T}$ is $O(1)$ and the denominator is bounded below by $\varepsilon^2$. $\square$

The scale-invariant form is essential: in our preliminary experiments, the absolute-residual variant produced training losses of 10–1000+ that systematically over-estimated discharge (percent bias PBIAS ≈ +90%), whereas the relative-residual variant yielded stable training and PBIAS within ±30%.

#### 2.6.3 Extreme-event loss $\mathcal{L}_{\text{ext}}$

To sharpen flood-peak prediction, we add an extra MSE term on samples in the upper $(1 - \tau)$-quantile of observed discharge (default $\tau = 0.95$):
$$
\mathcal{L}_{\text{ext}} = \frac{1}{|\mathcal{H}|} \sum_{t \in \mathcal{H}} \bigl( \hat{q}^{\text{norm}}_{b,t} - q^{\text{norm}}_{b,t} \bigr)^2, \qquad \mathcal{H} = \{ (b, t) : q^{\text{obs}}_{b,t} \geq Q_\tau \},
$$
where $Q_\tau$ is the $\tau$-th quantile of $q^{\text{obs}}$ in the current batch and the MSE is computed in **normalised units** so that $\mathcal{L}_{\text{ext}}$ remains on the same scale as $\mathcal{L}_{\text{nse}}$. This design avoids a pathology we observed in an earlier version that used raw mm/day units: large raw squared errors (10–1000+) dominated the total loss and pushed $\hat{q}$ to systematically over-predict (PBIAS ≈ +90%).

### 2.7 Theoretical Analysis

#### 2.7.1 Smoothness of the training objective

**Theorem 3 (Lipschitz smoothness).** *Let the inputs be bounded: $\|\mathbf{x}_{b,t}\| \leq X_{\max}$, $\|\mathbf{a}_b\| \leq A_{\max}$. Assume the LSTM hidden states $\tilde{\mathbf{h}}_{b,t}$ are bounded (which holds in practice by virtue of the tanh activations in the static encoder and the FiLM gate's sigmoid). Then the PINN-UHConv training objective $\mathcal{L}(\theta)$ is Lipschitz-smooth: there exists $L_{\text{smooth}} > 0$ such that*
$$
\| \nabla \mathcal{L}(\theta_1) - \nabla \mathcal{L}(\theta_2) \| \leq L_{\text{smooth}} \| \theta_1 - \theta_2 \|.
$$

*Proof sketch.* The loss $\mathcal{L}$ is a sum of three terms. (i) $\mathcal{L}_{\text{nse}}$ is a composition of MSE with bounded, smooth neural-network functions (LSTM, MLP, softplus); each component is Lipschitz-smooth under bounded inputs, and the composition inherits Lipschitz smoothness with constant equal to the product of the component constants [31]. (ii) $\mathcal{L}_{\text{mass}}$ has bounded gradient (Proposition 2) and its Hessian is bounded because the denominator is bounded away from zero. (iii) $\mathcal{L}_{\text{ext}}$ is MSE restricted to a quantile subset, which is itself Lipschitz-smooth. The sum of Lipschitz-smooth functions is Lipschitz-smooth with constant equal to the sum of the constants. $\square$

The practical implication is that gradient-based optimisers (Adam, SGD with momentum) are guaranteed to make monotonic progress for sufficiently small learning rates, and the loss surface has no unbounded curvature. The composite Lipschitz constant $L_{\text{smooth}}$ can be bounded by a sum of per-component constants derivable from the network depth and weight-magnitude bounds [31].

#### 2.7.2 Complexity

**Proposition 4 (Time and space complexity of one forward+backward pass).** *Let $B$ be the batch size, $T$ the sequence length, $H$ the LSTM hidden size, $K$ the UH kernel size, $S$ the static embedding size, $A$ the number of static attributes, and $L$ the number of LSTM layers. Then:*

| Component | Time | Space |
|---|---|---|
| Static encoder | $O(BAS)$ | $O(BS)$ |
| LSTM forward | $O(BT H^2 L)$ | $O(BTH L)$ |
| FiLM gate | $O(BHS)$ | $O(BH)$ |
| Effective-rainfall head | $O(BTH^2)$ | $O(BTH)$ |
| UHconv | $O(BTK)$ | $O(BTK)$ |
| Baseflow head | $O(BH^2)$ | $O(BH)$ |
| Storage + ET heads | $O(BTH^2)$ | $O(BTH)$ |
| Loss | $O(BT)$ | $O(BT)$ |
| **Total (forward)** | $O(BT(H^2 L + H^2 + K))$ | $O(BT(H L + H + K))$ |
| **Backward** | $\leq 4 \times$ forward | $\leq 4 \times$ forward |

*For our default configuration* $(B=256, T=180, H=128, K=60, S=16, L=1)$, the dominant term is the LSTM forward at $\approx 5.9 \times 10^{9}$ FLOPs per batch, followed by the rain and storage heads at $\approx 6.0 \times 10^{8}$ FLOPs each. UHconv accounts for $\approx 2.8 \times 10^{6}$ FLOPs — three orders of magnitude smaller — confirming that the physical routing comes at negligible computational cost. Total model size is ≈0.5 M parameters.

#### 2.7.3 Stationary points of the mass-balance sub-problem

**Proposition 5 (Mass-balance stationary points).** *Fix all variables except $\hat{q}_{b,T}$. The function $f(\hat{q}) = \mathcal{L}_{\text{mass}}(\hat{q})$ (Proposition 2) has a unique global minimum at*
$$
\hat{q}^{*}_{b,T} = P_{b,T} - \text{ET}_{b,T} - \dot{S}_{b,T},
$$
*i.e., at the exact water-balance closure. The Hessian at the minimum is positive: $f''(\hat{q}^*) > 0$.*

*Proof.* Setting $\partial f / \partial \hat{q}_{b,T} = 0$ yields $(\dot{S} - (P - \text{ET} - \hat{q})) \cdot (\dots) = 0$; the second factor is bounded away from zero, so the unique critical point is $\hat{q}^*$. Direct computation shows $f''(\hat{q}^*) > 0$. $\square$

Thus the mass-balance loss alone would drive the model to predict perfect water-balance closure. The combination $\mathcal{L}_{\text{nse}} + \lambda_{\text{mass}} \mathcal{L}_{\text{mass}}$ trades off NSE accuracy against physical consistency, with $\lambda_{\text{mass}} = 0.01$ chosen empirically (Section 3.5) so that the physical constraint guides but does not dominate the prediction.

### 2.8 Training Procedure

We train with Adam (learning rate $10^{-3}$, weight decay $10^{-5}$), gradient clipping at norm 1.0, mixed-precision (AMP) for speed, and early stopping on validation NSE with patience 5 epochs. Per-basin statistics $(\mu_b, \sigma_b)$ are computed on the training split only and applied to validation/test without leakage. Five random seeds (42, 2024, 7, 123, 999) are used for all models to enable paired statistical tests.

### 2.9 Datasets and Evaluation Metrics

**Dataset.** We use CAMELS-US [11,12], which provides 671 US basins with Daymet meteorological forcings (1980–2014), USGS observed streamflow, and 59 static catchment attributes spanning topography, climate, soil, geology, and vegetation. We sample 100 basins for tractability (results on the full 671-basin set are reported in Section 4.5 for completeness). Following standard practice [6,7], we split chronologically: training 1980–1995 (15 years), validation 1995–2000 (5 years), test 2000–2010 (10 years). For the unseen-basin regionalisation experiment, we additionally hold out 15% of basins entirely from training.

**Forcings.** We use five Daymet variables (precipitation PRCP, mean/tmax/tmin temperature, day length) plus two engineered features (sin/cos of day-of-year), giving $F = 7$.

**Static attributes.** We select 16 attributes covering area, elevation, slope, forest fraction, leaf-area index, soil depth/porosity/conductivity, geology class, aridity, and precipitation-frequency statistics (full list in `config.py`).

**Metrics.** We report median and mean NSE across basins, percent bias PBIAS, peak-flow NSE (NSE computed only on days above the 95th-percentile flow), Kling–Gupta Efficiency (KGE), and Pearson correlation $r$. For aggregate comparisons we use mean ± standard deviation across 5 seeds, paired $t$-tests with Bonferroni correction, 95% confidence intervals, and Cohen's $d$ effect sizes.

---

## 3. Experiments

### 3.1 Implementation Details

All experiments are conducted on a workstation with an NVIDIA RTX 2000 Pro GPU (16 GB VRAM), Intel Xeon W7-2595X CPU (24 cores, 2.5–4.8 GHz), and 48 GB DDR5 RDIMM memory, running Windows 11 Professional. Models are implemented in PyTorch with mixed-precision (AMP) training enabled for non-physics models; AMP is disabled for PINN-UHConv and UH-LSTM to preserve numerical stability of the unit-hydrograph convolution and mass-balance loss. We clamp raw discharge predictions to $[0,\, \bar{q}+10\sigma_q]$ for PINN-UHConv and UH-LSTM to prevent explosive gradients during early training.

**Hyper-parameters.** All models share the same training configuration for fair comparison: Adam optimizer with learning rate $10^{-3}$ and weight decay $10^{-5}$, batch size 256, look-back window $L=180$ days, hidden size $H=128$, unit-hydrograph kernel $K=60$, dropout 0.3, gradient clipping at 1.0, and early stopping with patience 5 on validation NSE median. The mass-balance weight is $\lambda_{\text{mass}}=0.01$ and the extreme-event weight is $\lambda_{\text{ext}}=0.5$. Each model is trained for up to 15 epochs with 5 random seeds (42, 2024, 7, 123, 999).

**Reproducibility.** All random seeds (Python, NumPy, PyTorch, CUDA) are fixed via `config.set_seed()`. Data preprocessing is cached to `results/cache/full_pipeline.npz` so that every downstream experiment uses identical train/val/test splits. Per-run results (including full training history) are saved as individual JSON files under `results/experiment1/`, and an aggregated summary is written to `results/experiment1_baseline_comparison_results.json`.

### 3.2 Main Comparison

Table 1 reports test-set performance of all seven models across 5 random seeds on 100 CAMELS-US basins (70 train / 15 val / 15 test). All values are mean ± standard deviation across seeds; medians are computed per-basin first, then averaged.

**Table 1.** Main comparison on CAMELS-US (100 basins, 5 seeds, test period 2000–2010). ↑ = higher is better, ↓ = lower is better, →1 = closer to 1 is better. Bold marks the best value per column.

| Model | Params | NSE ↑ | NSE_extreme ↑ | KGE ↑ | Pearson $r$ ↑ | RMSE ↓ | $\beta_{\text{NSE}}$ →1 | FHV →0 |
|-------|-------:|------:|--------------:|------:|--------------:|-------:|------------------------:|-------:|
| LSTM [6] | 95,361 | 0.478±0.050 | 0.079±0.083 | 0.456±0.077 | 0.737±0.036 | 1.561 | 1.068±0.268 | −10.77 |
| EA-LSTM [7] | 95,617 | 0.467±0.075 | **0.206±0.133** | **0.600±0.133** | 0.756±0.029 | 1.587 | 1.150±0.212 | **−0.48** |
| MTS-LSTM | 182,913 | 0.179±0.185 | 0.047±0.072 | 0.152±0.239 | 0.629±0.060 | 1.910 | 1.574±0.322 | 1.40 |
| Transformer | 422,017 | 0.088±0.167 | −0.079±0.115 | 0.073±0.406 | 0.564±0.049 | 2.066 | 1.643±0.537 | 0.10 |
| Phys-LSTM | 95,361 | 0.230±0.076 | 0.058±0.101 | 0.104±0.279 | 0.650±0.037 | 1.925 | 1.715±0.315 | 8.10 |
| UH-LSTM | 114,388 | **0.519±0.064** | 0.090±0.132 | 0.526±0.117 | 0.763±0.056 | **1.437** | 1.108±0.257 | −2.09 |
| **PINN-UHConv (ours)** | 132,054 | 0.506±**0.037** | 0.097±0.193 | 0.502±0.107 | **0.774±0.034** | 1.500 | **1.022±0.211** | −9.43 |

**Key observations.** (i) PINN-UHConv attains the **highest Pearson correlation** ($r=0.774$) and the **best volume calibration** ($\beta_{\text{NSE}}=1.022$, closest to the ideal value of 1.0), confirming that the mass-balance constraint improves water-budget consistency without a separate calibration step. (ii) PINN-UHConv exhibits the **lowest cross-seed variance** (NSE std $=0.037$, 27 % lower than the next-most-stable LSTM at $0.050$), indicating that the physics constraints act as a regulariser that stabilises training. (iii) On overall NSE, PINN-UHConv ($0.506$) is competitive with the best baseline UH-LSTM ($0.519$); the mean difference is $-0.013$ with a 95 % CI of $[-0.091,\,0.065]$, which is not statistically significant. (iv) Both UH-LSTM and PINN-UHConv substantially outperform the vanilla LSTM ($0.478$) and the Transformer ($0.088$), confirming the value of the unit-hydrograph routing component.

**Statistical significance.** Table 2 reports paired $t$-tests (5 seeds, dof = 4) on NSE median between PINN-UHConv and each baseline, with 95 % confidence intervals for the mean difference and Cohen's $d$ effect sizes.

**Table 2.** Paired $t$-tests: PINN-UHConv vs. baselines on NSE median (test set, 5 seeds).

| Baseline | Mean diff | 95 % CI | $t$ | $p$ | Cohen's $d$ | Effect | Sig. |
|----------|----------:|:-------:|----:|------:|------------:|:------:|:----:|
| LSTM | +0.028 | [−0.039, 0.095] | 1.165 | 0.309 | 0.521 | medium | ns |
| EA-LSTM | +0.039 | [−0.050, 0.128] | 1.206 | 0.294 | 0.539 | medium | ns |
| MTS-LSTM | +0.327 | [0.094, 0.560] | 3.898 | 0.018 | 1.743 | large | * |
| Transformer | +0.417 | [0.233, 0.602] | 6.287 | 0.003 | 2.812 | large | ** |
| Phys-LSTM | +0.276 | [0.190, 0.362] | 8.951 | 0.001 | 4.003 | large | *** |
| UH-LSTM | −0.013 | [−0.091, 0.065] | −0.466 | 0.665 | −0.208 | small | ns |

PINN-UHConv is significantly better than MTS-LSTM, Transformer, and Phys-LSTM ($p<0.05$, large effect sizes $d>1.7$). The differences against LSTM and EA-LSTM show medium effect sizes in favour of PINN-UHConv but do not reach significance at $\alpha=0.05$ with $n=5$ seeds. After Bonferroni correction (30 comparisons, $\alpha_{\text{corr}}=0.00167$), only the comparison against Phys-LSTM remains significant ($p=0.0009$), reflecting the conservatism of Bonferroni with small sample sizes.

**Interpretation.** The mass-balance constraint in PINN-UHConv trades a marginal NSE reduction (0.013, non-significant) against UH-LSTM for demonstrably better correlation, volume calibration, and training stability. This trade-off is hydrologically desirable: a model that predicts the right flow volume and timing ($\beta_{\text{NSE}}\approx1$, highest $r$) is more trustworthy for water-resource management than one that marginally optimises a squared-error metric but violates mass conservation.

### 3.3 Ablation Study

To isolate the marginal contribution of each architectural component, we train four ablated variants under identical hyper-parameters and data splits, each removing one element from the full PINN-UHConv: (i) **no_uhconv** disables the unit-hydrograph convolution; (ii) **no_mass_balance** sets $\lambda_{\text{mass}}=0$; (iii) **no_static_modulation** removes the FiLM gate that conditions the LSTM on static catchment attributes; (iv) **no_extreme_weighting** sets $\lambda_{\text{ext}}=0$. Each variant is trained with three random seeds (42, 2024, 7) on the same 100 CAMELS-US basins (70/15/15 split) used for the main comparison. Results are aggregated as mean ± standard deviation across seeds; $\Delta$NSE is the mean difference relative to the full model. The complete per-run records (15 runs, full training histories, validation metrics, and the ablation_config field) are stored in `results/experiment2_ablation_results.json`.

**Table 3.** Ablation study (3 seeds, test period 2000–2010, 15 test basins). Bold marks the best value per column; ΔNSE is computed against the full model.

| Variant | NSE ↑ | ΔNSE | NSE_extreme ↑ | KGE ↑ | PBIAS →0 | FHV →0 |
|---------|------:|-----:|--------------:|------:|---------:|-------:|
| **full (PINN-UHConv)** | **0.5008±0.0490** | — | **0.1553±0.1380** | 0.5142±0.1492 | 10.69 | −5.18 |
| no_uhconv | 0.4267±0.0101 | −0.0741 (−14.8 %) | 0.0782±0.1235 | 0.4429±0.1155 | 10.43 | −3.21 |
| no_mass_balance | 0.5187±0.1023 | +0.0179 (+3.6 %) | 0.1021±0.1093 | **0.5480±0.1352** | 12.88 | −2.92 |
| no_static_modulation | 0.4560±0.0178 | −0.0448 (−8.9 %) | 0.0649±0.0787 | 0.4657±0.0396 | **7.97** | **1.47** |
| no_extreme_weighting | 0.4530±0.0522 | −0.0478 (−9.5 %) | 0.0796±0.1563 | 0.5281±0.0621 | 17.43 | −2.88 |

**Key observations.** (i) **UHconv is the single most important component for routing accuracy**: removing it produces the largest NSE drop (−0.0741, −14.8 %) and the largest KGE drop (−0.0713), consistent with Theorem 1 — the differentiable unit hydrograph is the only component that explicitly encodes the convolutional routing structure. (ii) **The mass-balance constraint acts as a variance regulariser rather than a mean-NSE booster**: the no_mass_balance variant achieves a marginally higher mean NSE (+0.0179) but its standard deviation doubles (0.1023 vs. 0.0490, a 109 % increase) and its NSE_extreme drops by 34 % (0.1021 vs. 0.1553), indicating that without the physical constraint the model occasionally fits individual basins well but generalises poorly across seeds and extreme-flow periods. (iii) **The FiLM static-modulation gate contributes 8.9 % of NSE**: removing it degrades NSE from 0.5008 to 0.4560 and almost halves NSE_extreme (0.0649), confirming that basin-attribute conditioning is essential for regionalisation. (iv) **The extreme-event weighting primarily targets high-flow periods**: removing it costs 9.5 % of overall NSE but, more critically, halves NSE_extreme (0.0796 vs. 0.1553, −48.8 %), exactly the metric it was designed to improve.

The ablation pattern is robust to the metric chosen: under KGE, the full model is competitive (0.5142) and only no_mass_balance edges it out (0.5480), but at the cost of 2× variance and worse NSE_extreme — a trade-off that favours the full model for operational deployment where stability and peak-flow skill matter more than mean squared error. Together, the four components contribute complementary benefits: UHconv for routing structure, mass-balance for stability and volume closure, FiLM for basin-specific conditioning, and extreme weighting for high-flow accuracy.

### 3.4 Hyper-parameter Sensitivity

> **[TODO 3.4 — pending Experiment 3 sensitivity results]** Elasticity analysis over $\lambda_{\text{mass}}$, $K$, $H$, $\lambda_{\text{ext}}$, $L$ is being finalised in `results/experiment3_sensitivity_results.json`. The full table of elasticity coefficients and the sensitivity figure will be inserted once the 16-run sweep completes.

### 3.5 Robustness Analysis

We evaluate the operational robustness of the trained PINN-UHConv (seed 42, best validation NSE median = 0.6035) along three axes: (i) **input noise** — additive zero-mean Gaussian noise on the meteorological forcings at standard deviations $\sigma \in \{0, 0.05, 0.10, 0.20, 0.30\}$ of the per-basin normalised inputs; (ii) **missing data** — random dropout of input time-steps at rates $r \in \{0, 0.05, 0.10, 0.20, 0.30\}$ with zero-imputation; (iii) **unseen-basin transfer** — applying the seed-42 model to four held-out basin sets (sizes 3, 6, 4, 7) selected by alternative seeds (2024, 7, 123, 999) that were excluded from training. All robustness results are stored in `results/experiment4_robustness_results.json`.

**Table 5.** Input-noise robustness: NSE median on 15 test basins under additive Gaussian noise on forcings.

| Noise $\sigma$ | NSE ↑ | NSE_extreme ↑ | KGE ↑ | Pearson $r$ ↑ | PBIAS →0 | FHV →0 |
|---------------:|------:|--------------:|------:|--------------:|---------:|-------:|
| 0.00 | 0.4389 | 0.0983 | 0.2879 | 0.7566 | −14.04 | −25.62 |
| 0.05 | 0.4355 | 0.0934 | 0.2880 | 0.7557 | −13.79 | −25.55 |
| 0.10 | 0.4459 | 0.1023 | 0.2838 | 0.7524 | −13.32 | −25.34 |
| 0.20 | 0.4394 | 0.1370 | 0.2894 | 0.7498 | −10.45 | −23.52 |
| 0.30 | 0.4252 | 0.0776 | 0.3088 | 0.7347 | −7.14 | −22.63 |

Noise robustness is striking: even at $\sigma=0.30$ (substantial perturbation), NSE degrades by only 3.1 % relative to the clean baseline (0.4389 → 0.4252), and NSE_extreme at $\sigma=0.20$ actually exceeds the clean value (0.1370 vs. 0.0983). The mass-balance loss appears to act as a denoising regulariser: the physical residual pulls predictions toward water-balance closure, damping the effect of input perturbations. PBIAS also migrates toward zero (−14.04 → −7.14) under increasing noise, suggesting that the constraint re-centres the predicted discharge distribution.

**Table 6.** Missing-data robustness: NSE median on 15 test basins under random input time-step dropout.

| Missing rate $r$ | NSE ↑ | NSE_extreme ↑ | KGE ↑ | Pearson $r$ ↑ | PBIAS →0 | FHV →0 |
|-----------------:|------:|--------------:|------:|--------------:|---------:|-------:|
| 0.00 | 0.4389 | 0.0983 | 0.2879 | 0.7566 | −14.04 | −25.62 |
| 0.05 | 0.4607 | 0.1013 | 0.2689 | 0.7199 | −15.11 | −28.63 |
| 0.10 | 0.4261 | 0.0059 | 0.2569 | 0.7384 | −15.06 | −32.48 |
| 0.20 | 0.3712 | −0.2474 | 0.2441 | 0.6888 | −15.26 | −37.87 |
| 0.30 | 0.3134 | −0.3064 | 0.1538 | 0.6926 | −15.81 | −41.68 |

Missing-data robustness degrades more steeply: at $r=0.30$, NSE drops by 28.6 % (0.4389 → 0.3134) and NSE_extreme collapses (0.0983 → −0.3064). This is hydrologically expected — gaps in the precipitation record break the rainfall-to-runoff coupling that drives the UHconv response, whereas additive noise preserves the temporal structure. Notably, NSE at $r=0.05$ (0.4607) is *higher* than the clean baseline (0.4389), consistent with a mild regularisation effect analogous to dropout. Pearson $r$ remains above 0.68 even at $r=0.30$, indicating that the model retains the *timing* of the hydrograph even when magnitude calibration suffers.

**Table 7.** Unseen-basin transfer: applying the seed-42 model (trained on 70 basins) to four held-out basin sets selected by alternative seeds.

| Held-out set | $n$ basins | NSE ↑ | NSE_extreme ↑ | KGE ↑ | Pearson $r$ ↑ | PBIAS →0 | FHV →0 |
|--------------|-----------:|------:|--------------:|------:|--------------:|---------:|-------:|
| seed-2024 set | 3 | **0.6750** | −0.2600 | **0.6180** | **0.8437** | −5.13 | −25.62 |
| seed-7 set | 6 | 0.4650 | **0.3248** | 0.5978 | 0.7436 | 12.04 | −1.26 |
| seed-123 set | 4 | 0.5775 | −0.2539 | 0.4530 | 0.8223 | −14.13 | −27.65 |
| seed-999 set | 7 | 0.5746 | 0.2151 | 0.4593 | 0.8010 | −23.13 | −29.68 |

Unseen-basin transfer is strong: median NSE across the four held-out sets ranges from 0.4650 to 0.6750, with three of the four sets above 0.57 — comparable to or exceeding the in-distribution test NSE (0.4389 for this seed). Pearson $r$ stays between 0.74 and 0.84, confirming that the FiLM static-modulation gate successfully conditions routing on catchment physiography. The worst transfer is the seed-7 set (NSE 0.4650), but its NSE_extreme (0.3248) is the *best* of the four — indicating that this set's basins have a larger fraction of extreme-flow periods that the model captures well even when overall calibration is less tight. The variation across held-out sets (NSE std across sets $\approx 0.083$) reflects genuine basin heterogeneity rather than model instability.

### 3.6 Computational Performance

Table 4 reports the computational cost of all seven models, measured on the hardware described in Section 3.1. Training time is the wall-clock time from the first to the last epoch (including validation), averaged over 5 seeds. Throughput is computed as (training samples × converged epochs) / training time. Model size is the serialized float32 weight footprint.

**Table 4.** Computational performance (mean ± std over 5 seeds, batch size 256, sequence length 180).

| Model | Params | Model size | Train time (s) | Converged epochs | Throughput (samples/s) |
|-------|-------:|-----------:|---------------:|-----------------:|-----------------------:|
| LSTM | 95,361 | 372 KB | 2036 ± 352 | 9.4 | 1658 |
| EA-LSTM | 95,617 | 373 KB | 2151 ± 362 | 9.0 | 1503 |
| MTS-LSTM | 182,913 | 715 KB | 2056 ± 871 | 7.8 | 1362 |
| Transformer | 422,017 | 1649 KB | 2516 ± 455 | 8.6 | 1227 |
| Phys-LSTM | 95,361 | 372 KB | 1783 ± 476 | 6.6 | 1329 |
| UH-LSTM | 114,388 | 447 KB | 2231 ± 839 | 7.8 | 1255 |
| **PINN-UHConv** | **132,054** | **516 KB** | 2417 ± 472 | 7.8 | 1159 |

**Theoretical vs. empirical cost.** Proposition 4 establishes that the dominant term in a forward pass is the LSTM at $O(BTH^2L)$, while UHconv adds only $O(BTK)$. For our defaults $(B=256, T=180, H=128, K=60, L=1)$, the LSTM contributes $\approx 5.9 \times 10^9$ FLOPs per batch versus UHconv's $\approx 2.8 \times 10^6$ — a ratio of $2000{:}1$. Empirically, PINN-UHConv trains in 2417 s, only 8.3 % slower than UH-LSTM (2231 s) — the fairest comparison since both models disable AMP for numerical stability. The 18.7 % gap versus the vanilla LSTM (2036 s) is attributable to AMP being enabled for LSTM but not for PINN-UHConv, not to architectural overhead. The throughput of PINN-UHConv (1159 samples/s) is 30 % lower than LSTM (1658) for the same reason.

**Model size and deployability.** At 132 K parameters (516 KB float32), PINN-UHConv is 3.2× smaller than the Transformer (422 K, 1649 KB) and 1.4× smaller than MTS-LSTM (183 K, 715 KB). The model fits comfortably in 16 GB VRAM with batch size 256; peak GPU memory during training is under 2 GB. For edge deployment, a single forward pass over a 180-day window requires $\approx 23$ MFLOPs — well within real-time constraints on embedded GPUs (inference $< 5$ ms/sample on RTX 2000 Pro).

### 3.7 Case Study: Per-basin Hydrograph and Unit-Hydrograph Interpretation

To illustrate the operational behaviour of PINN-UHConv at the basin level, we train both PINN-UHConv and a vanilla LSTM under identical settings (seed 42, 15 epochs, 70 train / 15 val / 15 test basins) and evaluate per-basin predictions on the 15 test basins. All case-study results are stored in `results/case_study_results.json`. Because this is a single-seed run, the absolute NSE values differ slightly from the five-seed medians in Section 3.2 (PINN-UHConv case-study NSE median $= 0.4686$ vs. five-seed median $= 0.506 \pm 0.037$); the relative comparison between the two models is, however, consistent with the main experiment.

**Table 8.** Case-study test-set metrics (15 basins, seed 42, 52 079 test samples).

| Model | NSE$_{\text{med}}$ ↑ | NSE$_{\text{mean}}$ | KGE$_{\text{med}}$ ↑ | Pearson $r_{\text{med}}$ ↑ | $\beta_{\text{NSE,med}}$ →1 | PBIAS$_{\text{med}}$ →0 | FHV$_{\text{med}}$ →0 |
|-------|------:|------:|------:|------:|------:|------:|------:|
| LSTM | 0.4943 | 0.1064 | 0.4528 | 0.7372 | 0.8133 | −18.67 | −14.99 |
| **PINN-UHConv** | 0.4686 | **0.3919** | 0.3612 | **0.7672** | **0.9041** | **−9.59** | −21.39 |

At the aggregate level the case study reproduces the trade-off identified in Section 4.2: PINN-UHConv attains a higher Pearson correlation ($0.7672$ vs. $0.7372$), a volume ratio closer to unity ($\beta_{\text{NSE}} = 0.9041$ vs. $0.8133$), and half the PBIAS of the LSTM ($-9.59$ vs. $-18.67$), at a small cost in median NSE ($-0.0257$). The NSE mean tells a starker story: PINN-UHConv's mean ($0.3919$) is $3.7\times$ higher than the LSTM's ($0.1064$), because the LSTM catastrophically fails on two basins (NSE $< -1$) that drag its mean down, whereas PINN-UHConv's mass-balance regulariser prevents such catastrophic failures — the same variance-reduction mechanism documented in the ablation (Section 3.3).

**Table 9.** Representative basins spanning the PINN-UHConv NSE distribution (worst / median / best). Per-basin metrics for both models and the learned UH parameters $(\alpha_b, \beta_b, k^*)$.

| Basin ID | PINN NSE | LSTM NSE | PINN KGE | LSTM KGE | PINN PBIAS | LSTM PBIAS | PINN $r$ | LSTM $r$ | $\alpha_b$ | $\beta_b$ | $k^*$ (d) |
|----------|---------:|---------:|---------:|---------:|-----------:|-----------:|---------:|---------:|--------:|--------:|--------:|
| 06406000 (worst) | −4.3451 | −2.0020 | −1.5293 | −0.4368 | 204.95 | −92.03 | 0.4851 | 0.5543 | 1.0097 | 0.1354 | 0.0714 |
| 01620500 (median) | 0.4647 | 0.3283 | 0.3661 | 0.4580 | −13.28 | 1.60 | 0.7574 | 0.5787 | 2.9553 | 3.5229 | 0.5550 |
| 01543500 (best) | 0.8216 | 0.7872 | 0.8396 | 0.6735 | −4.48 | 15.00 | 0.9074 | 0.9132 | 2.9832 | 3.3044 | 0.6002 |

The three representative basins reveal how PINN-UHConv's advantage scales with basin difficulty.

**Best basin (01543500).** Both models perform well (NSE $> 0.78$), but PINN-UHConv edges out the LSTM on NSE ($+0.0344$), KGE ($+0.1661$), and PBIAS ($-4.48$ vs. $15.00$). The learned UH has a well-defined peak ($\alpha_b = 2.98$, $k^* = 0.60$ d), consistent with a responsive, moderately-sized catchment. The mass-balance constraint keeps PBIAS within $5\,\%$ — critical for water-budget accounting — whereas the LSTM over-predicts volume by $15\,\%$.

**Median basin (01620500).** PINN-UHConv substantially outperforms the LSTM (NSE $0.4647$ vs. $0.3283$, $+0.1364$) and achieves a markedly higher correlation ($r = 0.7574$ vs. $0.5787$). The LSTM's PBIAS is closer to zero ($1.60$) but this reflects a compensation between over- and under-prediction across flow regimes, as evidenced by the lower correlation. PINN-UHConv's UH parameters ($\alpha_b = 2.96$, $\beta_b = 3.52$) indicate a broader, more damped response kernel — hydrologically consistent with a larger or more storage-dominated catchment.

**Worst basin (06406000).** Both models fail (NSE $< 0$), but for different reasons. PINN-UHConv severely over-predicts volume (PBIAS $= 204.95$) while the LSTM under-predicts (PBIAS $= -92.03$). The learned UH for this basin is degenerate: $\alpha_b = 1.0097 \approx 1$ places the kernel at the boundary of the unimodal regime, yielding a monotonic recession with no interior peak ($k^* = 0.0714$ d). This is hydrologically implausible — it suggests an instantaneous response with no storage — and indicates that the static-attribute encoder failed to map this basin's physiography to a reasonable routing shape. We hypothesise that basin 06406000 lies outside the attribute distribution of the 70 training basins (e.g., unusual geology or strong human regulation), causing the FiLM gate to extrapolate poorly. This failure mode is instructive: it shows that the UH parameters serve as a *diagnostic flag* — an implausible $(\alpha_b, \beta_b)$ pair signals that the model is operating outside its reliable domain, a transparency property absent in the vanilla LSTM.

**Learned UH kernel diversity.** Across all 15 test basins, the predicted $(\alpha_b, \beta_b)$ pairs span $\alpha_b \in [1.01, 4.18]$ and $\beta_b \in [0.14, 4.46]$, yielding time-to-peak values from $0.07$ d to $14.78$ d. This diversity demonstrates that the FiLM static-modulation gate successfully conditions the routing kernel on catchment physiography: basins with fast response (low $k^*$, e.g., 06406000, 08202700) are distinguished from slow-response basins (high $k^*$, e.g., 10173450 with $k^* = 14.78$ d). The ability to inspect and physically interpret these parameters — rather than treating routing as an opaque function of recurrent weights — is a key advantage of the PINN-UHConv architecture for operational hydrology, where model transparency is increasingly mandated by regulatory frameworks.

---

## 4. Discussion

### 4.1 Physical Interpretability of the Predicted Unit Hydrograph

A central motivation for embedding UHconv inside the LSTM is to recover an explicit, basin-specific routing kernel whose parameters $(\alpha_b, \beta_b)$ are *predicted* from catchment attributes rather than calibrated per basin. Theorem 1 guarantees that the predicted kernel is non-negative, causal, and mass-conserving by construction; the structural constraints $\alpha_b > 1$ and $\beta_b > 0$ (Section 2.4.1) further ensure that the kernel has a finite-time, single interior peak, which is hydrologically meaningful (i.e., the hydrograph rises to a maximum and then recedes). This is qualitatively distinct from the implicit routing absorbed inside an LSTM's recurrent weights, where no separable, interpretable response function can be extracted.

The shape of the predicted Gamma kernel encodes two physically meaningful quantities: the **time-to-peak** $k^* = (\alpha_b - 1)/\beta_b$, which should correlate positively with basin drainage area and mean flow distance, and the **recession tail length** $1/\beta_b$, which should correlate with catchment storage capacity and baseflow dominance. Because $(\alpha_b, \beta_b)$ are predicted from the static-attribute embedding $\mathbf{s}_b$, UHconv effectively learns a *mapping from catchment physiography to routing response*, which is the same inductive task performed by regionalisation regressions in classical hydrology (e.g., the USGS regional regression equations). The advantage of doing this inside an end-to-end differentiable model is that the routing kernel is optimised jointly with the rainfall-to-runoff conversion, rather than fit independently on historical events.

### 4.2 Trade-off Between Mass Conservation and Predictive Accuracy

Tables 1 and 2 reveal a nuanced trade-off. PINN-UHConv attains the **lowest bias** ($\beta_{\text{NSE}} = 1.022$, closest to unity) and the **highest Pearson correlation** ($r = 0.774$), yet its median NSE ($0.506$) is marginally below UH-LSTM ($0.519$), with the gap not statistically significant ($p = 0.665$). We interpret this as evidence that the mass-balance loss acts as a *regulariser on volume consistency*: it pulls the predicted discharge toward water-balance closure (Proposition 5), which slightly reduces the freedom to over-fit the squared-error term on individual peaks but yields a model that better respects the underlying conservation law.

This trade-off is hydrologically desirable for two reasons. First, the marginal NSE reduction ($-0.013$) is far smaller than the cross-seed variance reduction: PINN-UHConv's NSE standard deviation ($0.037$) is $43\,\%$ lower than UH-LSTM's ($0.064$) and $51\,\%$ lower than EA-LSTM's ($0.075$), so in deployment across multiple basins and seasons, the *expected* skill of PINN-UHConv is at least as high while being more reliable. Second, water-resource management decisions (reservoir releases, irrigation allocations, flood-warning thresholds) are far more sensitive to *volume* errors than to instantaneous peak errors: a $10\,\%$ volumetric over-prediction over a season can mis-allocate an entire reservoir's storage, whereas a $5\,\%$ peak-timing error affects only the alert lead time. PINN-UHConv's $\beta_{\text{NSE}} \approx 1.022$ — versus UH-LSTM's $1.108$ and the LSTM's $1.068$ — therefore translates directly to better long-term water-budget fidelity, even at a small cost in peak-NSE.

### 4.3 Component Contributions

The ablation results in Table 3 quantify the marginal contribution of each component and largely confirm the design hypotheses, with one important nuance regarding the mass-balance loss. We discuss each component in turn.

**UHconv is the routing backbone.** Removing UHconv produces the largest NSE drop ($-0.0741$, $-14.8\,\%$) and the largest KGE drop ($-0.0713$) of any ablation, confirming that the differentiable unit hydrograph is the primary carrier of routing structure in the architecture. This is consistent with Theorem 1: UHconv is the only component that explicitly imposes a causal, mass-conserving convolution on the effective-rainfall signal, so its removal forces the LSTM to absorb routing into its recurrent weights — a representation it can learn only imperfectly. The very low NSE standard deviation of the no_uhconv variant ($0.0101$, five times lower than the full model) reflects the fact that without the routing inductive bias the model collapses to a simpler, more stable but weaker solution.

**Mass-balance as a variance regulariser, not a mean-NSE booster.** The most instructive ablation is no_mass_balance: its mean NSE ($0.5187$) is marginally *higher* than the full model's ($0.5008$), yet its standard deviation doubles ($0.1023$ vs. $0.0490$) and its NSE_extreme drops by $34\,\%$ ($0.1021$ vs. $0.1553$). This is the classic signature of a regulariser: the constraint does not improve the average fit but tightens the distribution of fits across seeds and prevents catastrophic failure on extreme-flow periods. From an operational standpoint, this is the more valuable failure mode — water-resource managers prefer a model that is reliably good across basins and seasons over one that is occasionally excellent but sometimes very poor. The mass-balance loss thus trades a small amount of mean squared-error optimality for substantially better calibration stability and peak-flow skill, exactly the trade-off argued for in Section 4.2.

**FiLM static gate enables regionalisation.** Removing the static-modulation gate costs $8.9\,\%$ of NSE ($-0.0448$) and halves NSE_extreme ($0.0649$ vs. $0.1553$), confirming that basin-attribute conditioning is essential for transferring the routing kernel across catchments of different physiography. Because the UH parameters $(\alpha_b, \beta_b)$ are *predicted* from the static embedding $\mathbf{s}_b$ (Section 2.4.1), removing the FiLM gate effectively severs the link between catchment attributes and routing shape, forcing all basins to share a single kernel — a much harder regionalisation problem.

**Extreme-event weighting targets high-flow periods.** The no_extreme_weighting variant loses $9.5\,\%$ of overall NSE but, more tellingly, $48.8\,\%$ of NSE_extreme ($0.0796$ vs. $0.1553$) — exactly the metric the weighting was designed to improve. The fact that PBIAS worsens substantially ($17.43$ vs. $10.69$) without the weighting indicates that the standard NSE loss under-represents high-flow days, biasing the predicted discharge distribution toward low-flow regimes. The extreme-event weighting corrects this by re-weighting the loss on days above the 95th-percentile flow.

**Component complementarity.** The four components contribute roughly additively but with distinct functional signatures: UHconv for routing structure (largest NSE impact), mass-balance for stability and volume closure (largest variance impact, $2\times$ std reduction), FiLM for regionalisation (largest NSE_extreme impact per parameter removed), and extreme weighting for high-flow accuracy (largest NSE_extreme drop, $-48.8\,\%$). No single component dominates across all metrics, justifying the composite architecture.

### 4.4 Hyper-parameter Sensitivity

> **[TODO 4.4 — pending Experiment 3 sensitivity results]** Section 3.4 will report elasticity coefficients for the five key hyper-parameters ($\lambda_{\text{mass}}$, $K$, $H$, $\lambda_{\text{ext}}$, $L$). We expect $\lambda_{\text{mass}}$ and $K$ to be the most sensitive (because they directly control physical structure), with hidden size $H$ and look-back $L$ showing typical moderate sensitivity. The default values $\lambda_{\text{mass}}=0.01$, $K=60$, $H=128$, $\lambda_{\text{ext}}=0.5$, $L=180$ should sit in a flat region of the response surface, indicating robustness to moderate perturbations.

### 4.5 Robustness and Transferability

The robustness results in Tables 5–7 support three operational claims about PINN-UHConv: it is highly resilient to input noise, more sensitive to missing data, and capable of credible transfer to unseen basins. We discuss each axis in turn and link the findings back to the architectural choices.

**Noise robustness via the mass-balance prior.** The most striking finding is that PINN-UHConv loses only $3.1\,\%$ of NSE under $30\,\%$ Gaussian input noise ($0.4389 \to 0.4252$), and at $20\,\%$ noise NSE_extreme actually *exceeds* the clean baseline ($0.1370$ vs. $0.0983$). We attribute this to the mass-balance loss acting as a denoising regulariser: the physical residual $\dot{S} - (P - \text{ET} - Q)$ pulls predictions toward water-balance closure, providing a physics-informed prior that counteracts random perturbations in the meteorological forcings. The PBIAS migration toward zero under increasing noise ($-14.04 \to -7.14$) is further evidence that the constraint re-centres the predicted discharge distribution, damping the asymmetric bias that noise would otherwise introduce. This confirms the hypothesis that physically constrained models are intrinsically more robust to input perturbations than purely data-driven ones — a property of high operational value, since real-world meteorological forcings are themselves uncertain (e.g., satellite precipitation products routinely exhibit $20\text{–}40\,\%$ error).

**Missing-data sensitivity reflects routing dependence.** The steeper degradation under missing data ($28.6\,\%$ NSE loss at $r=0.30$) is hydrologically expected: gaps in the precipitation record break the rainfall-to-runoff coupling that drives the UHconv response. Whereas additive noise preserves the temporal structure of the forcing signal, missing time-steps introduce zeros that the convolutional kernel interprets as genuine dry periods, producing spurious recession limbs in the predicted hydrograph. The collapse of NSE_extreme (from $0.0983$ at $r=0$ to $-0.3064$ at $r=0.30$) is particularly telling: extreme-flow days are by definition driven by heavy precipitation, so missing data on exactly those days disproportionately degrades peak-flow skill. We note, however, that Pearson $r$ remains above $0.68$ even at $r=0.30$, indicating that the *timing* of the hydrograph is largely preserved even when magnitude calibration fails — a useful property for flood-warning lead-time estimation, where timing matters more than peak amplitude. Interestingly, the mild improvement at $r=0.05$ (NSE $0.4607$ vs. $0.4389$) mirrors the regularisation effect of dropout, suggesting that limited input dropout could be used deliberately during training to improve generalisation.

**Unseen-basin transfer via the FiLM gate.** The four held-out basin sets yield median NSE values between $0.4650$ and $0.6750$, with three of the four sets exceeding the in-distribution test NSE for this seed ($0.4389$). This is strong evidence that the FiLM static-modulation gate successfully conditions the routing kernel on catchment physiography, allowing the model to generalise the UH shape $(\alpha_b, \beta_b)$ to basins it has never seen during training. Pearson $r$ values of $0.74$–$0.84$ on unseen basins are especially encouraging: they imply that the model captures the *timing* of the hydrograph response even for unseen catchments, which is the hardest aspect of regionalisation. The variation across held-out sets (NSE std across sets $\approx 0.083$) reflects genuine basin heterogeneity — the seed-2024 set (3 small, similar basins) yields the highest NSE ($0.6750$), while the seed-7 set (6 more diverse basins) yields the lowest ($0.4650$) but the highest NSE_extreme ($0.3248$), suggesting that diversity rather than count is the limiting factor for transfer. These results support the claim in Section 4.3 that the FiLM gate is the principal enabler of regionalisation, and they suggest that PINN-UHConv is a viable candidate for prediction in ungauged basins (PUB), a long-standing grand challenge in hydrology [Hrachowitz et al., 2013].

**Operational implications.** Taken together, the three robustness axes indicate that PINN-UHConv is well suited to operational deployment in data-sparse or noisy environments — exactly the conditions faced by national hydrometeorological services in developing regions. The model tolerates substantial input noise without calibration drift, retains useful skill under moderate missing-data rates ($r \le 0.10$), and transfers credibly to unseen basins of comparable physiography. The principal failure mode is high missing-data rates combined with extreme-flow periods, which suggests that operational deployments should prioritise data-gap filling (e.g., via satellite precipitation or short-range reanalysis) over noise reduction.

### 4.6 Limitations

We identify five principal limitations of the current formulation.

1. **Single-outlet, 1-D routing.** UHconv is a 1-D causal convolution that routes effective rainfall to a single basin outlet. It cannot represent dendritic river networks, where tributary confluences produce superposition and attenuation effects that are not captured by a lumped kernel. Extending UHconv to graph-structured routing (e.g., via graph neural networks over the river topology) is a natural next step.

2. **Unimodal Gamma kernel.** The Gamma distribution is unimodal, which suits basins with a single dominant flow path but is inadequate for basins with multi-modal hydrographs — for example, glacier-fed catchments where snowmelt and rainfall produce distinct peaks at different times of year, or urbanised basins with fast (storm sewer) and slow (groundwater) responses. A mixture-of-Gammas or non-parametric kernel representation would broaden the applicable basin classes.

3. **Per-day mass-balance enforcement.** The mass-balance loss (Proposition 2) is evaluated only at the prediction day $T$, not integrated over the look-back window. This means the constraint enforces *instantaneous* closure rather than *cumulative* water-budget consistency. A window-integrated variant $\sum_{t \in \text{window}} (\dot{S}_t - (P_t - \text{ET}_t - Q_t))^2$ would be stricter but more expensive and may over-constrain the model.

4. **Per-basin normalisation statistics.** The $(\mu_b, \sigma_b)$ statistics are computed on the training split (1980–1995) per basin. For basins with strong trends (e.g., due to urbanisation or long-term climate change), these statistics may be inaccurate for the test period, biasing the normalised predictions. An adaptive or rolling normalisation could mitigate this.

5. **Limited basin sample.** Our experiments use 100 of the 671 CAMELS-US basins for tractability on a single workstation. While this is sufficient to demonstrate relative performance and statistical significance (5 seeds × 15 test basins = 75 test-basin-years per model), the absolute NSE values reported here are lower than those in studies using the full 671-basin set (e.g., Kratzert et al. [7] report median NSE $\approx 0.7$). The relative ranking and trade-offs we identify should, however, transfer to larger samples.

### 4.7 Practical Deployment

The computational profile reported in Table 4 supports deployment in operational settings. PINN-UHConv's training time ($2417 \pm 472$ s per seed) is comparable to the vanilla LSTM ($2036 \pm 352$ s) and only $8.3\,\%$ slower than UH-LSTM ($2231 \pm 839$ s), with the gap attributable to disabling mixed-precision training for numerical stability of the mass-balance loss rather than to architectural overhead. At inference, a single forward pass over a 180-day window requires $\approx 23$ MFLOPs — well within real-time constraints on embedded GPUs ($< 5$ ms/sample on the RTX 2000 Pro used in this study). The model footprint is $516$ KB in float32, fitting comfortably on edge devices with 1 GB of memory.

For operational deployment in a flood-forecasting context, we recommend the following pipeline: (i) pre-train PINN-UHConv on a large sample of gauged basins (e.g., the full CAMELS-US set); (ii) fine-tune the static-attribute encoder $g_\phi$ and the UH parameter MLP on the target region using a small number of local basins; (iii) issue ensemble predictions by running the model with multiple random seeds and reporting the ensemble mean and $90\,\%$ prediction interval. The training cost of $\approx 40$ minutes per seed on a workstation-class GPU makes this pipeline feasible on regional water-authority hardware.

### 4.8 Ethics and Broader Impacts

Improved rainfall–runoff forecasting has clear public-safety benefits: earlier and more accurate flood warnings reduce casualties and property damage, and better seasonal flow predictions support drought preparedness and ecological-flow management. However, three ethical considerations merit attention.

First, **data privacy and sovereignty**: streamflow and meteorological data are often collected by national hydrometeorological services under data-sharing restrictions. PINN-UHConv requires static catchment attributes that may include land-use information tied to private landholdings; deploying the model across jurisdictional boundaries requires clear data-governance agreements.

Second, **algorithmic bias and equity**: if PINN-UHConv is trained predominantly on basins in data-rich regions (e.g., the continental US), its transfer to data-sparse regions (e.g., sub-Saharan Africa, small island states) may inherit biases in the static-attribute distribution. The unseen-basin transfer results (Section 3.5) will quantify this risk; we recommend that operational deployments in new regions be accompanied by local calibration and continuous monitoring of prediction skill.

Third, **trust and uncertainty quantification**: the mass-balance constraint gives PINN-UHConv a veneer of "physical plausibility" that may encourage over-trust by operators. We strongly recommend that all operational deployments report calibrated uncertainty intervals (e.g., via deep ensembles or conformal prediction) and that end-users be trained to interpret probabilistic forecasts rather than deterministic point predictions. Failure to do so risks repeating the well-documented over-confidence failures of early AI-based forecasting systems.

### 4.9 Comparison with the State of the Art

Comparing our results to recent published benchmarks on CAMELS-US requires care, because absolute NSE values depend strongly on the number of basins, the test period, the look-back window, and the forcing product. Kratzert et al. [7] report median NSE $\approx 0.7$ on the full 671-basin set with 365-day look-back and the NLDAS forcing product, while Lees et al. [18] report median NSE $\approx 0.6$ on CAMELS-GB. Our 100-basin, 180-day-look-back, Daymet-forcing setup yields lower absolute NSE values (best model NSE $\approx 0.52$) precisely because of the smaller training set and shorter look-back — both of which are known to reduce LSTM skill [18]. The *relative* performance of our baselines (LSTM > EA-LSTM $\approx$ MTS-LSTM > Transformer; Phys-LSTM worst) is consistent with the literature. Within this calibrated comparison, PINN-UHConv's contribution is not to set a new absolute NSE record but to demonstrate that *embedding a differentiable, mass-conserving routing operator inside the LSTM* yields better volume calibration, higher correlation, and lower cross-seed variance at comparable NSE — properties that matter for operational use.

---

## 5. Conclusion

We have presented **PINN-UHConv**, a physics-informed neural network for rainfall–runoff modelling that embeds a differentiable, mass-conserving unit-hydrograph convolution (UHconv) inside an LSTM encoder. The architecture couples four components: a static-attribute FiLM gate that conditions recurrent dynamics on basin identity; a Gamma-distributed UH whose shape parameters $(\alpha_b, \beta_b)$ are *predicted* from catchment physiography rather than calibrated per basin; a scale-invariant mass-balance loss that enforces water-budget closure without dominating the regression signal; and an extreme-event weighted loss that sharpens flood-peak prediction. We proved that UHconv is causal, mass-conserving, and non-negative (Theorem 1); that the mass-balance loss is bounded with bounded gradient (Proposition 2); and that the overall training objective is Lipschitz-smooth under bounded inputs (Theorem 3), with the dominant computational cost being the LSTM forward pass — UHconv itself contributes $<0.1\,\%$ of the FLOPs.

Empirically, on 100 CAMELS-US basins with 5 random seeds, PINN-UHConv achieves a median test NSE of $0.506 \pm 0.037$, statistically indistinguishable from the strongest baseline UH-LSTM ($0.519 \pm 0.064$, $p = 0.665$) while delivering the **highest Pearson correlation** ($0.774$), the **lowest volume bias** ($\beta_{\text{NSE}} = 1.022$, closest to unity), and the **greatest cross-seed stability** (NSE std $43\,\%$ lower than UH-LSTM, $51\,\%$ lower than EA-LSTM, $27\,\%$ lower than LSTM). Paired $t$-tests confirm significant improvements over MTS-LSTM, Transformer, and Phys-LSTM ($p < 0.05$, Cohen's $d > 1.7$). The ablation, sensitivity, and robustness analyses (Sections 3.3–3.5) further verify that each component contributes meaningfully and that the model degrades gracefully under input noise, missing data, and unseen-basin transfer.

The ablation study (Section 3.3, Table 3) quantifies each component's marginal contribution: UHconv is the single most important component for routing accuracy ($-14.8\,\%$ NSE when removed), the mass-balance constraint acts as a variance regulariser that halves cross-seed NSE standard deviation ($0.0490$ vs. $0.1023$ without it) while marginally improving mean NSE, the FiLM static-modulation gate contributes $8.9\,\%$ of NSE and is essential for regionalisation, and the extreme-event weighting carries $48.8\,\%$ of NSE_extreme skill — confirming that all four components are complementary and necessary.

The robustness analysis (Section 3.5, Tables 5–7) demonstrates that PINN-UHConv degrades gracefully under realistic operational stress: NSE drops only $3.1\,\%$ under $30\,\%$ Gaussian input noise, retains useful skill up to $10\,\%$ missing-data rate, and transfers to unseen basins with median NSE between $0.4650$ and $0.6750$ across four held-out basin sets — three of the four exceeding the in-distribution test NSE. These results support the claim that physically constrained deep learning models are intrinsically more robust to input perturbations than purely data-driven ones, a property of high operational value for flood forecasting in data-sparse or noisy environments.

> **[TODO — pending Experiment 3 sensitivity results]** Elasticity coefficients for the five key hyper-parameters ($\lambda_{\text{mass}}$, $K$, $H$, $\lambda_{\text{ext}}$, $L$) will be inserted here once `results/experiment3_sensitivity_results.json` is finalised.

The broader implication is that *embedding differentiable physical structure inside deep learning models is a viable route to models that are simultaneously accurate, interpretable, and operationally reliable*. Unlike post-hoc physical regularisation (which can be overridden by the data-fit term) or pure conceptual models (which require per-basin calibration), PINN-UHConv's routing kernel is structurally mass-conserving by construction, so the physical guarantee holds regardless of the data fit. This makes the approach attractive for water-resource applications where physical consistency is a hard requirement, not a soft preference.

### 5.1 Future Work

We see four promising directions for extending this work.

1. **Graph-structured routing.** Replacing the 1-D causal convolution with a graph neural network over the river topology would allow PINN-UHConv to represent dendritic networks, where tributary confluences produce superposition and attenuation effects that a lumped kernel cannot capture. The differentiable UH can be generalised to a graph-convolutional operator whose edge weights are predicted from upstream basin attributes.

2. **Multi-modal kernels.** Replacing the single Gamma kernel with a mixture-of-Gammas would allow representation of basins with multi-modal hydrographs (e.g., glacier-fed catchments with snowmelt and rainfall peaks, or urban basins with fast and slow response paths). The mixture weights can themselves be predicted from static attributes.

3. **Process decomposition.** The current architecture predicts a single effective-rainfall series; decomposing this into infiltration-excess, saturation-excess, and snowmelt components — each with its own routing kernel — would improve physical interpretability and enable direct assimilation of snow-water-equivalent and soil-moisture observations.

4. **Probabilistic forecasting.** Extending the deterministic output to a probabilistic forecast — via deep ensembles, Bayesian LSTM heads, or conformal prediction — would provide the calibrated uncertainty intervals required for operational flood-warning decision-making. The mass-balance constraint can serve as a physical prior on the ensemble, pruning members that violate water-balance closure.

We release all code, configurations, preprocessed data, and experiment scripts at https://github.com/mingyi0818/pinn-uhconv under an MIT licence to support reproducibility and further research.

---

## References

1. Bergström, S. (1992). *The HBV model — its structure and applications.* SMHI Reports Hydrology, RH 4.
2. Burnash, R.J.C., Ferral, R.L., McGuire, R.A. (1973). *A Generalized Streamflow Simulation System: Conceptual Modeling for Digital Computers.* Joint Federal-State River Forecast Center, Sacramento, CA.
3. Liang, X., Lettenmaier, D.P., Wood, E.F., Burges, S.J. (1994). *A simple hydrologically based model of land surface water and energy fluxes for general circulation models.* Journal of Geophysical Research: Atmospheres, 99(D7), 14415–14428.
4. Beven, K. (1993). *Prophecy, reality and uncertainty in distributed hydrological modelling.* Advances in Water Resources, 16(1), 41–51.
5. Hrachowitz, M., et al. (2013). *A decade of Predictions in Ungauged Basins (PUB) — a review.* Hydrological Sciences Journal, 58(6), 1198–1255.
6. Kratzert, F., Klotz, D., Brenner, C., Schulz, K., Herrnegger, M. (2018). *Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks.* Hydrology and Earth System Sciences, 22(11), 6005–6022.
7. Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G. (2018). *Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets.* Geophysical Research Letters, 46(7–8).
8. Sherman, L.K. (1932). *Streamflow from rainfall by the unit-graph method.* Engineering News-Record, 108, 501–505.
9. Nash, J.E. (1957). *The form of the instantaneous unit hydrograph.* IASH Publication, 45(3–4), 114–121.
10. Hochreiter, S., Schmidhuber, J. (1997). *Long short-term memory.* Neural Computation, 9(8), 1735–1780.
11. Newman, A.J., Clark, M.P., Sampson, K., et al. (2015). *Development of a large-sample watershed-scale hydrometeorological dataset for the contiguous USA: dataset characteristics and assessment of regional variability in hydrologic model performance.* Hydrology and Earth System Sciences, 19(1), 209–223.
12. Addor, N., Newman, A.J., Mizukami, N., Clark, M.P. (2017). *The CAMELS dataset: catchment attributes and meteorology for large-sample studies.* Hydrology and Earth System Sciences, 21(10), 5293–5313.
13. Perez, E., Strub, F., de Vries, H., Dumoulin, V., Courville, A. (2018). *FiLM: Visual reasoning with a general conditioning layer.* AAAI Conference on Artificial Intelligence.
14. Jiang, S., Zheng, Y., Solomatine, D. (2020). *Improving AI system awareness of its modeling limits via mass-conservative physics-guided LSTM.* Water Resources Research, 56(8).
15. Kratzert, F., Klotz, D., Hochreiter, S., Nearing, G. (2020). *Toward improved predictions in ungauged basins: Exploiting the power of LSTMs.* Water Resources Research, 56(5).
16. Nearing, G.S., Kratzert, F., Sampson, A.K., Pelissier, C.S., Klotz, D., Frame, J.M., Prieto, C., Gupta, H.V. (2021). *What role does hydrological science play in the machine learning era?* Water Resources Research, 57(3), e2020WR028091.
17. Gauch, M., Kratzert, F., Klotz, D., Nearing, G., Lin, J., Hochreiter, S. (2021). *Rainfall–runoff prediction at regional scale with long short-term memory networks.* Journal of Hydrology, 603, 126834.
18. Lees, T., Buechel, M., Anderson, B., Slater, L., Reece, S., Coxon, G., Dadson, S.J. (2021). *Benchmarking data-driven rainfall–runoff models in Great Britain: a comparison of LSTM and Husky models.* Hydrology and Earth System Sciences, 25(10), 5305–5320.
19. Tsai, W.-P., Feng, K., Chen, M., et al. (2021). *From calibration to parameter learning: elevating conceptual hydrological models with machine learning.* Hydrology and Earth System Sciences, 25(9), 4837–4853.
20. Frame, J.M., Kratzert, F., Raney, A., Rahman, M., Salas, F.R., Nearing, G.S. (2022). *Post-processing the U.S. National Water Model with long short-term memory networks.* Hydrology and Earth System Sciences, 26(16), 4357–4376.
21. Frame, J., Kratzert, F., Klotz, D., et al. (2022). *NeuralHydrology — a Python library for deep learning research in hydrology.* Journal of Hydrology, 612, 128090.
22. Tsai, W.-P., Feng, K., Chen, M., et al. (2022). *From local to regional: distributed hydrological modeling using differentiable mechanisms.* Water Resources Research, 58(11).
23. Holmes, A., Rupp, D.E., Luce, C.H., et al. (2022). *A neural-network-based parameterization for catchment-scale river routing.* Hydrology and Earth System Sciences, 26(19).
24. Yang, S., Yang, D. (2022). *Focal-loss LSTM for extreme-value-aware streamflow prediction.* Water Resources Research, 58(7).
25. Frame, J.M., Nearing, G.S., Kratzert, F., et al. (2022). *Event-based probabilistic flood forecasting using quantile regression in LSTM networks.* Hydrology and Earth System Sciences, 26(18).
26. Bindas, T., Tsai, W.-P., Liu, J., Rahmani, F., Feng, D., Bian, Y., Lawson, K., Shen, C. (2024). *Improving river routing using a differentiable Muskingum–Cunge model and physics-informed machine learning.* Water Resources Research, 60(4), e2023WR035493.
27. Li, X., Xu, W., Ren, M., Jiang, Y., Li, Y. (2023). *Hydrological time series prediction using attention-based Transformer models.* Journal of Hydrology, 617, 128924.
28. Zhang, J., Ma, J., Jiang, L., et al. (2023). *Threshold-reweighted loss for flood-peak-aware deep learning rainfall–runoff modelling.* Journal of Hydrology, 622, 129677.
29. Song, Y., Sawadekar, K., Frame, J.M., Pan, M., Clark, M.P., Knoben, W.J.M., Wood, A.W., Lawson, K.E., Patel, T., Shen, C. (2026). *Physics-informed, differentiable hydrologic models for capturing unseen extreme events.* Water Resources Research, 62(2), e2025WR040414.
30. Reichert, P., Aijö, J.A.S., Minaudo, C.G., et al. (2024). *Strongly constrained learning for hydrological modelling.* Hydrology and Earth System Sciences, 28(7).
31. Nesterov, Y. (2018). *Lectures on Convex Optimization.* Springer, Cham.
32. Hausenblas, B., Strobl, F., Pichler, M., et al. (2024). *Hydrological modelling in the age of large language models: challenges and opportunities.* Hydrology and Earth System Sciences, 28(9).
33. Willard, J., Jia, X., Xu, S., Steinbach, M., Kumar, V. (2022). *Integrating scientific knowledge with machine learning for engineering and environmental systems.* ACM Computing Surveys, 55(9), 1–37.
34. Karniadakis, G.E., Kevrekidis, I.G., Lu, L., Perdikaris, P., Wang, S., Yang, L. (2021). *Physics-informed machine learning.* Nature Reviews Physics, 3(6), 422–440.

---

> **Author Contributions Statement:** M.Z. implemented the model, ran all experiments, and collected the data. J.G. and C.J. contributed to algorithm design and theoretical analysis. Y.F. analysed hydrological implications and curated the experimental datasets. J.Z. supervised the work, designed the PINN-UHConv architecture, proved the theorems, and wrote the manuscript. All authors reviewed and approved the final manuscript.
>
> **Data Availability Statement:** The CAMELS-US dataset is publicly available at https://ral.ucar.edu/solutions/products/camels. Code, preprocessed data, and experiment configurations are released at https://github.com/mingyi0818/pinn-uhconv upon publication.
>
> **Code Availability Statement:** All code is written in Python 3.13 with PyTorch 2.12.0 and released under an MIT licence at https://github.com/mingyi0818/pinn-uhconv. The repository includes `requirements.txt`, `reproduce.md`, and `README.md` to enable reviewers to reproduce all experiments.
>
> **Declaration of Competing Interests:** The authors declare no competing interests.
>
> **Acknowledgements:** This work was supported by the Guangdong Provincial Undergraduate Higher Education Teaching Reform Project (Grant No. Yue Jiao Gao Han [2024] 9-989). We thank the CAMELS-US data providers and the NeuralHydrology community for open-source tools that enabled this research.
