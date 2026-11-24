We thank reviewer Dzsy for their positive assessment of our work and recognition of its motivations, theoretical contributions, and empirical evaluation. We address stated weaknesses and questions below.

> *Notions of adjacency and randomized degree capping (Q1, W1)*

We clarify that Definition 3.5 indeed includes incoming edges to a node. Importantly, our privacy analysis concerns the learning mechanism on bounded-out-degree graphs, and Theorem 4.3 provides valid guarantees for all neighboring graphs satisfying this constraint. This is a standard assumption in private graph learning (Daigavane et al., 2022; Sajadmanesh and Gatica-Perez, 2024), and is not a restriction which applies only to our algorithm. In our experiments, random edge pruning allows us to evaluate our method on real-world graph datasets while respecting this assumption. The preprocessing procedure itself is not required by our algorithm or its theoretical analysis.

Nevertheless, extending our guarantees to arbitrary input graphs is an interesting direction for future work, potentially through stable bounded-degree projections (Jain et al., 2024) or alternative notions of node adjacency.

> *Results with fixed $p_2$ (Q2, W2)*

We provide the following additional results with a fixed edge-retention probability of $p_2=0.1$. Each cell reports the mean test score (%) over seeds 6–10 for **SparseSAGE / SparseGIN**. 

| $\epsilon$ | Arxiv | Products | Reddit | Yelp | Amazon | Facebook-100 | MAG |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 43.51 / 38.10 | 56.58 / 61.55 | 63.51 / 66.31 | 44.78 / 40.35 | 37.16 / 23.46 | 33.69 / 31.94 | 8.58 / 18.27 |
| 2 | 48.30 / 47.73 | 59.66 / 65.15 | 72.66 / 77.33 | 48.25 / 42.16 | 45.53 / 35.72 | 35.10 / 32.28 | 8.64 / 23.28 |
| 5 | 52.09 / 52.10 | 62.21 / 67.96 | 79.92 / 81.06 | 51.08 / 43.28 | 51.63 / 41.46 | 33.60 / 31.15 | 9.47 / 23.30 |
| 8 | 53.62 / 53.87 | 62.87 / 68.33 | 81.84 / 81.64 | 51.77 / 43.54 | 53.11 / 42.45 | 33.21 / 27.73 | 10.07 / 22.92 |

We observe largely similar performance to our original results, suggesting that our empirical performance is not particularly dependent on tuning this parameter.

> *ProGAP with additional stages and DPAR performance (Q3, W3)*

We first remark that ProGAP does not set a fixed default number of stages. In fact, the original work tunes model depth over $\{1,\ldots,5\}$, and observes that shallower models can achieve better performance at more restrictive privacy budgets (Sajadmanesh and Gatica-Perez, 2024, Section 6.3).

Nevertheless, we provide the following additional experiments evaluating ProGAP at depths 1, 3, and 5. Each cell reports the mean test score (%) over 5 random seeds, in the order **depth 1 / depth 3 / depth 5**. 

| $\epsilon$ | Arxiv | Products | Reddit | Yelp | Amazon | Facebook-100 | MAG |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48.43 / 47.03 / 45.31 | 51.01 / 49.54 / 47.87 | 54.96 / 50.34 / 47.31 | 38.48 / 32.24 / 25.49 | 8.40 / 9.39 / 11.05 | 33.55 / 33.81 / 32.33 | 15.85 / 11.85 / 7.35 |
| 2 | 50.32 / 49.74 / 49.08 | 53.21 / 52.08 / 51.24 | 60.21 / 57.26 / 55.30 | 45.26 / 39.61 / 35.34 | 7.61 / 8.29 / 9.74 | 33.16 / 33.80 / 33.21 | 20.16 / 17.14 / 14.27 |
| 5 | 51.47 / 51.25 / 51.11 | 55.96 / 54.81 / 54.32 | 65.04 / 63.26 / 62.08 | 50.68 / 46.71 / 43.84 | 23.51 / 7.38 / 8.19 | 33.90 / 33.12 / 33.23 | 24.08 / 24.54 / 22.50 |
| 8 | 51.89 / 51.61 / 51.63 | 58.28 / 56.09 / 55.56 | 69.15 / 65.40 / 64.47 | 52.62 / 49.52 / 47.13 | 33.65 / 20.60 / 8.56 | 34.94 / 33.69 / 33.66 | 25.35 / 26.66 / 25.65 |

We observe that increasing depth provides little improvement in our experimental setting. In fact, depth 1 achieves the highest performance in 22/28 configurations above, supporting our choice of a depth-1 baseline.

For DPAR, although we have not conducted further experiments, we suspect that its suboptimal performance is due to stronger privacy requirements in our evaluation. The original DPAR work (Zhang et al., 2024b) uses values of $\delta$ that are 1–2 orders of magnitude larger than those in our evaluation (e.g., $\delta=10^{-4}$ on Reddit, compared to our choice of $\delta=N_{\mathrm{train}}^{-1.01}$ for $N_\mathrm{train} = 10^6$).

> *Gap between Theorems 4.3 and 4.4 for larger $p_1$ (Q4)*

We provide the following additional experiments evaluating the gap between Theorems 4.3 and 4.4 for larger $p_1$. We fix $p_2=0.5$, $\sigma=5$, $\delta=10^{-5}$, $T=1{,}000$, $K=5$, and $r=1$.

| $p_1$ | Lower $\epsilon$ (Thm. 4.4) | Upper $\epsilon$ (Thm. 4.3) | Upper / lower |
|---:|---:|---:|---:|
| 1.00 | 1,116.83 | 10,272.43 | 9.20× |
| 0.75 | 660.23 | 3,639.18 | 5.51× |
| 0.25 | 105.27 | 301.17 | 2.86× |
| 0.01 | 1.74 | 3.20 | 1.84× |

Clearly, the gap between our upper and lower bounds widens as the root-sampling probability $p_1 \to 1$. However, we argue that this regime is of limited practical relevance: even our lower bounds exhibit substantial privacy loss when $p_1$ is large, particularly under composition over many training iterations. As such, small root-sampling probabilities are essentially necessary to obtain meaningful privacy guarantees in the regimes we consider. Our bounds are considerably closer in precisely this practically relevant regime ($p_1\ll 1$), as demonstrated above and in our paper's numerical evaluations.

---

We again thank the reviewer for their feedback, which will be valuable in improving our paper. We hope these clarifications address your concerns and would be happy to discuss further.

---

P. Jain, A. Smith, and C. Wagaman. “Time-Aware Projections: Truly Node-Private Graph Statistics under Continual Observation.” IEEE Symposium on Security and Privacy (S&P), 2024. [https://arxiv.org/abs/2403.04630](https://arxiv.org/abs/2403.04630)
