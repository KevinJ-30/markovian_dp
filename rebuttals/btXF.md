We thank reviewer btXF for their positive assessment of our work and discussion of its strengths. We address the stated weaknesses below.

> *Degree of input graphs (W1)*

We agree with the reviewer that our experiments enforce degree caps through random edge pruning. However, we emphasize that our theoretical guarantees are fully valid under the stated assumptions, namely, on the space of graphs with bounded out-degree. Such restrictions on the dataset space are standard in differential privacy (e.g., bounded user contributions in user-level privacy (Charles et al., 2024)) and have been widely adopted in prior private graph learning work (Daigavane et al., 2022; Sajadmanesh and Gatica-Perez, 2024). Degree capping is not part of our proposed mechanism, but simply a means of constructing datasets satisfying our assumptions for empirical evaluation, rather than resorting to synthetic datasets.

Furthermore, we remark that there exist stable projections of arbitrary input graphs onto bounded-degree graphs (Jain et al., 2024). Such techniques provide a potential avenue for extending our framework to arbitrary input graphs.

Finally, a weaker notion of node-DP, defined with respect to changes in a node's features and outgoing edges, may also permit privacy guarantees under degree-capping preprocessing. However, our focus is on establishing privacy amplification for graph learning under bounded-degree assumptions, and we leave such extensions to future work. We will clarify the scope of our guarantees in the revision.

> *Training with and without edge subsampling (W2)*

We provide the following additional comparisons of training with and without edge subsampling ($p_2=0.1$ versus $p_2=1$), using both noise calibrated for $p_2=0.1$ and noise calibrated for $p_2=1$. We report GraphSAGE test performance at $\epsilon\in\{2,8\}$, averaged over five random seeds.

*Table 1: Fixed noise calibrated for $p_2=0.1$. Each entry reports performance with / without edge subsampling (%).*

| $\epsilon$ | ArXiv | Products | Reddit | Yelp | Amazon | FB-100 | MAG |
|---|---|---|---|---|---|---|---|
| 2 | 48.30 / 50.95 | 59.66 / 63.83 | 72.66 / 83.75 | 48.25 / 45.84 | 45.53 / 44.47 | 35.10 / 37.62 | 8.64 / 10.61 |
| 8 | 53.62 / 55.28 | 62.87 / 66.33 | 81.84 / 89.78 | 51.77 / 50.00 | 53.11 / 52.20 | 33.21 / 33.53 | 10.07 / 13.84 |

*Table 2: Fixed noise calibrated for $p_2=1$. Each entry reports performance with / without edge subsampling (%).*

| $\epsilon$ | ArXiv | Products | Reddit | Yelp | Amazon | FB-100 | MAG |
|---|---|---|---|---|---|---|---|
| 2 | 32.91 / 36.71 | 49.83 / 55.14 | 46.00 / 58.80 | 37.26 / 34.50 | 19.40 / 18.65 | 29.66 / 32.64 | 6.69 / 7.81 |
| 8 | 44.44 / 47.79 | 57.40 / 61.88 | 64.38 / 76.55 | 45.80 / 43.22 | 39.85 / 38.76 | 34.25 / 38.99 | 8.73 / 10.32 |

At fixed noise, edge subsampling generally decreases performance. However, at the same privacy budget (comparing subsampled performance in Table 1 with non-subsampled performance in Table 2), the improved amplification from edge subsampling allows substantially less noise, improving performance in 12/14 configurations.
As such, these results suggest that the primary utility benefits of edge subsampling arise from improved privacy amplification, rather than implicit regularization effects. We will incorporate the complete results into our appendix.


> Comparing RDP and PLD accounting (W2)
We provide the following additional comparison of RDP and PLD accounting applied to the same mechanism, using standard RDP-to-DP conversions (Mironov, 2017) and PLD accounting (Zhu et al., 2022). We report GraphSAGE test performance at $, averaged over five random seeds, with noise calibrated to the same privacy budget under each accountant.

*Table 3: Comparison of PLD and RDP accounting. Each entry reports test performance with PLD / RDP accounting (%).*

| $\epsilon$ | ArXiv | Products | Reddit | Yelp | Amazon | FB-100 | MAG |
|---|---|---|---|---|---|---|---|
| 2 | 48.30 / 47.58 | 59.66 / 58.00 | 72.66 / 68.92 | 48.25 / 47.12 | 45.53 / 38.96 | 35.10 / 35.10 | 8.64 / 8.69 |
| 8 | 53.62 / 53.07 | 62.87 / 61.10 | 81.84 / 80.27 | 51.77 / 50.26 | 53.11 / 47.51 | 33.21 / 33.32 | 10.07 / 9.51 |


Our results indicate that PLD accounting generally improves accuracy, although the extent of improvement varies based on the dataset. As such, while our algorithm benefits from tighter PLD accounting, it continues to achieve strong performance under RDP accounting, suggesting that our improvements are not solely attributable to the choice of accountant. We will incorporate this comparison into the appendix of our final revision.

---

As above, we again thank reviewer PpNz for your feedback. We hope our response has addressed your concerns and would be happy to discuss more. 

---

P. Jain, A. Smith, and C. Wagaman. “Time-Aware Projections: Truly Node-Private Graph Statistics under Continual Observation.” IEEE Symposium on Security and Privacy (S&P), 2024. [https://arxiv.org/abs/2403.04630](https://arxiv.org/abs/2403.04630)