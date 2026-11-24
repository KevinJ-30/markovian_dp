We thank reviewer PpNz for their positive assessment of our work and recognition of its theoretical and empirical contributions. We address the stated weaknesses below.

> *Algorithmic novelty of neighborhood sampling and edge dropping (W1)*

We agree that neighborhood sampling (Hamilton et al., 2017) and edge dropping (Rong et al., 2020) are established techniques in graph learning, and that our primary contributions lie in analyzing their privacy-amplifying properties. That being said, we argue that this is a positive aspect of our framework. Rather than introducing specialized architectures or restrictive learning procedures, as in prior work (Sajadmanesh and Gatica-Perez, 2024; Zhang et al., 2024b), our work analyzes the inherent randomness in standard graph learning procedures, enabling architecture-agnostic, end-to-end private learning. 

We will introduce the following additional sentence into our discussion in Section 2 to clarify this distinction:

"While neighborhood sampling and edge dropping are well-established techniques in non-private graph learning, our contribution lies in establishing their privacy-amplifying properties, thus enabling private graph learning without specialized model architectures or training procedures."

> *Extent of theoretical contributions beyond existing analyses (W1, W2)*

We indeed make use of group-privacy, message-interception (Scholten et al., 2022), and conditional-coupling (Schuchardt et al., 2024) arguments within our analysis. However, we emphasize that we do not directly apply the existing analyses of prior work. In particular, the group-privacy amplification results of Schuchardt et al. (2024) do not directly apply to our setting, since they concern independent subsampling of fixed records, whereas our setting involves data-dependent subgraph distributions, which involve dependencies between overlapping paths.

Consequently, our analysis requires several additional ingredients. Firstly, we design a new coupling of randomized expansion operations to establish path-sensitive privacy bounds (Proposition 4.2). Secondly, we provide a new decomposition of per-root expansion distributions, allowing us to construct conditional batch couplings (Lemmas D.5–D.8). Lastly, we  upper bound the resulting privacy profile using path-count arguments (Lemmas D.13–D.14). These arguments go beyond existing message-interception and group-privacy results in showing our primary result, Theorem 4.3.

We also remark that Theorem 4.3 is not our only theoretical contribution in this paper. For instance, Proposition 4.1 establishes a new worst-case privacy characterization for deterministic subgraph expansion, while Theorem 4.4 establishes a complementary lower bound to our main theorem which captures appropriate dependence on graph degree and edge-retention probability.

We will revise our discussion to more clearly isolate these technical contributions from the existing tools leveraged in our proofs.

---

We again thank reviewer PpNz for their feedback. We hope these clarifications address your concerns and would be happy to discuss further.
