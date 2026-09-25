# Reproduction of Results in the Main Paper

To reproduce the results in the main paper, you can use the following commands. 

## Main Experiments

The first step is to run all of the primary experiments, which is summarized in the following bash script and ablates over all relevant parameters.

```bash

```

The second step is to reproduce a markdown table that contains the main results of all the experiments.

```bash

```

This should render a Markdown table in ```enter me later```, which can be used to directly create the primary table in the experiments section of the main paper.

## Ablation Studies

We also provide a set of scripts to run, create Markdown tables for all ablation studies, and render plots for all ablation studies in relevant folder(s).

First, we consider the ablation study over the out-degree cap. 

```bash

```

Now, we consider the ablation study over the model depth.

```bash

```

Finally, we consider the ablation study over the sparsification parameter $p_2$, which can be run using the following command.

```bash


```


This should render a Markdown table in ```enter me later```, which can be used 