# eap-ig-faithfulness

This repo contains the code for the paper "Automatic Circuit Finding and Faithfulness" (COLM 2024; [Arxiv link](https://arxiv.org/abs/2403.17806)). The code for manipulating model graphs and performing EAP/-IG is contained in a submodule, [EAP-IG](https://github.com/hannamw/EAP-IG); make sure to pull the commit / submodule here, as I've updated the EAP-IG module since releasing this experimental code. After pulling that submodule, you can install it by calling `pip install -e .` in its directory. A Conda environment that can be used to run these experiments is contained in `environment.yml`.

You can replicate the experiments in the paper as follows:

- For each task in `['ioi', 'greater-than', 'sva', 'gender-bias', 'fact-retrieval-comma', 'hypernymy-comma']`:
  - first, run `pareto.py` to collect results for EAP/EAP-IG. You must specify the model name (`--model`), task (`--task`), task metric (either `logit_diff` or `prob_diff`, `--metric`), and `--batch_size`. If the model is large, you might instead want to run `pareto_big.py`, which provides options to separately set the `--eval_batch_size`.
  - then, run `get_real_values.py`, with similar options. This can be slow, as this script performs the actual activation patching metric change values with which you will compare the EAP/-IG estimates / circuits.
- Then, you will be ready to recreate most of the figures in the paper. Once you do so, the figures will be generated as a `.png` or `.pdf` file in the corresponding subfolder of the relevant directory in `results`.
  - To recreate Figure 1, you can run `results/first_figure/first_figure.py`.
  - To recreate Figure 3, you can run `results/real_pareto_combined/plot_real_pareto_normalized.py`.
  - To recreate Figure 4, you can run `results/real_rank/compare_real.py` and then `results/real_rank/plot_overlap.py`.
- To replicate the overlap experiments (Figure 5), you can run `overlap_heatmaps.py` as well as `all-cross-task-faithfulness.py`. Then, run `results/cross-task/make_all_heatmaps.py`; if you want recall heatmaps, as in Appendix I, run `results/cross-task/recall_heatmaps.py`
- The figures in Appendix can be replicated as follows.
  - For Figure 6, run `results/manual_overlap/ioi_overlap.py` and `results/manual_overlap/greater_overlap.py`
  - For Figure 7, run `test_ig_iterations.py`, taking care to specify the `--model` and `--task` of interest, as well as the `--attribution_metric` and the `--eval_metric`. You can generate the figure using `results/ig_test/plot_ig_test.py`.
  - For Figure 8, run `results/real_rank/plot_kendall.py`.
  - For Figure 9, run `results/real_rank/node_edge_overlap_plot.py`.
  - For Figure 10, run `results/pareto/plot_pareto_normalized_single.py`.
  - For Figure 11, run the larger models as described in the first step, and then again run `results/real_pareto_combined/plot_real_pareto_normalized.py`.

The data used in this paper is available in the `data` folder.
