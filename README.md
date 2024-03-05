# sngd

Code to accompany the paper ["Optimising Distributions with Natural Gradient Surrogates"](https://arxiv.org/abs/2310.11837).

NB: we have only included synthetic datasets in this repository; details of the datasets used in the paper can be found in Appendix D.

Example usage:
```bash
pip install -e .
python -m experiments.mvsn_ml sngd --dataset=synthetic --lr-theta=1e-2 --lr-lam=1e-2
```