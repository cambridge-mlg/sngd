# sngd

Code to accompany the paper ["Optimising Distributions with Natural Gradient Surrogates"](https://proceedings.mlr.press/v238/so24a.html) (AISTATS 2024).

NB: we have only included synthetic datasets in this repository; details of the datasets used in the paper can be found in Appendix D.

Example usage:
```bash
pip install -e .
python -m experiments.mvsn_ml sngd --dataset=synthetic --lr-theta=1e-2 --lr-lam=1e-2
```
