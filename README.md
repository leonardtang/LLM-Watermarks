# **Foundations for Understanding and Baselines for Detecting Watermarks in Large Language Models**


The main workhorses of this project are listed below. 


`single_digit.py` contains the code for generating RNG distributions and logit files from Flan-T5-XXL and Alpaca-LoRA. In particular, these are generated for watermarks of varying strengths, controlled by a meta-loop in the code.

`logit_rng.py` is used to generate RNG distributions via saved logit files instead of natively from LLMs. This significantly reduces latency, and is also valid due to the fixed token lookback of watermarking technology. 

`lorenz.py` is used to generate ranked probability Lorenz curves, as well as related metrics such as the Gini coefficient, mean adjacent token difference, and more.

`logit_amplification.py` contains the code for performing $\delta$-Amplification. 

`kolmogorov.py` contains the code used to perform K-S tests on RNG distributions.

`monitor.py` is a script for interacting with ChatGPT to detect watermarking characteristics.

`watermark_playground.py` and `kirchenbauer_watermarks.py` contain implementations of existing watermark methods and our varitions on them, as well as (non-black-box) methods for determing if text (not model) is watermarked.

Logits and RNG distributions are stored in their correspondingly named folders.

The writeup can be found at `LLM_Watermarks_229.pdf`.

Please let us know if you have any questions!

