# CASE-RAG: Decision Models as Risk-Controlled Routers for Retrieval-Augmented Generation

## Article
* **Journal**: IEEE Transactions on Artificial Intelligence
* **Title**: CASE-RAG: Decision Models as Risk-Controlled Routers for Retrieval-Augmented Generation
* **DOI**: 

## Authors
* **Adj. Asst. Prof. Konstantinos I. Roumeliotis | University of Peloponnese**
* **Asoc. Prof. Dionisis Margaris | University of Peloponnese**
* **Asoc. Prof. Dimitris Spiliotopoulos | University of Peloponnese**
* **Prof. Costas Vassilakis | University of Peloponnese**

## Abstract
Retrieval-augmented generation (RAG) is deployed where wrong answers are costly, yet is evaluated by average accuracy, which says nothing about the error rate on the queries a system chooses to answer. We present CASE-RAG (Certified Action Selection and Escalation), which replaces prompted self-reflection with a dedicated decision model emitting a typed action—ANSWER, RETRIEVE MORE, ABSTAIN or ESCALATE—with a probability, and releases answers only above a threshold certified by an exact-binomial selective-risk procedure. Across 116 runs on SciFact, FEVER and HotpotQA, spanning two decision models (TypeSafe’s Jev and Cloudflare’s open-weight Clef), three answer backbones (Gemma-4-31B, Qwen3.8-27B, GLM-5.3) and up to three seeds on four NVIDIA H100 NVL GPUs, it reduces the area under the risk–coverage curve by 34.6–42.9% against a heuristic router and by 26.0–40.9% against a forced-budget policy in the hosted and open-weight studies. In the three fully replicated configurations, it is the only policy that certifies a 20% error target on SciFact in every seed, answering 41–47% of queries, and the only one certifying any FEVER target below 30%. The binding property is score granularity, not calibration: self-reflection confidence takes one to eight distinct values per 300 queries, while the open-weight signal with the best recalibrated calibration error, 0.038, admits no selective certificate. We also show that applying the conformal risk control correction to selective risk is inadmissible: 28.6% of its certificates are violated on test data, against 1.6% for the exact procedure. Code and run records are released at GitHub.

## Keywords
Abstention, calibration, CASE-RAG, conformal risk control, decision models, retrieval-augmented generation, selective prediction, trustworthy artificial intelligence.

