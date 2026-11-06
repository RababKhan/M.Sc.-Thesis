# Reference list of the final-v2 Paper-3 manuscript (IEEE style), read by paper_build_final_v2.py.
# Rule (author request): 2022-2026 peer-reviewed journals/conferences only; no preprints or arXiv-only items.
# Every entry was checked against the publisher or proceedings record (CVF, PMLR, NeurIPS, MLSys, ACL Anthology,
# iclr.cc, ECVA/Springer; Crossref for IEEE Xplore and ACM DL). CVPR pages follow CVF Open Access. More than six
# authors -> first author et al. (IEEE).
REFS = {
    "menghani2023":
        "G. Menghani, “Efficient deep learning: A survey on making deep learning models smaller, faster, and "
        "better,” *ACM Comput. Surv.*, vol. 55, no. 12, pp. 1–37, 2023.",
    "cheng2024survey":
        "H. Cheng, M. Zhang, and J. Q. Shi, “A survey on deep neural network pruning: Taxonomy, comparison, "
        "analysis, and recommendations,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 46, no. 12, "
        "pp. 10558–10578, Dec. 2024.",
    "he2024survey":
        "Y. He and L. Xiao, “Structured pruning for deep convolutional neural networks: A survey,” *IEEE "
        "Trans. Pattern Anal. Mach. Intell.*, vol. 46, no. 5, pp. 2900–2919, May 2024.",
    "kurtic2022obert":
        "E. Kurtic *et al.*, “The Optimal BERT Surgeon: Scalable and accurate second-order pruning for large "
        "language models,” in *Proc. Conf. Empirical Methods Nat. Lang. Process. (EMNLP)*, 2022, "
        "pp. 4163–4181.",
    "chen2023fasternet":
        "J. Chen *et al.*, “Run, don’t walk: Chasing higher FLOPS for faster neural networks,” in *Proc. "
        "IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2023, pp. 12021–12031.",
    "fu2022depthshrinker":
        "Y. Fu *et al.*, “DepthShrinker: A new compression paradigm towards boosting real-hardware efficiency of "
        "compact neural networks,” in *Proc. 39th Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 162, 2022, "
        "pp. 6849–6862.",
    "han2022lasnet":
        "Y. Han *et al.*, “Latency-aware spatial-wise dynamic networks,” in *Proc. Adv. Neural Inf. Process. "
        "Syst. (NeurIPS)*, vol. 35, 2022, pp. 36845–36857.",
    "fang2023":
        "G. Fang, X. Ma, M. Song, M. B. Mi, and X. Wang, “DepGraph: Towards any structural pruning,” in "
        "*Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2023, pp. 16091–16101.",
    "shen2022halp":
        "M. Shen, H. Yin, P. Molchanov, L. Mao, J. Liu, and J. M. Alvarez, “Structural pruning via "
        "latency-saliency knapsack,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 35, 2022, "
        "pp. 12894–12908.",
    "humble2022smcp":
        "R. Humble, M. Shen, J. A. Latorre, E. Darve, and J. M. Alvarez, “Soft masking for cost-constrained "
        "channel pruning,” in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2022, pp. 641–657.",
    "hou2022chex":
        "Z. Hou *et al.*, “CHEX: CHannel EXploration for CNN model compression,” in *Proc. IEEE/CVF Conf. "
        "Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 12287–12298.",
    "li2022random":
        "Y. Li, K. Adamczewski, W. Li, S. Gu, R. Timofte, and L. Van Gool, “Revisiting random channel pruning "
        "for neural network compression,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, "
        "2022, pp. 191–201.",
    "chen2023otov2":
        "T. Chen, L. Liang, T. Ding, Z. Zhu, and I. Zharkov, “OTOv2: Automatic, generic, user-friendly,” in "
        "*Proc. Int. Conf. Learn. Represent. (ICLR)*, 2023.",
    "xia2022cofi":
        "M. Xia, Z. Zhong, and D. Chen, “Structured pruning learns compact and accurate models,” in *Proc. "
        "60th Annu. Meeting Assoc. Comput. Linguistics (ACL)*, 2022, pp. 1513–1528.",
    "yin2024owl":
        "L. Yin *et al.*, “Outlier weighed layerwise sparsity (OWL): A missing secret sauce for pruning LLMs to "
        "high sparsity,” in *Proc. 41st Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 235, 2024, "
        "pp. 57101–57115.",
    "yu2022gnnrl":
        "S. Yu, A. Mazaheri, and A. Jannesari, “Topology-aware network pruning using multi-stage graph embedding "
        "and reinforcement learning,” in *Proc. 39th Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 162, 2022, "
        "pp. 25656–25667.",
    "alwani2022decore":
        "M. Alwani, Y. Wang, and V. Madhavan, “DECORE: Deep compression with reinforcement learning,” in "
        "*Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 12349–12359.",
    "chitty2022nas":
        "K. T. Chitty-Venkata and A. K. Somani, “Neural architecture search survey: A hardware perspective,” "
        "*ACM Comput. Surv.*, vol. 55, no. 4, pp. 1–36, 2022.",
    "akhauri2024latency":
        "Y. Akhauri and M. S. Abdelfattah, “On latency predictors for neural architecture search,” in *Proc. "
        "Mach. Learn. Syst. (MLSys)*, vol. 6, 2024, pp. 512–523.",
    "dehghani2022misnomer":
        "M. Dehghani, A. Arnab, L. Beyer, A. Vaswani, and Y. Tay, “The efficiency misnomer,” in *Proc. Int. "
        "Conf. Learn. Represent. (ICLR)*, 2022.",
    "krishnakumar2022nasbenchzero":
        "A. Krishnakumar, C. White, A. Zela, R. Tu, M. Safari, and F. Hutter, “NAS-Bench-Suite-Zero: Accelerating "
        "research on zero cost proxies,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), Datasets and "
        "Benchmarks Track*, vol. 35, 2022, pp. 28037–28051.",
    "ansel2024pytorch2":
        "J. Ansel *et al.*, “PyTorch 2: Faster machine learning through dynamic Python bytecode transformation "
        "and graph compilation,” in *Proc. 29th ACM Int. Conf. Archit. Support Program. Lang. Oper. Syst. "
        "(ASPLOS)*, 2024, pp. 929–947.",
}
