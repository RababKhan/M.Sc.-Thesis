# Reference list of the revised Paper-3 manuscript (IEEE style), keyed by citation key; read by paper_build_revised.py.
# Every entry was checked against the publisher or proceedings page (CVF, OpenReview, PMLR, NeurIPS, JMLR, IEEE,
# ACM, Springer, MLSys); see the revision report.
REFS = {
    "han2015":
        "S. Han, J. Pool, J. Tran, and W. J. Dally, “Learning both weights and connections for efficient neural "
        "network,” in *Proc. Adv. Neural Inf. Process. Syst. (NIPS)*, vol. 28, 2015, pp. 1135–1143.",
    "li2017":
        "H. Li, A. Kadav, I. Durdanovic, H. Samet, and H. P. Graf, “Pruning filters for efficient ConvNets,” "
        "in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2017.",
    "hoefler2021":
        "T. Hoefler, D. Alistarh, T. Ben-Nun, N. Dryden, and A. Peste, “Sparsity in deep learning: Pruning and "
        "growth for efficient inference and training in neural networks,” *J. Mach. Learn. Res.*, vol. 22, "
        "no. 241, pp. 1–124, 2021.",
    "blalock2020":
        "D. Blalock, J. J. Gonzalez Ortiz, J. Frankle, and J. Guttag, “What is the state of neural network "
        "pruning?” in *Proc. Mach. Learn. Syst. (MLSys)*, vol. 2, 2020, pp. 129–146.",
    "ma2018":
        "N. Ma, X. Zhang, H.-T. Zheng, and J. Sun, “ShuffleNet V2: Practical guidelines for efficient CNN "
        "architecture design,” in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2018, pp. 116–131.",
    "zhang2021nnmeter":
        "L. L. Zhang, S. Han, J. Wei, N. Zheng, T. Cao, Y. Yang, and Y. Liu, “nn-Meter: Towards accurate latency "
        "prediction of deep-learning model inference on diverse edge devices,” in *Proc. 19th Annu. Int. Conf. "
        "Mobile Syst., Appl., Services (MobiSys)*, 2021, pp. 81–93.",
    "fang2023":
        "G. Fang, X. Ma, M. Song, M. B. Mi, and X. Wang, “DepGraph: Towards any structural pruning,” in "
        "*Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2023, pp. 16091–16101.",
    "he2024survey":
        "Y. He and L. Xiao, “Structured pruning for deep convolutional neural networks: A survey,” *IEEE "
        "Trans. Pattern Anal. Mach. Intell.*, vol. 46, no. 5, pp. 2900–2919, May 2024.",
    "yang2018":
        "T.-J. Yang, A. Howard, B. Chen, X. Zhang, A. Go, M. Sandler, V. Sze, and H. Adam, “NetAdapt: "
        "Platform-aware neural network adaptation for mobile applications,” in *Proc. Eur. Conf. Comput. Vis. "
        "(ECCV)*, 2018, pp. 285–300.",
    "cai2019":
        "H. Cai, L. Zhu, and S. Han, “ProxylessNAS: Direct neural architecture search on target task and "
        "hardware,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.",
    "wu2019":
        "B. Wu *et al.*, “FBNet: Hardware-aware efficient ConvNet design via differentiable neural architecture "
        "search,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 10734–10742.",
    "liu2019":
        "Z. Liu, M. Sun, T. Zhou, G. Huang, and T. Darrell, “Rethinking the value of network pruning,” in "
        "*Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.",
    "li2022random":
        "Y. Li, K. Adamczewski, W. Li, S. Gu, R. Timofte, and L. Van Gool, “Revisiting random channel pruning "
        "for neural network compression,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, "
        "2022, pp. 191–201.",
    "lee2021":
        "J. Lee, S. Park, S. Mo, S. Ahn, and J. Shin, “Layer-adaptive sparsity for the magnitude-based "
        "pruning,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.",
    "evci2020":
        "U. Evci, T. Gale, J. Menick, P. S. Castro, and E. Elsen, “Rigging the lottery: Making all tickets "
        "winners,” in *Proc. Int. Conf. Mach. Learn. (ICML)*, vol. 119, 2020, pp. 2943–2952.",
    "he2018amc":
        "Y. He, J. Lin, Z. Liu, H. Wang, L.-J. Li, and S. Han, “AMC: AutoML for model compression and "
        "acceleration on mobile devices,” in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2018, pp. 784–800.",
    "schulman2017":
        "J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization "
        "algorithms,” 2017, *arXiv:1707.06347*.",
    "shen2022halp":
        "M. Shen, H. Yin, P. Molchanov, L. Mao, J. Liu, and J. M. Alvarez, “Structural pruning via "
        "latency-saliency knapsack,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 35, 2022, "
        "pp. 12894–12908.",
    "li2021hwnas":
        "C. Li, Z. Yu, Y. Fu, Y. Zhang, Y. Zhao, H. You, Q. Yu, Y. Wang, C. Hao, and Y. Lin, “HW-NAS-Bench: "
        "Hardware-aware neural architecture search benchmark,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, "
        "2021.",
    "krizhevsky2009":
        "A. Krizhevsky, “Learning multiple layers of features from tiny images,” Univ. Toronto, Toronto, "
        "ON, Canada, Tech. Rep., 2009.",
    "he2016":
        "K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *Proc. "
        "IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.",
    "paszke2019":
        "A. Paszke *et al.*, “PyTorch: An imperative style, high-performance deep learning library,” in "
        "*Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019, pp. 8024–8035.",
}
