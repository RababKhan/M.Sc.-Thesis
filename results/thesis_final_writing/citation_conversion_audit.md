# Citation conversion audit (IEEE numbered → MIST author-date)

Old numbers are those of the last IEEE-numbered assembly (commit `f01d9636`). New citations are produced
by `tools/refs_mist.py` from the same citation keys in the final chapter files.

| # | Chapter | Old IEEE citation | New author-date citation | Reference key(s) | Status |
|---|---|---|---|---|---|
| 1 | Ch. 1 | [1], [2] | (Desislavov, Martínez-Plumed and Hernández-Orallo, 2023; Luccioni, Jernite and Strubell, 2024) | `desislavov2023trends`, `luccioni2024powerhungry` | converted; entry present |
| 2 | Ch. 1 | [3] | (Menghani, 2023) | `menghani2023efficient` | converted; entry present |
| 3 | Ch. 1 | [4], [5] | (LeCun, Denker and Solla, 1989; Han et al., 2015) | `lecun1989obd`, `han2015learning` | converted; entry present |
| 4 | Ch. 1 | [6] | (Cheng, Zhang and Shi, 2024) | `cheng2024pruningsurvey` | converted; entry present |
| 5 | Ch. 1 | [7] | (He and Xiao, 2024) | `he2024structuredsurvey` | converted; entry present |
| 6 | Ch. 1 | [8] | (Hoefler et al., 2021) | `hoefler2021sparsity` | converted; entry present |
| 7 | Ch. 1 | [9] | (Lee et al., 2021) | `lee2021lamp` | converted; entry present |
| 8 | Ch. 1 | [10] | (He et al., 2018) | `he2018amc` | converted; entry present |
| 9 | Ch. 1 | [11] | (Schulman et al., 2017) | `schulman2017ppo` | converted; entry present |
| 10 | Ch. 1 | [12], [13] | (Engstrom et al., 2020; Andrychowicz et al., 2021) | `engstrom2020implementation`, `andrychowicz2021onpolicy` | converted; entry present |
| 11 | Ch. 1 | [14], [15] | (Renda, Frankle and Carbin, 2020; Le and Hua, 2021) | `renda2020rewinding`, `le2021retraining` | converted; entry present |
| 12 | Ch. 1 | [16] | (Dehghani et al., 2022) | `dehghani2022misnomer` | converted; entry present |
| 13 | Ch. 1 | [10] | (He et al., 2018) | `he2018amc` | converted; entry present |
| 14 | Ch. 1 | [17] | (Pan, Bhatia and Steinhardt, 2022) | `pan2022misspecification` | converted; entry present |
| 15 | Ch. 1 | [18] | (Blalock et al., 2020) | `blalock2020state` | converted; entry present |
| 16 | Ch. 1 | [15] | (Le and Hua, 2021) | `le2021retraining` | converted; entry present |
| 17 | Ch. 2 | [3], [8] | (Hoefler et al., 2021; Menghani, 2023) | `menghani2023efficient`, `hoefler2021sparsity` | converted; entry present |
| 18 | Ch. 2 | [3], [6], [7] | (Menghani, 2023; Cheng, Zhang and Shi, 2024; He and Xiao, 2024) | `menghani2023efficient`, `cheng2024pruningsurvey`, `he2024structuredsurvey` | converted; entry present |
| 19 | Ch. 2 | [4] | (LeCun, Denker and Solla, 1989) | `lecun1989obd` | converted; entry present |
| 20 | Ch. 2 | [5] | (Han et al., 2015) | `han2015learning` | converted; entry present |
| 21 | Ch. 2 | [19] | (Frankle and Carbin, 2019) | `frankle2019lth` | converted; entry present |
| 22 | Ch. 2 | [6] | (Cheng, Zhang and Shi, 2024) | `cheng2024pruningsurvey` | converted; entry present |
| 23 | Ch. 2 | [20] | (Kuzmin et al., 2023) | `kuzmin2023pruningvsquant` | converted; entry present |
| 24 | Ch. 2 | [16], [8] | (Hoefler et al., 2021; Dehghani et al., 2022) | `dehghani2022misnomer`, `hoefler2021sparsity` | converted; entry present |
| 25 | Ch. 2 | [18], [21] | (Blalock et al., 2020; Liu et al., 2023) | `blalock2020state`, `liu2023sparsitymaycry` | converted; entry present |
| 26 | Ch. 2 | [5], [8] | (Han et al., 2015; Hoefler et al., 2021) | `han2015learning`, `hoefler2021sparsity` | converted; entry present |
| 27 | Ch. 2 | [8], [7] | (Hoefler et al., 2021; He and Xiao, 2024) | `hoefler2021sparsity`, `he2024structuredsurvey` | converted; entry present |
| 28 | Ch. 2 | [22], [23], [24] | (Wen et al., 2016; He, Zhang and Sun, 2017; Li et al., 2017) | `li2017filters`, `he2017channel`, `wen2016ssl` | converted; entry present |
| 29 | Ch. 2 | [7] | (He and Xiao, 2024) | `he2024structuredsurvey` | converted; entry present |
| 30 | Ch. 2 | [25] | (Fang et al., 2023) | `fang2023depgraph` | converted; entry present |
| 31 | Ch. 2 | [26] | (Fang et al., 2024) | `fang2024isomorphic` | converted; entry present |
| 32 | Ch. 2 | [27] | (Hou et al., 2022) | `hou2022chex` | converted; entry present |
| 33 | Ch. 2 | [28] | (Chen et al., 2023) | `chen2023otov2` | converted; entry present |
| 34 | Ch. 2 | [29] | (El Halabi, Srinivas and Lacoste-Julien, 2022) | `halabi2022submodular` | converted; entry present |
| 35 | Ch. 2 | [30] | (Li et al., 2022) | `li2022randomchannel` | converted; entry present |
| 36 | Ch. 2 | [9], [31], [32] | (Lee et al., 2021; Liu et al., 2022; Gadhikar, Mukherjee and Burkholz, 2023) | `lee2021lamp`, `liu2022randompruning`, `gadhikar2023random` | converted; entry present |
| 37 | Ch. 2 | [33] | (Evci et al., 2020) | `evci2020rigl` | converted; entry present |
| 38 | Ch. 2 | [9] | (Lee et al., 2021) | `lee2021lamp` | converted; entry present |
| 39 | Ch. 2 | [10], [34] | (He et al., 2018; Camci et al., 2022) | `he2018amc`, `camci2022qlp` | converted; entry present |
| 40 | Ch. 2 | [35] | (Chin et al., 2020) | `chin2020legr` | converted; entry present |
| 41 | Ch. 2 | [36] | (Li et al., 2020) | `li2020eagleeye` | converted; entry present |
| 42 | Ch. 2 | [37] | (Zhang et al., 2022) | `zhang2022bip` | converted; entry present |
| 43 | Ch. 2 | [31] | (Liu et al., 2022) | `liu2022randompruning` | converted; entry present |
| 44 | Ch. 2 | [32] | (Gadhikar, Mukherjee and Burkholz, 2023) | `gadhikar2023random` | converted; entry present |
| 45 | Ch. 2 | [38] | (Yin et al., 2024) | `yin2024owl` | converted; entry present |
| 46 | Ch. 2 | [39] | (Xu et al., 2024) | `xu2024besa` | converted; entry present |
| 47 | Ch. 2 | [40] | (Diao et al., 2023) | `diao2023sparsityperspective` | converted; entry present |
| 48 | Ch. 2 | [5], [18] | (Han et al., 2015; Blalock et al., 2020) | `han2015learning`, `blalock2020state` | converted; entry present |
| 49 | Ch. 2 | [18] | (Blalock et al., 2020) | `blalock2020state` | converted; entry present |
| 50 | Ch. 2 | [41] | (Liu et al., 2019) | `liu2019rethinking` | converted; entry present |
| 51 | Ch. 2 | [21] | (Liu et al., 2023) | `liu2023sparsitymaycry` | converted; entry present |
| 52 | Ch. 2 | [9] | (Lee et al., 2021) | `lee2021lamp` | converted; entry present |
| 53 | Ch. 2 | [9] | (Lee et al., 2021) | `lee2021lamp` | converted; entry present |
| 54 | Ch. 2 | [35] | (Chin et al., 2020) | `chin2020legr` | converted; entry present |
| 55 | Ch. 2 | [38] | (Yin et al., 2024) | `yin2024owl` | converted; entry present |
| 56 | Ch. 2 | [39] | (Xu et al., 2024) | `xu2024besa` | converted; entry present |
| 57 | Ch. 2 | [42] | (Zoph and Le, 2017) | `zoph2017nas` | converted; entry present |
| 58 | Ch. 2 | [10] | (He et al., 2018) | `he2018amc` | converted; entry present |
| 59 | Ch. 2 | [10] | (He et al., 2018) | `he2018amc` | converted; entry present |
| 60 | Ch. 2 | [43], [44] | (Yang et al., 2018; Wang et al., 2019) | `yang2018netadapt`, `wang2019haq` | converted; entry present |
| 61 | Ch. 2 | [34] | (Camci et al., 2022) | `camci2022qlp` | converted; entry present |
| 62 | Ch. 2 | [45] | (Wang and Li, 2022) | `wang2022lookahead` | converted; entry present |
| 63 | Ch. 2 | [46] | (Alwani, Wang and Madhavan, 2022) | `alwani2022decore` | converted; entry present |
| 64 | Ch. 2 | [47] | (Yu, Mazaheri and Jannesari, 2022) | `yu2022gnnrl` | converted; entry present |
| 65 | Ch. 2 | [48] | (Ganjdanesh, Gao and Huang, 2024) | `ganjdanesh2024agent` | converted; entry present |
| 66 | Ch. 2 | [49] | (Yang, Chen and Sze, 2017) | `yang2017energyaware` | converted; entry present |
| 67 | Ch. 2 | [10] | (He et al., 2018) | `he2018amc` | converted; entry present |
| 68 | Ch. 2 | [43] | (Yang et al., 2018) | `yang2018netadapt` | converted; entry present |
| 69 | Ch. 2 | [9] | (Lee et al., 2021) | `lee2021lamp` | converted; entry present |
| 70 | Ch. 2 | [34] | (Camci et al., 2022) | `camci2022qlp` | converted; entry present |
| 71 | Ch. 2 | [45] | (Wang and Li, 2022) | `wang2022lookahead` | converted; entry present |
| 72 | Ch. 2 | [46] | (Alwani, Wang and Madhavan, 2022) | `alwani2022decore` | converted; entry present |
| 73 | Ch. 2 | [47] | (Yu, Mazaheri and Jannesari, 2022) | `yu2022gnnrl` | converted; entry present |
| 74 | Ch. 2 | [50] | (S. Yu et al., 2022) | `yu2022hap` | converted; entry present |
| 75 | Ch. 2 | [51] | (Shen et al., 2022) | `shen2022halp` | converted; entry present |
| 76 | Ch. 2 | [52] | (Humble et al., 2022) | `humble2022smcp` | converted; entry present |
| 77 | Ch. 2 | [27] | (Hou et al., 2022) | `hou2022chex` | converted; entry present |
| 78 | Ch. 2 | [25] | (Fang et al., 2023) | `fang2023depgraph` | converted; entry present |
| 79 | Ch. 2 | [28] | (Chen et al., 2023) | `chen2023otov2` | converted; entry present |
| 80 | Ch. 2 | [38] | (Yin et al., 2024) | `yin2024owl` | converted; entry present |
| 81 | Ch. 2 | [48] | (Ganjdanesh, Gao and Huang, 2024) | `ganjdanesh2024agent` | converted; entry present |
| 82 | Ch. 2 | [11] | (Schulman et al., 2017) | `schulman2017ppo` | converted; entry present |
| 83 | Ch. 2 | [53] | (Schulman et al., 2016) | `schulman2016gae` | converted; entry present |
| 84 | Ch. 2 | [54] | (Raffin et al., 2021) | `raffin2021sb3` | converted; entry present |
| 85 | Ch. 2 | [12] | (Engstrom et al., 2020) | `engstrom2020implementation` | converted; entry present |
| 86 | Ch. 2 | [13] | (Andrychowicz et al., 2021) | `andrychowicz2021onpolicy` | converted; entry present |
| 87 | Ch. 2 | [55] | (C. Yu et al., 2022) | `yu2022mappo` | converted; entry present |
| 88 | Ch. 2 | [4] | (LeCun, Denker and Solla, 1989) | `lecun1989obd` | converted; entry present |
| 89 | Ch. 2 | [56] | (Nonnenmacher et al., 2022) | `nonnenmacher2022sosp` | converted; entry present |
| 90 | Ch. 2 | [50] | (S. Yu et al., 2022) | `yu2022hap` | converted; entry present |
| 91 | Ch. 2 | [57] | (Yang et al., 2023) | `yang2023globalvit` | converted; entry present |
| 92 | Ch. 2 | [58], [59] | (Molchanov et al., 2017, 2019) | `molchanov2017taylor`, `molchanov2019importance` | converted; entry present |
| 93 | Ch. 2 | [60] | (Lee, Ajanthan and Torr, 2019) | `lee2019snip` | converted; entry present |
| 94 | Ch. 2 | [61], [62] | (Frantar and Alistarh, 2022, 2023) | `frantar2022obc`, `frantar2023sparsegpt` | converted; entry present |
| 95 | Ch. 2 | [63] | (Sun et al., 2024) | `sun2024wanda` | converted; entry present |
| 96 | Ch. 2 | [64] | (Yu et al., 2020) | `yu2020evalnas` | converted; entry present |
| 97 | Ch. 2 | [65] | (Oh et al., 2018) | `oh2018sil` | converted; entry present |
| 98 | Ch. 2 | [66] | (Ecoffet et al., 2021) | `ecoffet2021goexplore` | converted; entry present |
| 99 | Ch. 2 | [36] | (Li et al., 2020) | `li2020eagleeye` | converted; entry present |
| 100 | Ch. 2 | [67] | (Abdelfattah et al., 2021) | `abdelfattah2021zerocost` | converted; entry present |
| 101 | Ch. 2 | [14] | (Renda, Frankle and Carbin, 2020) | `renda2020rewinding` | converted; entry present |
| 102 | Ch. 2 | [15] | (Le and Hua, 2021) | `le2021retraining` | converted; entry present |
| 103 | Ch. 2 | [41] | (Liu et al., 2019) | `liu2019rethinking` | converted; entry present |
| 104 | Ch. 2 | [18] | (Blalock et al., 2020) | `blalock2020state` | converted; entry present |
| 105 | Ch. 2 | [68], [62] | (Kwon et al., 2022; Frantar and Alistarh, 2023) | `kwon2022posttraining`, `frantar2023sparsegpt` | converted; entry present |
| 106 | Ch. 2 | [69] | (Ma, Fang and Wang, 2023) | `ma2023llmpruner` | converted; entry present |
| 107 | Ch. 2 | [70] | (Xia et al., 2024) | `xia2024sheared` | converted; entry present |
| 108 | Ch. 2 | [71] | (Liebenwein et al., 2021) | `liebenwein2021lost` | converted; entry present |
| 109 | Ch. 2 | [72] | (Tran et al., 2022) | `tran2022disparate` | converted; entry present |
| 110 | Ch. 2 | [73] | (Iofinova et al., 2022) | `iofinova2022transfer` | converted; entry present |
| 111 | Ch. 2 | [43] | (Yang et al., 2018) | `yang2018netadapt` | converted; entry present |
| 112 | Ch. 2 | [44] | (Wang et al., 2019) | `wang2019haq` | converted; entry present |
| 113 | Ch. 2 | [74] | (Tan et al., 2019) | `tan2019mnasnet` | converted; entry present |
| 114 | Ch. 2 | [75] | (Cai et al., 2020) | `cai2020ofa` | converted; entry present |
| 115 | Ch. 2 | [51] | (Shen et al., 2022) | `shen2022halp` | converted; entry present |
| 116 | Ch. 2 | [52] | (Humble et al., 2022) | `humble2022smcp` | converted; entry present |
| 117 | Ch. 2 | [76] | (Ma et al., 2018) | `ma2018shufflenetv2` | converted; entry present |
| 118 | Ch. 2 | [16] | (Dehghani et al., 2022) | `dehghani2022misnomer` | converted; entry present |
| 119 | Ch. 2 | [8] | (Hoefler et al., 2021) | `hoefler2021sparsity` | converted; entry present |
| 120 | Ch. 2 | [7], [51] | (Shen et al., 2022; He and Xiao, 2024) | `he2024structuredsurvey`, `shen2022halp` | converted; entry present |
| 121 | Ch. 2 | [43] | (Yang et al., 2018) | `yang2018netadapt` | converted; entry present |
| 122 | Ch. 2 | [38] | (Yin et al., 2024) | `yin2024owl` | converted; entry present |
| 123 | Ch. 2 | [77] | (Strubell, Ganesh and McCallum, 2019) | `strubell2019energy` | converted; entry present |
| 124 | Ch. 2 | [78] | (Schwartz et al., 2020) | `schwartz2020greenai` | converted; entry present |
| 125 | Ch. 2 | [79] | (Henderson et al., 2020) | `henderson2020reporting` | converted; entry present |
| 126 | Ch. 2 | [80] | (Wu et al., 2022) | `wu2022sustainable` | converted; entry present |
| 127 | Ch. 2 | [81] | (Patterson et al., 2022) | `patterson2022plateau` | converted; entry present |
| 128 | Ch. 2 | [82] | (Luccioni, Viguier and Ligozat, 2023) | `luccioni2023bloom` | converted; entry present |
| 129 | Ch. 2 | [1] | (Desislavov, Martínez-Plumed and Hernández-Orallo, 2023) | `desislavov2023trends` | converted; entry present |
| 130 | Ch. 2 | [2] | (Luccioni, Jernite and Strubell, 2024) | `luccioni2024powerhungry` | converted; entry present |
| 131 | Ch. 2 | [83] | (Samsi et al., 2023) | `samsi2023wordswatts` | converted; entry present |
| 132 | Ch. 2 | [84] | (Wang et al., 2023) | `wang2023bertenergy` | converted; entry present |
| 133 | Ch. 2 | [85] | (Tschand et al., 2025) | `tschand2025mlperfpower` | converted; entry present |
| 134 | Ch. 2 | [49] | (Yang, Chen and Sze, 2017) | `yang2017energyaware` | converted; entry present |
| 135 | Ch. 2 | [86] | (Khan et al., 2018) | `khan2018rapl` | converted; entry present |
| 136 | Ch. 2 | [87] | (Jay et al., 2023) | `jay2023powermeters` | converted; entry present |
| 137 | Ch. 2 | [88] | (Raffin and Trystram, 2025) | `raffin2024dissecting` | converted; entry present |
| 138 | Ch. 2 | [82], [80] | (Wu et al., 2022; Luccioni, Viguier and Ligozat, 2023) | `luccioni2023bloom`, `wu2022sustainable` | converted; entry present |
| 139 | Ch. 2 | [9], [31], [38] | (Lee et al., 2021; Liu et al., 2022; Yin et al., 2024) | `lee2021lamp`, `liu2022randompruning`, `yin2024owl` | converted; entry present |
| 140 | Ch. 2 | [59], [50], [56] | (Molchanov et al., 2019; Nonnenmacher et al., 2022; S. Yu et al., 2022) | `molchanov2019importance`, `yu2022hap`, `nonnenmacher2022sosp` | converted; entry present |
| 141 | Ch. 2 | [10], [34], [46], [48] | (He et al., 2018; Alwani, Wang and Madhavan, 2022; Camci et al., 2022; Ganjdanesh, Gao and Huang, 2024) | `he2018amc`, `camci2022qlp`, `alwani2022decore`, `ganjdanesh2024agent` | converted; entry present |
| 142 | Ch. 2 | [64], [66] | (Yu et al., 2020; Ecoffet et al., 2021) | `yu2020evalnas`, `ecoffet2021goexplore` | converted; entry present |
| 143 | Ch. 2 | [17], [89] | (Pan, Bhatia and Steinhardt, 2022; Skalse et al., 2022) | `pan2022misspecification`, `skalse2022rewardgaming` | converted; entry present |
| 144 | Ch. 2 | [90], [91] | (Achiam et al., 2017; Gu et al., 2024) | `achiam2017cpo`, `gu2024saferl` | converted; entry present |
| 145 | Ch. 2 | [14], [15], [18] | (Blalock et al., 2020; Renda, Frankle and Carbin, 2020; Le and Hua, 2021) | `renda2020rewinding`, `le2021retraining`, `blalock2020state` | converted; entry present |
| 146 | Ch. 2 | [16], [76], [43], [49] | (Yang, Chen and Sze, 2017; Ma et al., 2018; Yang et al., 2018; Dehghani et al., 2022) | `dehghani2022misnomer`, `ma2018shufflenetv2`, `yang2018netadapt`, `yang2017energyaware` | converted; entry present |
| 147 | Ch. 2 | [86], [88] | (Khan et al., 2018; Raffin and Trystram, 2025) | `khan2018rapl`, `raffin2024dissecting` | converted; entry present |
| 148 | Ch. 3 | [92] | (Paszke et al., 2019) | `paszke2019pytorch` | converted; entry present |
| 149 | Ch. 3 | [17], [89] | (Pan, Bhatia and Steinhardt, 2022; Skalse et al., 2022) | `pan2022misspecification`, `skalse2022rewardgaming` | converted; entry present |
| 150 | Ch. 3 | [14], [15] | (Renda, Frankle and Carbin, 2020; Le and Hua, 2021) | `renda2020rewinding`, `le2021retraining` | converted; entry present |
| 151 | Ch. 3 | [8], [7] | (Hoefler et al., 2021; He and Xiao, 2024) | `hoefler2021sparsity`, `he2024structuredsurvey` | converted; entry present |
| 152 | Ch. 3 | [22], [23] | (He, Zhang and Sun, 2017; Li et al., 2017) | `li2017filters`, `he2017channel` | converted; entry present |
| 153 | Ch. 3 | [54] | (Raffin et al., 2021) | `raffin2021sb3` | converted; entry present |
| 154 | Ch. 3 | [11] | (Schulman et al., 2017) | `schulman2017ppo` | converted; entry present |
| 155 | Ch. 3 | [53] | (Schulman et al., 2016) | `schulman2016gae` | converted; entry present |
| 156 | Ch. 4 | [93] | (Krizhevsky, 2009) | `krizhevsky2009cifar` | converted; entry present |
| 157 | Ch. 4 | [94] | (LeCun et al., 1998) | `lecun1998lenet` | converted; entry present |
| 158 | Ch. 4 | [95] | (He et al., 2016) | `he2016resnet` | converted; entry present |
| 159 | Ch. 4 | [18] | (Blalock et al., 2020) | `blalock2020state` | converted; entry present |
| 160 | Ch. 4 | [33] | (Evci et al., 2020) | `evci2020rigl` | converted; entry present |
| 161 | Ch. 4 | [9] | (Lee et al., 2021) | `lee2021lamp` | converted; entry present |
| 162 | Ch. 4 | [54] | (Raffin et al., 2021) | `raffin2021sb3` | converted; entry present |
| 163 | Ch. 4 | [11] | (Schulman et al., 2017) | `schulman2017ppo` | converted; entry present |
| 164 | Ch. 4 | [53] | (Schulman et al., 2016) | `schulman2016gae` | converted; entry present |
| 165 | Ch. 4 | [96] | (Youden, 1950) | `youden1950index` | converted; entry present |
| 166 | Ch. 4 | [92] | (Paszke et al., 2019) | `paszke2019pytorch` | converted; entry present |
| 167 | Ch. 4 | [97] | (Wilcoxon, 1945) | `wilcoxon1945individual` | converted; entry present |
| 168 | Ch. 4 | [98] | (Holm, 1979) | `holm1979sequential` | converted; entry present |
| 169 | Ch. 4 | [86] | (Khan et al., 2018) | `khan2018rapl` | converted; entry present |
| 170 | Ch. 4 | [87], [88] | (Jay et al., 2023; Raffin and Trystram, 2025) | `jay2023powermeters`, `raffin2024dissecting` | converted; entry present |
| 171 | Ch. 5 | [33] | (Evci et al., 2020) | `evci2020rigl` | converted; entry present |
| 172 | Ch. 5 | [9] | (Lee et al., 2021) | `lee2021lamp` | converted; entry present |
| 173 | Ch. 6 | [8] | (Hoefler et al., 2021) | `hoefler2021sparsity` | converted; entry present |
| 174 | Ch. 6 | [16] | (Dehghani et al., 2022) | `dehghani2022misnomer` | converted; entry present |
| 175 | Ch. 6 | [22], [23], [7] | (He, Zhang and Sun, 2017; Li et al., 2017; He and Xiao, 2024) | `li2017filters`, `he2017channel`, `he2024structuredsurvey` | converted; entry present |
| 176 | Ch. 6 | [86] | (Khan et al., 2018) | `khan2018rapl` | converted; entry present |
| 177 | Ch. 6 | [87], [88] | (Jay et al., 2023; Raffin and Trystram, 2025) | `jay2023powermeters`, `raffin2024dissecting` | converted; entry present |
| 178 | Ch. 7 | [10] | (He et al., 2018) | `he2018amc` | converted; entry present |
| 179 | Ch. 7 | [64] | (Yu et al., 2020) | `yu2020evalnas` | converted; entry present |
| 180 | Ch. 7 | [66] | (Ecoffet et al., 2021) | `ecoffet2021goexplore` | converted; entry present |
| 181 | Ch. 7 | [65] | (Oh et al., 2018) | `oh2018sil` | converted; entry present |
| 182 | Ch. 7 | [17], [89] | (Pan, Bhatia and Steinhardt, 2022; Skalse et al., 2022) | `pan2022misspecification`, `skalse2022rewardgaming` | converted; entry present |
| 183 | Ch. 7 | [90] | (Achiam et al., 2017) | `achiam2017cpo` | converted; entry present |
| 184 | Ch. 7 | [9] | (Lee et al., 2021) | `lee2021lamp` | converted; entry present |
| 185 | Ch. 7 | [9] | (Lee et al., 2021) | `lee2021lamp` | converted; entry present |
| 186 | Ch. 7 | [14], [15], [18] | (Blalock et al., 2020; Renda, Frankle and Carbin, 2020; Le and Hua, 2021) | `renda2020rewinding`, `le2021retraining`, `blalock2020state` | converted; entry present |
| 187 | Ch. 7 | [8] | (Hoefler et al., 2021) | `hoefler2021sparsity` | converted; entry present |
| 188 | Ch. 7 | [16] | (Dehghani et al., 2022) | `dehghani2022misnomer` | converted; entry present |
| 189 | Ch. 7 | [86], [87], [88] | (Khan et al., 2018; Jay et al., 2023; Raffin and Trystram, 2025) | `khan2018rapl`, `jay2023powermeters`, `raffin2024dissecting` | converted; entry present |

## Final verification

- Citation occurrences converted: **189** (keys: 98).
- Numbered IEEE citations remaining after conversion (`\[[0-9]+\]`): **0**.
- Unconverted citation markers remaining: **0**.
- Citations without a bibliography entry: **0**.
- Bibliography entries never cited: **0** (the list is generated from the cited keys; 2 verified but uncited candidates were excluded).

Format checks (MIST TP.19): one author → `(Surname, Year)`; two or three → `(A, B and C, Year)`; more than three →
`(A et al., Year)`; several works → semicolon-separated, chronological; same label and year → `2022a`, `2022b`;
a citation inside an existing parenthesis is merged into it rather than nested. No corporate-author source is cited.
