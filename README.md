# Drug-likeness-Space



## drug-likeness

https://github.com/AspirinCode/DrugAI_Drug-Likeness

### QED
quantitative estimation of drug-likeness

Bickerton, G., Paolini, G., Besnard, J. et al. Quantifying the chemical beauty of drugs. Nature Chem 4, 90–98 (2012). https://doi.org/10.1038/nchem.1243

### QEPPI
quantitative estimate of protein-protein interaction targeting drug-likeness

https://github.com/ohuelab/QEPPI


## Requirements

```python
Python==3.7
pytorch==1.9.0
RDKit==2020.09.10
```



## Training 

```python
python -u  train_main.py --max_epochs 25
```


## Generation
novel compound generation please follow notebook:

```python
python sampling.py --block_size=141 --vocab_size 114 --gen_size=5000
```

## Model Metrics
Molecular Sets (MOSES), a benchmarking platform to support research on machine learning for drug discovery. MOSES implements several popular molecular generation models and provides a set of metrics to evaluate the quality and diversity of generated molecules. With MOSES, MOSES aim to standardize the research on molecular generation and facilitate the sharing and comparison of new models.
https://github.com/molecularsets/moses


### QEPPI
quantitative estimate of protein-protein interaction targeting drug-likeness

https://github.com/ohuelab/QEPPI

Kosugi T, Ohue M. Quantitative estimate index for early-stage screening of compounds targeting protein-protein interactions. International Journal of Molecular Sciences, 22(20): 10925, 2021. doi: 10.3390/ijms222010925
Another QEPPI publication (conference paper)

Kosugi T, Ohue M. Quantitative estimate of protein-protein interaction targeting drug-likeness. In Proceedings of The 18th IEEE International Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB 2021), 2021. doi: 10.1109/CIBCB49929.2021.9562931 (PDF) * © 2021 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.


## Cation

Jianmin Wang, Yanyi Chu, Jiashun Mao, Hyeon-Nae Jeon, Haiyan Jin, Amir Zeb, Yuil Jang, Kwang-Hwi Cho, Tao Song, Kyoung Tai No, De novo molecular design with deep molecular generative models for PPI inhibitors, Briefings in Bioinformatics, 2022;, bbac285, https://doi.org/10.1093/bib/bbac285

