### 🕸️ Literary Complexity

This is the repository containing code & data for our paper on literary multidimensional complexity.

#### Scripts
To examine the relation between features at the stylistic/syntactic and sentiment level, see ```part1_relations_features.py```

To reproduce results on the relations between features as human perceived complexity, see the scripts:
- reading time experiment in ```part2_reading_time_vs_features.py```
- difficulty rank experiment in ```part2_difficulty_rank_vs_features.py```

#### In ```data``` you will find:
- The [Chicago Corpus dataset](https://github.com/centre-for-humanities-computing/chicago_corpus) of features & meta-data
- The filtered word reaction times from the [Natural Stories corpus](https://github.com/languageMIT/naturalstories/)
- The list of 200 novels' difficulty rank from the paper of [Dalvean & Enkhbayar, 2018](https://dx.doi.org/10.2139/ssrn.3097706)

...


#### _Please use the following references if you utilize any of these data sources:_

For the _Difficulty Rank_ list:
```
@article{dalvean_new_2018,
	title = {A {New} {Text} {Readability} {Measure} for {Fiction} {Texts}},
	issn = {1556-5068},
	url = {https://www.ssrn.com/abstract=3097706},
	doi = {10.2139/ssrn.3097706},
	abstract = {English teachers often have difficulty matching the complexity of fiction texts with students' reading levels. Texts that seem appropriate for students of a given level can turn out to be too difficult. Furthermore, it is difficult to choose a series of texts that represent a smooth gradation of text difficulty. This paper attempts to address both problems by providing a complexity ranking of a corpus of 200 fiction texts consisting of 100 adults' and 100 children's texts. Using machine learning, several standard readability measures are used as variables to create a classifier which is able to classify the corpus with an accuracy of 84\%. A classifier created with linguistic variables is able to classify the corpus with an accuracy of 89\%. The 'latter classifier is then used to provide a linear complexity rank for each text. The resulting ranking instantiates a fine-grained increase in complexity. This can be used by a reading or ESL teacher to select a sequence of texts that represent an increasing challenge to a student without there being a frustratingly perceptible increase in difficulty.},
	language = {en},
	urldate = {2023-08-29},
	journal = {SSRN Electronic Journal},
	author = {Dalvean, Michael Coleman and Enkhbayar, Galbadrakh},
	year = {2018},
	file = {Dalvean and Enkhbayar - 2018 - A New Text Readability Measure for Fiction Texts.pdf:/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/Zotero/storage/84DFB6LW/Dalvean and Enkhbayar - 2018 - A New Text Readability Measure for Fiction Texts.pdf:application/pdf},
}
```

Fot the _Natural Stories corpus_:
```
@article{futrell2021natural,
author={Richard Futrell and Edward Gibson and Harry J. Tily and Idan Blank and Anastasia Vishnevetsky and Steven T. Piantadosi and Evelina Fedorenko},
year={2021},
title={The Natural Stories corpus: A reading-time corpus of English texts containing rare syntactic constructions},
journal={Language Resources and Evaluation},
volume={55},
number={1},
pages={63--77}}
```

For the _Chicago Corpus_ dataset of features:
```
@inproceedings{bizzoni-etal-2024-matter,
    title = "A Matter of Perspective: Building a Multi-Perspective Annotated Dataset for the Study of Literary Quality",
    author = "Bizzoni, Yuri  and
      Moreira, Pascale Feldkamp  and
      Lassen, Ida Marie S.  and
      Thomsen, Mads Rosendahl  and
      Nielbo, Kristoffer",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.71",
    pages = "789--800",
    abstract = "Studies on literary quality have constantly stimulated the interest of critics, both in theoretical and empirical fields. To examine the perceived quality of literary works, some approaches have focused on data annotated through crowd-sourcing platforms, and others relied on available expert annotated data. In this work, we contribute to the debate by presenting a dataset collecting quality judgments on 9,000 19th and 20th century English-language literary novels by 3,150 predominantly Anglophone authors. We incorporate expert opinions and crowd-sourced annotations to allow comparative analyses between different literary quality evaluations. We also provide several textual metrics chosen for their potential connection with literary reception and engagement. While a large part of the texts is subjected to copyright, we release quality and reception measures together with stylometric and sentiment data for each of the 9,000 novels to promote future research and comparison.",
}
```
