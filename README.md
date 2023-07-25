# Cohesion Tools

[![test](https://github.com/nobu-g/cohesion-tools/actions/workflows/test.yml/badge.svg)](https://github.com/nobu-g/cohesion-tools/actions/workflows/test.yml)
[![lint](https://github.com/nobu-g/cohesion-tools/actions/workflows/lint.yml/badge.svg)](https://github.com/nobu-g/cohesion-tools/actions/workflows/lint.yml)
[![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/nobu-g/cohesion-tools)](https://www.codefactor.io/repository/github/nobu-g/cohesion-tools)
[![license](https://img.shields.io/github/license/nobu-g/cohesion-tools?color=blue)](https://github.com/nobu-g/cohesion-tools/blob/main/LICENSE)

## Requirements

- Python: 3.8+
- Dependencies: See [pyproject.toml](./pyproject.toml).


## Reference

```bibtex
@inproceedings{ueda-etal-2020-bert,
  title     = {{BERT}-based Cohesion Analysis of {J}apanese Texts},
  author    = {Ueda, Nobuhiro  and
               Kawahara, Daisuke  and
               Kurohashi, Sadao},
  booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
  month     = dec,
  year      = {2020},
  address   = {Barcelona, Spain (Online)},
  publisher = {International Committee on Computational Linguistics},
  url       = {https://aclanthology.org/2020.coling-main.114},
  doi       = {10.18653/v1/2020.coling-main.114},
  pages     = {1323--1333},
  abstract  = {The meaning of natural language text is supported by cohesion among various kinds of entities, including coreference relations, predicate-argument structures, and bridging anaphora relations. However, predicate-argument structures for nominal predicates and bridging anaphora relations have not been studied well, and their analyses have been still very difficult. Recent advances in neural networks, in particular, self training-based language models including BERT (Devlin et al., 2019), have significantly improved many natural language processing tasks, making it possible to dive into the study on analysis of cohesion in the whole text. In this study, we tackle an integrated analysis of cohesion in Japanese texts. Our results significantly outperformed existing studies in each task, especially about 10 to 20 point improvement both for zero anaphora and coreference resolution. Furthermore, we also showed that coreference resolution is different in nature from the other tasks and should be treated specially.}
}
```

```bibtex
@inproceedings{ueda-etal-2023-kwja,
  title     = {{KWJA}: A Unified {J}apanese Analyzer Based on Foundation Models},
  author    = {Ueda, Nobuhiro  and
               Omura, Kazumasa  and
               Kodama, Takashi  and
               Kiyomaru, Hirokazu  and
               Murawaki, Yugo  and
               Kawahara, Daisuke  and
               Kurohashi, Sadao},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
  month     = jul,
  year      = {2023},
  address   = {Toronto, Canada},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2023.acl-demo.52},
  pages     = {538--548},
  abstract  = {We present KWJA, a high-performance unified Japanese text analyzer based on foundation models.KWJA supports a wide range of tasks, including typo correction, word segmentation, word normalization, morphological analysis, named entity recognition, linguistic feature tagging, dependency parsing, PAS analysis, bridging reference resolution, coreference resolution, and discourse relation analysis, making it the most versatile among existing Japanese text analyzers.KWJA solves these tasks in a multi-task manner but still achieves competitive or better performance compared to existing analyzers specialized for each task.KWJA is publicly available under the MIT license at https://github.com/ku-nlp/kwja.}
}
```
