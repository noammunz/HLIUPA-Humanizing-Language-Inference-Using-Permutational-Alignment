# HLIUPA: Humanizing Language Inference Using Permutational Alignment

## Project Overview

Natural Language Inference (NLI) is a fundamental task in natural language processing that assesses the relationship between two sentences (premise and hypothesis). While state-of-the-art Transformer-based models have achieved impressive performance on NLI benchmarks, recent research has identified a critical limitation: these models often disregard word order, treating permuted sentences similarly to their original forms—a behavior significantly divergent from human language processing.

This repository presents **HLIUPA** (Humanizing Language Inference Using Permutational Alignment), an extension of prior work that addresses this limitation through adversarial training techniques designed to enhance syntactic sensitivity in NLI models.

## Research Background

Our work builds upon the foundation established by Sinha et al. (2021) in their paper ["Unnatural Language Inference"](https://arxiv.org/abs/2101.00010) and the associated [codebase](https://github.com/facebookresearch/UNLU/). While their research identified the permutation invariance problem, our approach focuses on remediation strategies to develop more syntactically-aware models.

## Key Contributions

- **Advanced Adversarial Training Methodology**: We implement and evaluate fine-tuned approaches to adversarial training with permuted sentences to build robust NLI models
- **Comprehensive Evaluation Framework**: Our evaluation assesses model performance on both original and permuted inputs using metrics including:
  - Standard accuracy
  - Omega-max (ability to correctly classify at least one permutation)
  - Omega-rand (proportion of examples where the model correctly classifies most permutations)
  - PC and PF metrics (permutation consistency for correctly and incorrectly classified examples)
- **Multiple Model Architectures**: We compare performance across RoBERTa, DistilBERT, and BART architectures to evaluate permutation sensitivity across different model designs

## Implementation

Our implementation includes:

- `mnli_preprocessor.py`: Handles data preparation, tokenization, and permutation generation
- `train.py`: Implements adversarial training methods with support for multiple model architectures
- `evaluate.py`: Provides comprehensive evaluation with metrics for assessing permutation robustness

## Usage

The repository can be used to:

1. Train NLI models with adversarial permutation examples
2. Evaluate model robustness to word-order changes
3. Compare performance across different model architectures and training strategies

## Future Directions

- Exploration of alternative permutation strategies beyond random shuffling
- Investigation of syntactic knowledge transfer across languages
- Integration with interpretability techniques to better understand how models process word order

## Citation

If you use this code or findings in your research, please cite our work as well as the original UNLU paper:

```bibtex
@article{sinha2021unnatural,
  title={Unnatural Language Inference},
  author={Sinha, Koustuv and Jia, Robin and Hupkes, Dieuwke and Pineau, Joelle and Williams, Adina and Kiela, Douwe},
  journal={arXiv preprint arXiv:2101.00010},
  year={2021}
}
```
