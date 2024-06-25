# HLIUPA-Humanizing-Language-Inference-Using-Permutational-Alignment

# Adversarial Training for NLI Models

## Project Overview
Natural Language Inference (NLI) models, especially Transformer-based architectures, have demonstrated excellent capabilities across various benchmarks. Nonetheless, recent studies have highlighted a critical flaw: these models often do not account for word order, misclassifying permuted sentences similarly to their originals. This oversight is a stark departure from human linguistic processing, where syntactic structures are essential.

## Reference Work
This project builds upon the work available at [facebookresearch/UNLU](https://github.com/facebookresearch/UNLU/tree/main), which has pioneered investigations into the impact of word order on NLI model performance.

## Objective
Our project seeks to enhance the syntactic sensitivity of NLI models to word permutations through adversarial training, aiming to align their performance closer to human language understanding.

## Approach
- **Adversarial Training:** We train models using both original and permuted sentence examples to foster resilience to syntactic perturbations.
- **Benchmark Evaluation:** We assess the models' robustness using the benchmarks established in the [UNLU repository](https://github.com/facebookresearch/UNLU/tree/main), focusing on their response to word-order changes.

## Contributing
Contributions to this project are welcome. Interested parties can fork the repository, commit modifications, and submit a pull request for review.

## License
This project is available under the MIT License. For more details, see the [LICENSE.md](LICENSE.md) file in the repository.

