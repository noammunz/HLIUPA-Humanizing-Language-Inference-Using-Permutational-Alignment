# HLIUPA-Humanizing-Language-Inference-Using-Permutational-Alignment

# Adversarial Training for NLI Models

## Project Overview
Natural Language Inference (NLI) models, particularly those based on Transformer architectures, have achieved remarkable performance on various benchmarks. However, recent studies have shown a significant flaw: these models often fail to recognize the importance of word order, classifying permuted sentences similarly to their original forms. This insensitivity starkly contrasts with human language processing, where syntax is crucial.

## Objective
This project aims to address the insensitivity of NLI models to word order permutations. By implementing an adversarial training approach, we seek to enhance the syntactic sensitivity of these models, making them more robust and aligned with human linguistic capabilities.

## Approach
- **Adversarial Training:** We will train NLI models on both original and syntactically permuted examples to encourage sensitivity to word order.
- **Evaluation:** The effectiveness of this approach will be measured by comparing the performance on standard NLI benchmarks before and after adversarial training.

## Contributing
We welcome contributions to this project. Please feel free to fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
