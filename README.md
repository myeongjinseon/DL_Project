# Title of Your Project
Term project for the Spring 2026 Deep Learning course.

## Student name
Myeognjin Seon, 22200376

## Summary
Multimodal Machine Translation (MMT) leverages both textual and visual information to improve translation quality, especially in cases where textual input alone is ambiguous. In this project, we investigate how different types of image features influence translation performance in an MMT system. Specifically, we compare visual representations extracted from convolutional neural networks (ResNet) and multimodal models such as CLIP.

We implement a Transformer-based encoder–decoder architecture and integrate image features into the model through a simple fusion mechanism. To ensure a fair comparison, the overall model structure and training conditions are kept consistent, while only the image feature extractor is varied. This allows us to isolate the effect of different visual representations on translation quality.

Beyond overall performance evaluation using BLEU scores, we further analyze how each feature type contributes to resolving ambiguous expressions (e.g., pronouns or context-dependent phrases). Through both quantitative and qualitative analysis, this project aims to provide insights into when and why certain visual features are more effective in multimodal translation tasks.
