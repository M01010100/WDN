# Walsh Decomposition Network
**Matthew Townsend** \
**Thesis Spring 2026**

---

This repository is the code base for the WDN and Expierements ran. \
\_ 
https://www.researchgate.net/publication/404602294_Walsh_Decomposition_Networks_Canonical_Representations_and_Interpretability_for_Neural_Distinguishers_in_Cryptanalysis?channel=doi&linkId=69fd32c2e48e8125fa383c67&showFulltext=true

---
## Abstract
This thesis introduces Walsh Decomposition Networks (WDN), a novel neural architecture designed to be interpretable by construction. By utilizing the Walsh-Hadamard Transform(WHT) basis as the network’s architecture, WDNs directly parameterize the unique Fourier spectrum of a cipher’s distinguishing function. The model employs a Fixed Basis Layer to evaluate bitwise parities and a Trainable Coefficient Layer where the weights are the mathematically unique Walsh coefficients. To ensure stability, WDNs are trained using the Donsker-Varadhan (DV) loss, which possesses a unique population-level maximizer, forcing the network to converge to the same ”algebraic fingerprint” regardless of training seeds. Empirical results demonstrate that WDNs successfully recover the algebraic structure of ciphers. In a controlled comparison, the WDN isolated the degree-3 signature of modular addition in the ARX cipher SPECK, identifying the cubic interactions of the carry bit that are absent in the bit-oriented Feistel cipher SIMON. Furthermore, by exploiting rotational symmetries through Orbit Decomposition, the high-dimensional spectrum was compressed by 126x into a minimal canonical form. The framework also recovers ”advantage bits” directly from spectral power, grounding neural findings in classical cryptanalytic theory. Ultimately, WDNs bridge the gap between Boolean function analysis and deep learning, transforming neural cryptanalysis into a transparent and mathematically rigorous study of nonlinear structural interactions.