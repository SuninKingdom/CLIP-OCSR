### Dataset Summary: CLIP-OCSR ~100k Training Dataset

This subset is provided to support the reproduction of experimental results in **Section 3.2** (*Impact of domain-adapted CLIP initialization*) and **Section 3.3** (*Data ablation study*). While the full CLIP-OCSR model utilizes a 5M corpus, this dataset represents the core diversity needed for validation.

* **Training Corpus ($n \approx 107k$):**
    * 80,000 **Synthetic Non-Markush** structure images.
    * 20,000 **Synthetic Markush** structure images (generated via **MarkushGen**).
    * 6,773 **Real-world** chemical images sourced from the USPTO.
* **Configuration:** $512 \times 512$ pixels, grayscale.

**For more details about implementation:** [https://github.com/YourUsername/CLIP-OCSR](https://github.com/YourUsername/CLIP-OCSR)  
**Original Paper:** [Insert Paper Title/DOI Here]
