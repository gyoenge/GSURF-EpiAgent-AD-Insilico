# EpiAgent Insilico Task Extension for Alzheimer
## 2025 Summer G-SURF Undergraduate Research Project

- G-SURF: GIST Summer Undergraduate Research Fellowship
- Idea: **EpiAgent AD(Alzheimer's Disease) Insilico Simulation Task with Cell-type Specific LoRA** 

### Overview 

<p align="justify" style="text-indent: 2em;">
Foundation models for single-cell epigenomics, such as EpiAgent, have recently demonstrated the ability to capture chromatin accessibility landscapes and perform in-silico perturbation tasks, including regulatory element knockouts in cancer. However, extending such frameworks to neurodegenerative diseases requires modeling brain-specific regulatory contexts and cellular heterogeneity. In this study, we <b>expand the EpiAgent in-silico treatment paradigm from cancer to Alzheimer’s disease (AD)</b> by incorporating <b>cell-type specific low-rank adapters (LoRA)</b>. These adapters enable fine-grained modulation of the pretrained foundation model, allowing differential simulation of regulatory responses across major brain cell types, including neurons, astrocytes, microglia, and oligodendrocytes. Using single-cell ATAC-seq datasets from AD and control brain tissues, we demonstrate that cell-type specific adapters enhance sensitivity in detecting disease-relevant regulatory programs and improve the fidelity of in-silico treatment simulations. Our results suggest that integrating modular adapters into foundation models provides a scalable strategy to simulate therapeutic effects in a cell-type aware manner, opening new avenues for computational drug discovery and mechanistic insight into complex brain disorders.
</p>

<p align="center">
<img width="80%" alt="github_figure" src="https://github.com/user-attachments/assets/ce749244-9037-4136-a1e4-de3a1b2a2f68" />
</p>

---

## Description
### Environment Setup

```bash
$ chmod +x ./setup.sh
$ ./setup.sh
```

### Data Preprocessing

```bash
jupyter notebook preprocess.ipynb
```

### Train EpiAgent-LoRA

```bash
jupyter notebook train.ipynb
```

### Inference EpiAgent-LoRA

```bash
jupyter notebook inference.ipynb
```

### In-silico Knock-out with EpiAgent-LoRA

```bash
jupyter notebook insilico.ipynb
```

---

### Reference (IMPORTANT!)

- Chen, X., Li, K., Cui, X., Wang, Z., Jiang, Q., Lin, J., Li, Z., Gao, Z., & Jiang, R. (2024).  
**EpiAgent: Foundation model for single-cell epigenomic data.** *bioRxiv*, 2024.12.19.629312.  
[https://doi.org/10.1101/2024.12.19.629312](https://doi.org/10.1101/2024.12.19.629312)  

- Official implementation: [https://github.com/xy-chen16/EpiAgent](https://github.com/xy-chen16/EpiAgent)
