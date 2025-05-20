<h1 align="center"><span style="font-weight:normal">How Hard is this Test Set? NLI Characterization by Exploiting Training Dynamics</h1>
<h2 align="center"> Accepted at "The 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024)"</h2>

<div align="center">

[Adrian Cosma](https://scholar.google.com/citations?user=cdYk_RUAAAAJ&hl=en), [Stefan Ruseti](https://scholar.google.com/citations?user=aEyJTykAAAAJ&hl=en), [Mihai Dascalu](https://scholar.google.ro/citations?user=3L9yY8UAAAAJ&hl=en), [Cornelia Caragea](https://scholar.google.com/citations?user=vkX6VV4AAAAJ&hl=en)
</div>

<div align="center">
  
[📘 Abstract](#intro)|
[⚒️ Usage](#usage)|
[📖 Citation](#citation)|
[📝 License](#license)
</div>

## <a name="intro"></a> 📘 Abstract
_Natural Language Inference (NLI) evaluation is crucial for assessing language understanding models; however, popular datasets suffer from systematic spurious correlations that artificially inflate actual model performance. To address this, we propose a method for the automated creation of a challenging test set without relying on the manual construction of artificial and unrealistic examples. We categorize the test set of popular NLI datasets into three difficulty levels by leveraging methods that exploit training dynamics. This categorization significantly reduces spurious correlation measures, with examples labeled as having the highest difficulty showing markedly decreased performance and encompassing more realistic and diverse linguistic phenomena. When our characterization method is applied to the training set, models trained with only a fraction of the data achieve comparable performance to those trained on the full dataset, surpassing other dataset characterization techniques. Our research addresses limitations in NLI dataset construction, providing a more authentic evaluation of model performance with implications for diverse NLU applications._

## <a name="usage"></a> ⚒️ Usage
```
TBD
```

## <a name="citation"></a> 📖 Citation
If you found our work useful, please cite our paper:

```
@inproceedings{cosma2024hard,
    title = "How Hard is this Test Set? {NLI} Characterization by Exploiting Training Dynamics",
    author = "Cosma, Adrian  and
      Ruseti, Stefan  and
      Dascalu, Mihai  and
      Caragea, Cornelia",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.175/",
    doi = "10.18653/v1/2024.emnlp-main.175",
    pages = "2990--3001"
}
```

## <a name="license"></a> 📝 License

This work is protected by [Attribution-NonCommercial 4.0 International](LICENSE)
