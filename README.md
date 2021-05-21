## DSAKT-pytorch  
Pytorch Implementation of **"Application of Deep Self-Attention in Knowledge Tracing"** based on https://arxiv.org/abs/2105.07909.    
The development of intelligent tutoring system has greatly influenced the way students learn and practice, which increases their learning efficiency. The intelligent tutoring system must model learners' mastery of the knowledge before providing feedback and advices to learners, so one class of algorithm called "knowledge tracing" is surely important. This paper proposed Deep Self-Attentive Knowledge Tracing (DSAKT) based on the data of PTA, an online assessment system used by students in many universities in China, to help these students learn more efficiently. Experimentation on the data of PTA shows that DSAKT outperforms the other models for knowledge tracing an improvement of AUC by 2.1% on average, and this model also has a good performance on the ASSIST dataset.
### DSAKT model architecture  
  
<img src="https://github.com/Fusion4233919/SAKT/blob/main/dsakt.png">

## Parameters
- `window_size`: int.  
Input sequence length.  
- `dim`: int.  
Dimension of embeddings.
- `heads`: int.  
Number of heads in multi-head attention.    
- `dropout`: float.  
Dropout for feed forward layer.   
- `learn_rate`: float.
Learning rate of model.

## Citations

```bibtex
@misc{zeng2021application,
      title={Application of Deep Self-Attention in Knowledge Tracing}, 
      author={Junhao Zeng and Qingchun Zhang and Ning Xie and Bochun Yang},
      year={2021},
      eprint={2105.07909},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```



