* **papers**<br>

| Models(点击查看阅读笔记)  | Paper Url | Datasets & Metrics |
| ------------- | ------------- | ------------- |
| AFM     | [IJCAI 2017][Attentional Factorization Machines:Learning the Weight of Feature Interactions via Attention Networks](https://www.ijcai.org/Proceedings/2017/0435.pdf)  | **Movielens tag**(RMSE:0.4325)<br> **Frappe**(RMSE:0.3102)|
| AutoInt | [CIKM 2019][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf)  | **Criteo**(AUC:0.8083 logloss:0.4434)<br>**Avazu**(AUC:0.7774 logloss:0.3811)<br>**Movielens-1M**(AUC:0.8488 logloss:0.3753)|
| FiBiNET | [RecSys 2019][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf) | **Criteo**(AUC:0.8103 logloss:0.4423)<br>**Avazu**(AUC:0.7832 logloss:0.3786)|
| [FLEN](https://github.com/NiuJiaJun-BUPT/RecommenderSystems/blob/master/Deep%20Learning/Matching%20Function/CTR/FLEN_note.md)    | [arxiv 2019][FLEN: Leveraging Field for Scalable CTR Prediction](https://arxiv.org/pdf/1911.04690.pdf) |**Avazu**(AUC:0.7519 logloss:0.3944)|
| FGCNN | [WWW 2019][Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1904.04447.pdf) | **Criteo**(AUC:0.8022 logloss:0.5388)<br>**Avazu**(AUC:0.7883 logloss:0.3746)|

* **codes**<br>
https://github.com/shenweichen/DeepCTR-Torch<br>
https://github.com/shenweichen/DeepCTR （实际实验过程中tensorflow版本的ctr成绩更好）

* **datasets**<br>
**Criteo:**<br>
    small: http://labs.criteo.com/downloads/2014-kaggle-display-advertising-challenge-dataset/<br>
    large(tb级别数据，按天划分): https://labs.criteo.com/2013/12/download-terabyte-click-logs/<br>
**Avazu：**<br>
    https://www.kaggle.com/c/avazu-ctr-prediction/data<br>
**Movielens：**<br>
    将原本ratings进行处理，大于3为1，小于3为0，中立抛去，转化为CTR任务。
    https://grouplens.org/datasets/movielens/
## Some resources
* [《搜索与推荐中的深度学习匹配》之推荐篇 ](https://zhuanlan.zhihu.com/p/45849695)——黄冠 知乎文章
* [《搜索与推荐中的深度学习匹配》之搜索篇 ](https://zhuanlan.zhihu.com/p/38296950)——黄冠 知乎文章
