# RecommenderSystem
Some academic resources in recommender system.

## Deep learning based recommender system
### Match function methods
#### Feature-based(Click-throught Rate predict methods)
| Models  | Paper Url | Datasets & Metrics(<font color="green"> AUC </font>  <font color="red"> logloss </font> <font color="blue"> RMSE </font>) |
| ------------- | ------------- | ------------- |
| AFM     | [IJCAI 2017][Attentional Factorization Machines:Learning the Weight of Feature Interactions via Attention Networks](https://www.ijcai.org/Proceedings/2017/0435.pdf)  | Movielens tag(<font color="blue"> 0.4325 </font>)<br>Frappe(<font color="blue"> 0.3102 </font>)|
| AutoInt | [CIKM 2019][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf)  | Criteo(AUC:0.8083 logloss:0.4434)<br>Avazu(AUC:0.7774 logloss:0.3811)<br>Movielens-1M(AUC:0.8488 logloss:0.3753)|
| FiBiNET | [RecSys 2019][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf) | Criteo(AUC:0.8103 logloss:0.4423)<br>Avazu(AUC:0.7832 logloss:0.3786)|
