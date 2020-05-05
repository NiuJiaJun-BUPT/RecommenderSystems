# FLEN: Leveraging Field for Scalable CTR Prediction
## Main idea:
深化特征域的概念，每个特征属于各自的特征域，该论文指出，在工业中可以将这些特征继续划分特征域，例如：age, gender, occupation等都属于用户域，类似地还
可以分为项目域（item field）、内容域（context field），先对个数较多的子特征域进行聚合，形成大特征域，可以降低模型复杂度。<br>
论文提出Dicefactor的dropout机制，在两两特征交叉的时候可以drop掉两个特征embedding的某几维度，在推理阶段则不drop，Avazu上的实验效果AUC较不增加该机制
提升了0.001(**值得注意的是**，在CTR领域已公认，在性能可接受的情况下，性能0.001提升是可以接受的)
## Key sentences:
Click-Through Rate (CTR) prediction has been an indispensable component for many industrial applications,
such as recommendation systems and online advertising.
CTR prediction systems are usually based on multi-field categorical features,
i.e., every feature is categorical and belongs to one and only one field.
## Model architecture:
![FLEN](https://github.com/NiuJiaJun-BUPT/RecommenderSystem/blob/master/Deep%20Learning/Matching%20Function/CTR/pictures/FLEN_model.png)

引用论文中各部分进行讲解：
* The first sub-module (denoted as S) is a sub-network with one hidden layer and one output neuron. It operates on the
feature representation vectors, i.e. <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;h_S=\sum_{i=0}^{\sum_n^N{K_n}}{w[i]x[i]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;h_S=\sum_{i=0}^{\sum_n^N{K_n}}{w[i]x[i]}" title="h_S=\sum_{i=0}^{\sum_n^N{K_n}}{w[i]x[i]}" /></a> <br>
其实第一部分就是把所有特征直接过一个LR，其他CTR模型中均有该模块，因为LR的记忆性。
* The second sub-module is called an MF module, which focuses on learning inter-field feature interactions between
each pair of the hierarchical fields. 
第二部分被称为MF部分，该部分可以理解为对大的域进行交互，类似矩阵分解，矩阵分解一开始也是只输入user item的id embedding来做的。
* The third sub-module is called an FM module, which focuses on learning intra-field feature interactions in each MF
module.
第三部分就是FM部分，对大的特征域下的各子特征域进行交叉。
该部分可以理解为wide&deep中的 wide部分，只是更加复杂，在符号表述上也较难理解。

## Thoughts:
总的来说，这篇论文条理还是比较清晰的，因为公式和画图的原因，需要多看几遍才能理解模型结构。但是论文来源于工业界，有充足的实验证明这样的结构确实比FFM更好，**大小特征域，dropout机制均属于本论文的亮点**。
