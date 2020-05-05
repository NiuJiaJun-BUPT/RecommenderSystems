# FLEN: Leveraging Field for Scalable CTR Prediction
## Main idea:
深化特征域的概念，每个特征属于各自的特征域，该论文指出，在工业中可以将这些特征继续划分特征域，例如：age, gender, occupation等都属于用户域，类似地还
可以分为项目域（item field）、内容域（context field），先对个数较多的子特征域进行聚合，形成大特征域，可以降低模型复杂度。<br>
论文提出Dicefactor的dropout机制，在两两特征交叉的时候可以drop掉两个特征embedding的某几维度，在推理阶段则不drop，Avazu上的实验效果AUC较不增加该机制
提升了0.001(**值得注意的是**，在CTR领域已公认，在性能可接受的情况下，性能0.001提升是可以接受的)
## Key sentences：
Click-Through Rate (CTR) prediction has been an indispensable component for many industrial applications,
such as recommendation systems and online advertising.
CTR prediction systems are usually based on multi-field categorical features,
i.e., every feature is categorical and belongs to one and only one field.
## Model architecture
