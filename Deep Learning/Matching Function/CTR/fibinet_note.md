* 以往文献的缺陷：
  * 在计算 feature interactions 时，仅利用 Hadamard product 和 inner product，而没有考虑 feature importance
 
*	本文贡献：
  *	通过 Squeeze-Excitation network (SENET，计算机视觉领域的模型) mechanism 动态的学习 feature importance（权重）; 
  *	三种 Bilinear-Interaction layer (bilinear function) 学习 feature interactions (而不是简单的利用Hadamard product or inner product)
 
*	FiBiNET
  
  *	第三层：SENET layer
    * 给各特征embedding重新赋予权重，与attention类似
 
  * 第四层：Bilinear-Interaction layer
    *	将原始 embedding 和 SENET-Like embedding 上的二阶 feature interactions 分别建模得到p, q
    
*	浅层模型 shallow models 如 FM 和 FFM → inner product
*	深层模型 deep models 如 AFM 和 NFM → Hadamard product

 
* throughts
  SENet的想法可以借鉴，特征交叉层的思想可以借鉴，但论文所用的点积和哈达玛积的交互其实没有什么新意信息利用上不充分，
  点积和哈达玛积实质上都是对两向量的对应维度的操作。
 
 
