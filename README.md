# Paper_CLASS-M
We provide our code and explanations of our CLASS-M model for semi-supervised classification on TCGA ccRCC dataset based on algorithms introduced in https://arxiv.org/abs/2312.06978

Please go into each section and check each section's readme file for details.

Brief introduction of each section:

Section1: we provide annotations to 420 TCGA ccRCC Whole Slide Images (WSIs), and also our code for cropping tiles from WSIs based on annotations to generate our dataset. 

Section2: we provide our original code for performing adaptive stain separation. The result will be used in our CLASS-M model. 

Section3: we provide code for our CLASS-M model. The model will read images generated from Section1 and stain separation matrices calculated in Section2 to do semi-supervised classification tasks. 
