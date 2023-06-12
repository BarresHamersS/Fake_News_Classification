# Research Methodology & Experimental Details
This readme provides an overview of the research methodology and experimental details for the classification of fake news using natural language processing (NLP) techniques. For the full details please visit the corresponding paper.


## Experimental Design
An incremental examination design is employed, allowing a step-by-step breakdown of different variables. The experimental design is based on a methodology adapted from a previous study and consists of four main parts: data collection, experimental pipeline, classification, and validation.

#### Data Collection
The experiment uses the [Liar dataset](https://paperswithcode.com/dataset/liar), which is a publicly available dataset for fake news detection. The dataset contains manually labeled statements collected from PolitiFact, a non-profit organization that fact-checks the accuracy of claims. The statements are categorized into three labels: true, false, and in-between (IB) claims. The dataset is used to reduce label bias in the experiment. The distribution of claims in the dataset is as follows:

True claims: 16% (1676 statements)
False claims: 28% (2833 statements)
IB claims: 56% (5730 statements)
 
#### Experimental Pipeline
refer to notebook: Liar_data_processing_through_faKy

The NLP pipeline involves converting text objects from the Liar dataset into a machine-readable format using the spaCy library. The pipeline includes the computation of various features, including readability, information complexity (IC), sentiment, named entity recognition (NER), and part-of-speech (POS) tagging.

Readability: The Flesch-Kincaid Reading Ease (FKE) score is computed to determine the readability of a text.
Information Complexity: The Kolmogorov complexity is used to compute the information complexity of an object.
Sentiment: The Valence Sentiment Score (VSS) is used to measure the polarity and intensity of emotion in a text.
NER and POS Tagging: Named entity recognition and part-of-speech tagging are applied to identify entities and syntactic structures in the text.

#### Classification
refer to notebook: ML_classification

Three machine learning models, Naive Bayes (NB), Random Forest (RF), and Gradient Booster (GB), are trained using the computed features. These models are chosen for their effectiveness in classifying fake news. The dataset is divided into training and test data, and oversampling is applied to balance the classes. The features do not require scaling for tree-based models like RF and GB, while NB relies on feature frequencies.

#### Evaluation
refer to notebook: Feature evaluation

The evaluation involves testing the significance of differences in parametersbetween the three independent groups corresponding to the labels (true, false, and in-between). The distributions of features are analyzed, and parametric tests are conducted to ass ess differences. Continuous features are tested using the Kolmogorov-Smirnoff Test (KST), refer to notebook Distributions, while discrete features such as POS and NER tags are not applicable for parametric tests.