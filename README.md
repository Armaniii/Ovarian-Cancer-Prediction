<h1 align="center">
</h1>

<h4 align="center">Ovarian Cancer Prediction from Mass Spectrometry Data</h4>

<p align="center">
    <a href="https://github.com/Armaniii/Ovarian-Cancer-Prediction/commits/master">
    <img src="https://img.shields.io/github/last-commit/ArmynC/ArminC-AutoExec.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub last commit">
    <a href="https://github.com/Ovarian-Cancer-Prediction/issues">
    <img src="https://img.shields.io/github/issues-raw/ArmynC/ArminC-AutoExec.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub issues">
    <a href="https://github.com/Ovarian-Cancer-Prediction/pulls">
    <img src="https://img.shields.io/github/issues-pr-raw/ArmynC/ArminC-AutoExec.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub pull requests">
    <a href="https://twitter.com/intent/tweet?text=Try this CS:GO AutoExec:&url=https%3A%2F%2Fgithub.com%2FArmynC%2FArminC-AutoExec">

</p>
      
<p align="center">
  <a href="#Introduction">Introduction</a> •
  <a href="#Background">Background</a> •
  <a href="#Data">Data</a> •
  <a href="#Methods">Method</a> •
  <a href="#Results">Results</a>
</p>

---

## Introduction

<table>
<tr>
<td>
  
The goal of this project was to build an accurate classifier of Mass Spectrometry data that could accurately and precisely identify a sample as containing Ovarian Cancer. 


![Mass Spectrometry](https://bitesizebio.com/wp-content/uploads/2015/09/Mass-spec-1.png)
<p align="center">
<sub>(Example of Mass Spectrometry Data)</sub>
</p>

</td>
</tr>
</table>

## Background

>##### Mass Spectrometry:
 >>###### Background
   >>> The basic idea of Mass Spectrometry (MS) is to generate ions from organic compounds and seperate these ions by their mass-to-charge ratio (m/z) and their respective abundance.
 >>###### Method Used
>>>The method used to collect the dataset was a soft ionization method called SELDI-TOF. It's used for the analysis of protein mixtures, where proteins of interest in a sample become bound to a surface before MS analysis. Typically used in conjunction with time-of-flight (TOF) mass spectrometers and is used to detect proteins in tissue samples. 


##### Data:

[MS Dataset Download](https://home.ccr.cancer.gov/ncifdaproteomics/OvarianCD_PostQAQC.zip)
* 216 text files. Each file represents a sample. 
* 121 labelled as Ovarian Cancer, 95 as Normal.
  * Format: 
  
M/Z ratio  | Intensity/Abundance
------------- | -------------
4549.885  | 6.000
4549.915  |  8.000| 

  * 300,000 lines per sample. 

## Method
>### Preprocessing
>##### 1. Read in the resepctive datasets and organize into a dataframe. Create a second dataframe to hold the targets ( class ) of each file. 
>   Class Distribution
>   [Image](pics/class_distribution.png)
>   Each row in the dataframe represents the sample and each value in the row contains a tuple of the m/z and intensity.
>   [Image](pics/data_row.png)
>##### 2. Peak-Finding
>Using a threshold-based peak finder, find the top 80 peaks for each sample.
>##### Cancer
>> In the Cancer dataset, from those 80 peaks get the largest m/z index values.
>##### Normal
>> In the Normal dataset, from those 80 peaks get the smallest m/z index values.
>##### 3. Principle Component Analysis (PCA) on sample tuples. 
> Next PCA is applied as a linear dimensionality reduction technique to transform the 2D data into a single dimension. 
>##### 4. Singular Value Decomposition (SVD) for overall dimensionality reducion
> We perform SVD along with KFold Cross Validation on our 80 features to find the best features to use and as to not fall victim to the curse of dimensionality.
> [Image](pics/svd.png)
>### Model
>##### 1. Support Vector Machine
> Using 70/30 train-test split on SVD produced data
>##### 2. Deep Neural Network
>Using 70/30 train-test split on SVD produced data
>Modeled using hyperparameters 


## Results

Metric  |  SVM  |  SVM + SVD  |  DNN  |  DNN+SVD     
| :--- | ---: | :---: | :---: | :---:
Accuracy  |  82.5%  |  80.95%  | 60.31%  |  79.37%      
Precision |  82.14%  |  83.33%  | 56.25%  |  79.37%      
Recall  |  79.31%  |  78.12%  | 62.07%  |  96.89%      
