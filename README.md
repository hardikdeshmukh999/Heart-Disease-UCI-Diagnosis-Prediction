## License
[MIT LICENSE](https://github.com/smarthardik10/Heart-Disease-UCI-Diagnosis-Prediction/blob/master/LICENSE)

[MIT](https://choosealicense.com/licenses/mit/)


## Links

Blog link:
[![Medium: ](https://img.icons8.com/ios-filled/25/000000/medium-monogram.png)](https://towardsdatascience.com/heart-disease-uci-diagnosis-prediction-b1943ee835a7)

Notebook link:
[![Open In Colab: ](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16iFRPq0vx_CZypo4ZyJ_qTLrvds3FlDb?usp=sharing)

## Blog - Towards Data Science

# Heart Disease UCI-Diagnosis & Prediction

### *Prediction using Logistic Regression with 87% accuracy*

![Photo by [Robina Weermeijer](https://unsplash.com/@averey?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://cdn-images-1.medium.com/max/12000/0*P-EN5Z4l3LbWg4V8)

Every day, the average human heart beats around 100,000 times, pumping 2,000 gallons of blood through the body. Inside your body there are 60,000 miles of blood vessels.

The signs of a woman having a heart attack are much less noticeable than the signs of a male. In women, heart attacks may feel uncomfortable squeezing, pressure, fullness, or pain in the center of the chest. It may also cause pain in one or both arms, the back, neck, jaw or stomach, shortness of breath, nausea and other symptoms. Men experience typical symptoms of heart attack, such as chest pain , discomfort, and stress. They may also experience pain in other areas, such as arms, neck , back, and jaw, and shortness of breath, sweating, and discomfort that mimics heartburn.

It’s a lot of work for an organ which is just like a large fist and weighs between 8 and 12 ounces.

**Code by Hardik:**

Link to the Google colab notebook: [https://colab.research.google.com/drive/16iFRPq0vx_CZypo4ZyJ_qTLrvds3FlDb?usp=sharing](https://colab.research.google.com/drive/16iFRPq0vx_CZypo4ZyJ_qTLrvds3FlDb?usp=sharing)

GitHub: [https://github.com/smarthardik10/Heart-Disease-UCI-Diagnosis-Prediction](https://github.com/smarthardik10/Heart-Disease-UCI-Diagnosis-Prediction)

**Dataset by Heart Disease UCI:**

Dataset source: [https://archive.ics.uci.edu/ml/datasets/Heart+Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

**Dataset columns:**

* age: The person’s age in years

* sex: The person’s sex (1 = male, 0 = female)

* cp: chest pain type
 — Value 0: asymptomatic
 — Value 1: atypical angina
 — Value 2: non-anginal pain
 — Value 3: typical angina

* trestbps: The person’s resting blood pressure (mm Hg on admission to the hospital)

* chol: The person’s cholesterol measurement in mg/dl

* fbs: The person’s fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)

* restecg: resting electrocardiographic results
 — Value 0: showing probable or definite left ventricular hypertrophy by Estes’ criteria
 — Value 1: normal
 — Value 2: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)

* thalach: The person’s maximum heart rate achieved

* exang: Exercise induced angina (1 = yes; 0 = no)

* oldpeak: ST depression induced by exercise relative to rest (‘ST’ relates to positions on the ECG plot. See more here)

* slope: the slope of the peak exercise ST segment — 0: downsloping; 1: flat; 2: upsloping
0: downsloping; 1: flat; 2: upsloping

* ca: The number of major vessels (0–3)

* thal: A blood disorder called thalassemia Value 0: NULL (dropped from the dataset previously
Value 1: fixed defect (no blood flow in some part of the heart)
Value 2: normal blood flow
Value 3: reversible defect (a blood flow is observed but it is not normal)

* target: Heart disease (1 = no, 0= yes)

**Context:**

This is multivariate type of dataset which means providing or involving a variety of separate mathematical or statistical variables, multivariate numerical data analysis. It is composed of 14 attributes which are age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise induced angina, oldpeak — ST depression induced by exercise relative to rest, the slope of the peak exercise ST segment, number of major vessels and Thalassemia. This database includes 76 attributes, but all published studies relate to the use of a subset of 14 of them. The Cleveland database is the only one used by ML researchers to date. One of the major tasks on this dataset is to predict based on the given attributes of a patient that whether that particular person has a heart disease or not and other is the experimental task to diagnose and find out various insights from this dataset which could help in understanding the problem more.

**The dataset was created by: -**

1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
 2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
 3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
 4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

### Table of Content

 1. [Imports and Reading Dataset](#a3db)

 2. [Data Description](#1257)

* [Describe](#0140)

* [Null](#de14)

3. [Data Analysis](#d39b)

* [1) Univariate Selection](#70b1)

* [2) Feature Selection](#70f9)

* [3) Correlation Matrix with Heatmap](#2bcc)

4. [Data Visualization](#82cd)

* 1)[Countplot](#61e6)

* 2)[Distplot](#ecb0)

* 3)[Jointplot](#077e)

* 4)[Boxplot/violinplot](#23b0)

* 5)[Cluster map](#c467)

* 6)[Pairplot](#36ad)

* [Classification Tree](#3ec5)

5. [Data Pre-processing](#ba87)

* 1)[Pre-processing](#5ada)

* 2)[One Hot Encoding](#3cd8)

6. [Logistic Regression](#7e86)

* 1)[Gather columns](#2bd0)

* 2)[Splitting Data](#f346)

* 3)[Normalization](#1edc)

* 4)[Fitting into the model](#b904)

* 5)[Prediction](#c22c)

* 6)[Model Evaluation](#0621)

7. [Conclusion](#7f6e)

* 1)[Coefficients](#4600)

* 2)[Analysis](#da70)

* 3)[Conculsuion](#d3f0)

## 1. Imports and Reading Dataset

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    %matplotlib inline

    df = pd.read_csv('/content/drive/My Drive/dataset/heart.csv')

    df.head()

![heart.csv](https://cdn-images-1.medium.com/max/2000/1*FChFX1q0zMKFSoF585Fz3g.png)

## 2. Data Description

### Describe

There has been lot of confusion about the meta data, as there are various different meta data available out there. Over here below I have got the two most used meta data description from kaggle. So we are going to follow the second description(2 — description).

1 — description

    It's a clean, easy to understand set of data. However, the meaning of some of the column headers are not obvious. Here's what they mean,
    •    age: The person's age in years
    •    sex: The person's sex (1 = male, 0 = female)
    •    cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
    •    trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
    •    chol: The person's cholesterol measurement in mg/dl
    •    fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
    •    restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
    •    thalach: The person's maximum heart rate achieved
    •    exang: Exercise induced angina (1 = yes; 0 = no)
    •    oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
    •    slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
    •    ca: The number of major vessels (0-3)
    •    thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
    •    target: Heart disease (0 = no, 1 = yes)

2 — description

    I chose this dataset for a final project on a multivariate statistics course I'm taking, and for the last couple of days I've been struggling to get the appropriate description of the features.
    For some unknown reason, the dataset for download on Kaggle is VERY different from the one you can download at https://archive.ics.uci.edu/ml/datasets/heart+Disease
    And what's worse: the description here on Kaggle is the same as the one in the Cleveland page, that means every interpretation you make based on the Kaggle dataset is WRONG.
    So here it goes, the CORRECT description of the kaggle dataset.
     
    cp: chest pain type
    -- Value 0: asymptomatic
    -- Value 1: atypical angina
    -- Value 2: non-anginal pain
    -- Value 3: typical angina
     
    restecg: resting electrocardiographic results
    -- Value 0: showing probable or definite left ventricular hypertrophy by Estes' criteria
    -- Value 1: normal
    -- Value 2: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
     
    slope: the slope of the peak exercise ST segment
    0: downsloping; 1: flat; 2: upsloping
     
    thal
    Results of the blood flow observed via the radioactive dye.
     
    Value 0: NULL (dropped from the dataset previously)
    Value 1: fixed defect (no blood flow in some part of the heart)
    Value 2: normal blood flow
    Value 3: reversible defect (a blood flow is observed but it is not normal)
    This feature and the next one are obtained through a very invasive process for the patients. But, by themselves, they give a very good indication of the presence of a heart disease or not.
     
    target (maybe THE most important feature): 0 = disease, 1 = no disease
     
    A few more things to consider:
    data #93, 139, 164, 165 and 252 have ca=4 which is incorrect. In the original Cleveland dataset they are NaNs (so they should be removed)
    data #49 and 282 have thal = 0, also incorrect. They are also NaNs in the original dataset.
     
    I'll copy a sentence so you get more insight about the "thal" column (thal is for Thalium, a radioactive tracer injected during a stress test):
    --Nuclear stress testing requires the injection of a tracer, commonly technicium 99M (Myoview or Cardiolyte), which is then taken up by healthy, viable myocardial cells. A camera (detector) is used afterwards to image the heart and compare segments. A coronary stenosis is detected when a myocardial segment takes up the nuclear tracer at rest, but not during cardiac stress. This is called a "reversible defect." Scarred myocardium from prior infarct will not take up tracer at all and is referred to as a "fixed defect." --
     
    You can check all of this by comparing the Kaggle and the UCI datasets. Feel free to ask/correct/comment/say hi.
     
    To open a .data file, change the extension to a .txt and then open it with excel or similars.
     

    df.info()

![](https://cdn-images-1.medium.com/max/2000/1*yxC3bm0HVtC_D2vHnCX91w.png)

    df.describe()

![](https://cdn-images-1.medium.com/max/2914/1*UWNBo9FyXb0xAwX0fIFPhw.png)

### Null

Checking for null values

    df.isnull().sum()

![](https://cdn-images-1.medium.com/max/2000/1*dEG2y85mRD4rKildsE42cw.png)

    #visualizing Null values if it exists 
    plt.figure(figsize=(22,10))

    plt.xticks(size=20,color='grey')
    plt.tick_params(size=12,color='grey')

    plt.title('Finding Null Values Using Heatmap\n',color='grey',size=30)

    sns.heatmap(df.isnull(),
                yticklabels=False,
                cbar=False,
                cmap='PuBu_r',
                )

![](https://cdn-images-1.medium.com/max/2842/1*4fSN0cVrxWSxO1a1qUwyfA.png)

Dataset has no null values

### pandas-profiling

    !pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip

    import pandas_profiling as pp
    pp.ProfileReport(df)

![](https://cdn-images-1.medium.com/max/3818/1*miyjIt6hf5wjgAKDZJZ2ZA.png)

For detailed view: [https://colab.research.google.com/drive/16iFRPq0vx_CZypo4ZyJ_qTLrvds3FlDb#scrollTo=5crIBqgn9FiI&line=1&uniqifier=1](https://colab.research.google.com/drive/16iFRPq0vx_CZypo4ZyJ_qTLrvds3FlDb#scrollTo=5crIBqgn9FiI&line=1&uniqifier=1)

## 3. Data Analysis

**Feature Selection**

 1. Univariate Selction — Statistical tests may be used to pick certain features that have the best relationship to the performance variable.
The scikit-learn library provides the SelectKBest class that can be used to select a specific number of features in a suite of different statistical tests.
The following example uses the chi-squared (chi2) statistical test for non-negative features to select 10 of the best features from the Mobile Price Range Prediction Dataset.

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    data = df.copy()
    X = data.iloc[:,0:13]  #independent columns
    y = data.iloc[:,-1]    #target column i.e price range
    #apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(12,'Score'))  #print 10 best features

![](https://cdn-images-1.medium.com/max/2000/1*JL8RAzx-lPaRWIUebanpkA.png)

2. Feature Importance — You can gain the significance of each feature of your dataset by using the Model Characteristics property.
Feature value gives you a score for every function of your results, the higher the score the more significant or appropriate the performance variable is.
Feature importance is the built-in class that comes with Tree Based Classifiers, we will use the Extra Tree Classifier to extract the top 10 features for the dataset.

    from sklearn.ensemble import ExtraTreesClassifier

    model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(13).plot(kind='barh')
    plt.show()

![](https://cdn-images-1.medium.com/max/2000/1*7VsCFn67kr0qeCdzXyHjxw.png)

3. Correlation Matrix with Heatmap — Correlation indicates how the features are related to each other or to the target variable.
The correlation may be positive (increase in one value of the feature increases the value of the target variable) or negative (increase in one value of the feature decreases the value of the target variable)
Heatmap makes it easy to classify the features are most relevant to the target variable, and we will plot the associated features of the heatmap using the seaborn library.

Correlation shows whether the characteristics are related to each other or to the target variable. Correlation can be positive (increase in one value, the value of the objective variable increases) or negative (increase in one value, the value of the target variable decreased). From this heatmap we can observe that the ‘cp’ chest pain is highly related to the target variable. Compared to relation between other two variables we can say that chest pain contributes the most in prediction of presences of a heart disease. Medical emergency is a heart attack. A cardiac occurs usually when blood clot blocks blood flow to the cardiac. Tissue loses oxygen without blood and dies causing chest pain.

    plt.figure(figsize=(12,10))
    sns.heatmap(df.corr(),annot=True,cmap="magma",fmt='.2f')

![](https://cdn-images-1.medium.com/max/2000/1*OJWYxl9iZz3P6airL5fmgA.png)

    for i in df.columns:
        print(i,len(df[i].unique()))

![](https://cdn-images-1.medium.com/max/2000/1*86e8b8XzoBIREqwNZQs7Vg.png)

## 4. Data Visualization

**Seaborn**

    sns.set_style('darkgrid')
    sns.set_palette('Set2')

Preparing Data

    df2 = df.copy()

    def chng(sex):
        if sex == 0:
            return 'female'
        else:
            return 'male'

    df2['sex'] = df2['sex'].apply(chng)

    def chng2(prob):
        if prob == 0:
            return ‘Heart Disease’
        else:
            return ‘No Heart Disease’

    df2['target'] = df2['target'].apply(chng2)

### 1. Countplot

    df2['target'] = df2['target'].apply(chng2)
    sns.countplot(data= df2, x='sex',hue='target')
    plt.title('Gender v/s target\n')

![](https://cdn-images-1.medium.com/max/2000/1*Frqo3efoecW7-FKQO74rOQ.png)

According to this Cleveland dataset males are more susceptible to get Heart Disease than females. Men experience heart attacks more than women. Sudden Heart Attacks are experienced by men between 70% — 89%. Woman may experience a heart attack with no chest pressure at all, they usually experience nausea or vomiting which are often confused with acid reflux or the flu.

    sns.countplot(data= df2, x='cp',hue='target')
    plt.title('Chest Pain Type v/s target\n')

![](https://cdn-images-1.medium.com/max/2000/1*RBxhQbotl_CtbE0olR0awg.png)

There are four types of chest pain, asymptomatic, atypical angina, non-anginal pain and typical angina. Most of the Heart Disease patients are found to have asymptomatic chest pain. These group of people might show atypical symptoms like indigestion, flu or a strained chest muscle. A asymptomatic attack, like any heart attack, involves, blockage of blood flow to your heart and possible damage to the heart muscle. The risk factors for asymptomatic heart attacks are same as those with heart symptoms. These factors include:

· Age

· Diabetes

· Excess weight

· Family History of Heart Disease

· High Blood Pressure

· High cholesterol

· Lack of exercise

· Prior Heart attack

· Tobacco use

Asymptomatic Heart attack puts you at a greater risk of having another heart attack which could be d deadly. Having another heart attack also increases your risk of complications, such as heart failure. There are no test to determine your potential for asymptomatic heart attack. The only way to tell If you had asymptomatic attack is by an electrocardiogram or echocardiogram. These tests can reveal changes that signal a heart attack.

    sns.countplot(data= df2, x='sex',hue='thal')
    plt.title('Gender v/s Thalassemia\n')
    print('Thalassemia (thal-uh-SEE-me-uh) is an inherited blood disorder that causes your body to have less hemoglobin than normal. Hemoglobin enables red blood cells to carry oxygen')

![](https://cdn-images-1.medium.com/max/2000/1*tTTn5oE0Jmjpr4QTwf3xsw.png)

The Beta thalassemia cardiomyopathy is mainly characterized by two distinct pheno types , dilated type, with left ventricular dilatation and impaired contractility and a restrictive pheno type, with restrictive left ventricular feeling , pulmonary hyper tension and right heart failure. Heart problems, congestive heart failures and abnormal heart rhythms can be associated with severe thalassemia.

    sns.countplot(data= df2, x='slope',hue='target')
    plt.title('Slope v/s Target\n')

![](https://cdn-images-1.medium.com/max/2000/1*41GDC-L_TbvbDQCtoRuZwg.png)

    sns.countplot(data= df2, x='exang',hue='thal')
    plt.title('exang v/s Thalassemia\n')

![](https://cdn-images-1.medium.com/max/2000/1*MpGogTHk2UYrLDzw7JMN8Q.png)

### 2. Distplot

    plt.figure(figsize=(16,7))
    sns.distplot(df[df['target']==0]['age'],kde=False,bins=50)
    plt.title('Age of Heart Diseased Patients\n')

![](https://cdn-images-1.medium.com/max/2000/1*srpiDM9ZzegmZmxm2lA7IA.png)

Heart Disease is very common in the seniors which is composed of age group 60 and above and common among adults which belong to the age group of 41 to 60. But it’s rare among the age group of 19 to 40 and very rare among the age group of 0 to 18.

    plt.figure(figsize=(16,7))
    sns.distplot(df[df['target']==0]['chol'],kde=False,bins=40)
    plt.title('Chol of Heart Diseased Patients\n')

![](https://cdn-images-1.medium.com/max/2000/1*JZpR2DHBWlDRwOj2o8Ls6g.png)

* Total cholesterol

* LDL — ‘bad cholesterol”

* HDL — ‘good cholesterol”

In adults, the total cholesterol levels are considered desirable less than 200 milligram per decilitre ( mg / dL). Borderlines are considered to be high between 200 to 239 mg / dL and 240 mg / dL and above. LDL should contain less than 100 mg / dL of cholesterol. 100 mg / dl rates for individuals without any health issue are appropriate but may be more relevant for those with cardiac problems or risk factors for heart disease. The levels are borderline moderate between 130 and 159 mg / dL and moderate between 160 and 189 mg / dL. The reading is very high at or above 190 mg / dL. Levels of HDL are to be maintained higher. The risk factor for cardiovascular diseases is called a reading less than 40 mg / dL. Borderline low is considered to be between 41 mg / dL and 59 mg / dL. The HDL level can be measured with a maximum of 60 mg / dL.

    plt.figure(figsize=(16,7))
    sns.distplot(df[df['target']==0]['thalach'],kde=False,bins=40)
    plt.title('thalach of Heart Diseased Patients\n')

![](https://cdn-images-1.medium.com/max/2000/1*eSAbEyeHF2RvAgY2NGME9w.png)

### 3. Jointplot

Preparing data

    df3 = df[df['target'] == 0 ][['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
           'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']] 
    #target 0 - people with heart disease

    pal = sns.light_palette("blue", as_cmap=True)

    print('Age vs trestbps(Heart Diseased Patinets)')
    sns.jointplot(data=df3,
                  x='age',
                  y='trestbps',
                  kind='hex',
                  cmap='Reds'
               
                  )

![](https://cdn-images-1.medium.com/max/2000/1*HfoR09Ivh91PZAROLGwNBQ.png)

    sns.jointplot(data=df3,
                  x='chol',
                  y='age',
                  kind='kde',
                  cmap='PuBu'
                  )

![](https://cdn-images-1.medium.com/max/2000/1*1ot8W_nVB73yrWuHsrAn8w.png)

Joint plots in seaborn helps us to understand the trend seen among two features. As observed from the above plot we can see that most of the Heart diseased patients in their age of upper 50s or lower 60s tend to have Cholesterol between 200mg/dl to 300mg/dl.

    sns.jointplot(data=df3,
                  x='chol',
                  y='trestbps',
                  kind='resid',
                 
                  )

![](https://cdn-images-1.medium.com/max/2000/1*Ts0mlN8Boh9neGxhBwQoeA.png)

### 4. Boxplot / violinplot

    sns.boxplot(data=df2,x='target',y='age')

![](https://cdn-images-1.medium.com/max/2000/1*h0pHBGQNlrlNquS5ifXCQg.png)

    plt.figure(figsize=(14,8))
    sns.violinplot(data=df2,x='ca',y='age',hue='target')

![](https://cdn-images-1.medium.com/max/2000/1*ipsuoIwrltp1pmVKP_WvoA.png)

    sns.boxplot(data=df2,x='cp',y='thalach',hue='target')

![](https://cdn-images-1.medium.com/max/2000/1*npZjW1eWpomZl4E_vYWTyw.png)

    plt.figure(figsize=(10,7))
    sns.boxplot(data=df2,x='fbs',y='trestbps',hue='target')

![](https://cdn-images-1.medium.com/max/2000/1*Rxy38lPAQ-zyzLpx3ZGiTg.png)

    plt.figure(figsize=(10,7))
    sns.violinplot(data=df2,x='exang',y='oldpeak',hue='target')

![](https://cdn-images-1.medium.com/max/2000/1*EJPC677y31PcLdDjN5NTaQ.png)

    plt.figure(figsize=(10,7))
    sns.boxplot(data=df2,x='slope',y='thalach',hue='target')

![](https://cdn-images-1.medium.com/max/2000/1*UhH1dECGke9Fgrpt3NwyJA.png)

    sns.violinplot(data=df2,x='thal',y='oldpeak',hue='target')

![](https://cdn-images-1.medium.com/max/2000/1*HGIPA_to-lsXWssal7B0eA.png)

    sns.violinplot(data=df2,x='target',y='thalach')

![](https://cdn-images-1.medium.com/max/2000/1*Yn39mFkfdphzjjjKZmFGIg.png)

### 5. Clusterplot

    sns.clustermap(df.corr(),annot=True)

![](https://cdn-images-1.medium.com/max/2000/1*C9A7UBboA25i5zyHGq_FXQ.png)

### 6. Pairplot

    sns.pairplot(df,hue='cp')

![](https://cdn-images-1.medium.com/max/4788/1*_CARlkLO_NKpeXWnHLr2ZA.png)

### Classification Tree

    from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
    from sklearn.model_selection import train_test_split # Import train_test_split function
    from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
    X = df.iloc[:,0:13] # Features
    y = df.iloc[:,13] # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

Accuracy: 0.7142857142857143

    feature_cols = ['age', 'sex', 'cp', 'trestbps','chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']

    from sklearn.tree import export_graphviz
    from sklearn.externals.six import StringIO  
    from IPython.display import Image  
    import pydotplus

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = feature_cols  ,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('diabetes.png')
    Image(graph.create_png())

![](https://cdn-images-1.medium.com/max/4942/1*BdWLh3XlVwJhgv3X93eprw.png)

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

Accuracy: 0.7362637362637363

    from sklearn.externals.six import StringIO  
    from IPython.display import Image  
    from sklearn.tree import export_graphviz
    import pydotplus
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True, feature_names = feature_cols,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('diabetes.png')
    Image(graph.create_png())

![](https://cdn-images-1.medium.com/max/2776/1*e8vzu4R6PrDiRnCC5_XeKA.png)

## 5. Data Pre-processing

### Pre-processing

Change Name of the column

    df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg_type', 'max_heart_rate_achieved',
           'exercise_induced_angina', 'st_depression', 'st_slope_type', 'num_major_vessels', 'thalassemia_type', 'target']

    df.columns

Index([‘age’, ‘sex’, ‘chest_pain_type’, ‘resting_blood_pressure’, ‘cholesterol’, ‘fasting_blood_sugar’, ‘rest_ecg_type’, ‘max_heart_rate_achieved’, ‘exercise_induced_angina’, ‘st_depression’, ‘st_slope_type’, ‘num_major_vessels’, ‘thalassemia_type’, ‘target’], dtype=’object’)

We have 4 Categorical columns as seen in Data Description using pandas profiling:

cp — chest_pain_type

restecg — rest_ecg_type

slope — st_slope_type

thal — thalassemia_type

Generating categorical columns values

    #cp - chest_pain_type
    df.loc[df['chest_pain_type'] == 0, 'chest_pain_type'] = 'asymptomatic'
    df.loc[df['chest_pain_type'] == 1, 'chest_pain_type'] = 'atypical angina'
    df.loc[df['chest_pain_type'] == 2, 'chest_pain_type'] = 'non-anginal pain'
    df.loc[df['chest_pain_type'] == 3, 'chest_pain_type'] = 'typical angina'

    #restecg - rest_ecg_type
    df.loc[df['rest_ecg_type'] == 0, 'rest_ecg_type'] = 'left ventricular hypertrophy'
    df.loc[df['rest_ecg_type'] == 1, 'rest_ecg_type'] = 'normal'
    df.loc[df['rest_ecg_type'] == 2, 'rest_ecg_type'] = 'ST-T wave abnormality'

    #slope - st_slope_type
    df.loc[df['st_slope_type'] == 0, 'st_slope_type'] = 'downsloping'
    df.loc[df['st_slope_type'] == 1, 'st_slope_type'] = 'flat'
    df.loc[df['st_slope_type'] == 2, 'st_slope_type'] = 'upsloping'

    #thal - thalassemia_type
    df.loc[df['thalassemia_type'] == 0, 'thalassemia_type'] = 'nothing'
    df.loc[df['thalassemia_type'] == 1, 'thalassemia_type'] = 'fixed defect'
    df.loc[df['thalassemia_type'] == 2, 'thalassemia_type'] = 'normal'
    df.loc[df['thalassemia_type'] == 3, 'thalassemia_type'] = 'reversable defect'

### One Hot Encoding

    data = pd.get_dummies(df, drop_first=False)
    data.columns

Index([‘age’, ‘sex’, ‘resting_blood_pressure’, ‘cholesterol’, ‘fasting_blood_sugar’, ‘max_heart_rate_achieved’, ‘exercise_induced_angina’, ‘st_depression’, ‘num_major_vessels’, ‘target’, ‘chest_pain_type_asymptomatic’, ‘chest_pain_type_atypical angina’, ‘chest_pain_type_non-anginal pain’, ‘chest_pain_type_typical angina’, ‘rest_ecg_type_ST-T wave abnormality’, ‘rest_ecg_type_left ventricular hypertrophy’, ‘rest_ecg_type_normal’, ‘st_slope_type_downsloping’, ‘st_slope_type_flat’, ‘st_slope_type_upsloping’, ‘thalassemia_type_fixed defect’, ‘thalassemia_type_normal’, ‘thalassemia_type_nothing’, ‘thalassemia_type_reversable defect’], dtype=’object’)

    df_temp = data['thalassemia_type_fixed defect']
    data = pd.get_dummies(df, drop_first=True)
    data.head()

![](https://cdn-images-1.medium.com/max/3250/1*qiFPnmrwrnzSsnxz3FYKFA.png)

![](https://cdn-images-1.medium.com/max/3582/1*mkRdyJkuzpTcYZRLtxT1kw.png)

Since one hot encoding dropped “thalassemia_type_fixed defect” column which was a useful column compared to ‘thalassemia_type_nothing’ which is a null column, we dropped ‘thalassemia_type_nothing’ and concatinated ‘thalassemia_type_fixed defect’

    frames = [data, df_temp]
    result = pd.concat(frames,axis=1)
    result.drop('thalassemia_type_nothing',axis=1,inplace=True)
    resultc = result.copy()# making a copy for further analysis in conclusion section

## 6. Logistic Regression

### 1. Gather columns

    X = result.drop('target', axis = 1)
    y = result['target']

### 2. Splitting Data

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

### 3. Normalization

Min-Max Normalization method is used to Normalize the data. This method scales the data range to [0,1]. Standardization is also used on a feature-wise basis in most cases.

![](https://cdn-images-1.medium.com/max/3540/0*k_w3tvviIMw62syi.png)

    X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)).values

    X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values

### 4. Fitting into Model

    from sklearn.linear_model import LogisticRegression
    logre = LogisticRegression()
    logre.fit(X_train,y_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class=’auto’, n_jobs=None, penalty=’l2', random_state=None, solver=’lbfgs’, tol=0.0001, verbose=0, warm_start=False)

### 5. Prediction

    y_pred = logre.predict(X_test)
    actual = []
    predcition = []

    for i,j in zip(y_test,y_pred):
      actual.append(i)
      predcition.append(j)

    dic = {'Actual':actual,
           'Prediction':predcition
           }

    result  = pd.DataFrame(dic)

    import plotly.graph_objects as go
     
    fig = go.Figure()
     
     
    fig.add_trace(go.Scatter(x=np.arange(0,len(y_test)), y=y_test,
                        mode='markers+lines',
                        name='Test'))
    fig.add_trace(go.Scatter(x=np.arange(0,len(y_test)), y=y_pred,
                        mode='markers',
                        name='Pred'))

![](https://cdn-images-1.medium.com/max/3534/1*JGQoM7lbZ9YjNjCD4Wrl4w.png)

The red dots represent the predicted values that is either 0 or 1 and the blue line & and dot represents the actual value of that particular patient. In the places where the red dot and blue dot do not overlap are the wrong predictions and where the both dots overlap those are the right predicted values.

### 6. Model Evaluation

    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test,y_pred))

0.8688524590163934

    from sklearn.metrics import classification_report
    print(classification_report(y_test,y_pred))

![](https://cdn-images-1.medium.com/max/2000/1*-9lbMeLmK3yGrKZw_OF5fw.png)

The classification report of the model shows that 91% prediction of absence of heart disease was predicted correct and 83% of presence of heart disease was predicted correct.

    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test,y_pred))
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)

![](https://cdn-images-1.medium.com/max/2000/1*gg4ckjv0q5ZotniFrDQXfw.png)

The confusion Matrix

![](https://cdn-images-1.medium.com/max/2000/1*gdIoF8dsWv3dbKSeHLZy_A.png)

The Confusion Matrix True Positive value is 24 and true Negative was 29. And the False Positive came out to be 3 and False Negative is 5.

**ROC Curve**

ROC Curves summarizes the trade-off between the true positive rate and the false positive rate for the predictive model using different probability thresholds.

The accuracy of the ROC curve came out to be 87.09%.

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr,tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for Heart disease classifier')
    plt.xlabel('False positive rate (1-Specificity)')
    plt.ylabel('True positive rate (Sensitivity)')
    plt.grid(True)

![](https://cdn-images-1.medium.com/max/2000/1*xh0P0q4Xu1mNROBkBzNZcA.png)

    import sklearn
    sklearn.metrics.roc_auc_score(y_test,y_pred)

0.8709150326797386

## 7. Conclusion

### 1. Coefficients

    print(logre.intercept_)
    plt.figure(figsize=(10,12))
    coeffecients = pd.DataFrame(logre.coef_.ravel(),X.columns)
    coeffecients.columns = ['Coeffecient']
    coeffecients.sort_values(by=['Coeffecient'],inplace=True,ascending=False)
    coeffecientsts

![](https://cdn-images-1.medium.com/max/2000/1*FuFP6VdibESr58DLeTF1jw.png)

### 2. Analysis

Preparing data for analysis

    df4 = df[df['target'] == 0 ][['age', 'sex', 'chest_pain_type', 'resting_blood_pressure','cholesterol', 'fasting_blood_sugar', 'rest_ecg_type', 'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression','st_slope_type', 'num_major_vessels', 'thalassemia_type', 'target']] #target 0 - people with heart disease

Heart Diseased Patient’s visualization

    plt.figure(figsize=(16,6))
    sns.distplot(df4['max_heart_rate_achieved'])

![](https://cdn-images-1.medium.com/max/2176/1*voC0Ng0bQ8saP5AzXkbVyw.png)

Normal Heart rate is found to be between 60 and 100 bpm. Some areas of cardiac muscles will start to die during a Heart Attack because of Lack of Blood. A person’s pulse may become slower (bradycardia) or faster (tachycardiac) depending on the type of Heart Attack they are experiencing.

    plt.figure(figsize=(20,6))
    sns.boxenplot(data=df4,x='rest_ecg_type',y='cholesterol',hue='st_slope_type')

![](https://cdn-images-1.medium.com/max/2816/1*99ONer3ZGeAFC7p_Yl4rDQ.png)

In normal type of rest ECG proves to be important for the prediction model along with the down sloping ST slope. The patient composed of these two features usually has cholesterol level between 170 to 225 mg/dl. The other types of slops with the rest ECG seems to be more spread out and less concise.

    plt.figure(figsize=(20,6))
    sns.boxenplot(data=df4,x='chest_pain_type',y='max_heart_rate_achieved',hue='thalassemia_type')

![](https://cdn-images-1.medium.com/max/2686/1*X-mKq6DazvhlQl1gaTkFtQ.png)

### Shap

Shap Values

    !pip install shap 
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test,check_additivity=False)

    shap.summary_plot(shap_values[1], X_test, plot_type="bar")

![](https://cdn-images-1.medium.com/max/2000/1*xe_p8TFLuOxnuQ95PMuV1w.png)

Shap value for Model Explaination

    shap.summary_plot(shap_values[1], X_test)

![](https://cdn-images-1.medium.com/max/2000/1*VklK3V9lSXkLBeybfe23rA.png)

    def patient_analysis(model, patient):
      explainer = shap.TreeExplainer(model)
      shap_values = explainer.shap_values(patient)
      shap.initjs()
      return shap.force_plot(explainer.expected_value[1], shap_values[1], patient)

Reports for two Patients

    patients = X_test.iloc[3,:].astype(float)
    patients_target = y_test.iloc[3:4]
    print('Target : ',int(patients_target))
    patient_analysis(model, patients)

Target : 0

![](https://cdn-images-1.medium.com/max/3630/1*V7DfU0tMbRYDH5yPIOEjCA.png)

    patients = X_test.iloc[33,:].astype(float)
    patients_target = y_test.iloc[33:34]
    print('Target : ',int(patients_target))
    patient_analysis(model, patients)

Target : 1

![](https://cdn-images-1.medium.com/max/3632/1*RbTJNy40MWLSgVjjmrOkxA.png)

    # dependence plot

    shap.dependence_plot('num_major_vessels', shap_values[1], X_test, interaction_index = "st_depression")

![](https://cdn-images-1.medium.com/max/2000/1*93b8oTL6cSWn9SNJtxReKA.png)

    shap_values = explainer.shap_values(X_train.iloc[:50],check_additivity=False)

    shap.initjs()

    shap.force_plot(explainer.expected_value[1], shap_values[1], X_test.iloc[:50])

![](https://cdn-images-1.medium.com/max/3592/1*CGYrFNEdkHF0hMkTZJBUAg.png)

### 3. Conclusion

* The Area under the ROC curve is 87.09% which is somewhat satisfactory.

* The model predicted with 86.88% accuracy. The model is more specific than sensitive.

* According to this model the major features contributing in precision of predicting model are shown in the heatmap in Ascending order.

    plt.figure(figsize=(10,12))

    coeffecients = pd.DataFrame(logre.coef_.ravel(),X.columns)

    coeffecients.columns = ['Coeffecient']

    coeffecients.sort_values(by=['Coeffecient'],inplace=True,ascending=False)

    sns.heatmap(coeffecients,annot=True,fmt='.2f',cmap='Set2',linewidths=0.5)

![](https://cdn-images-1.medium.com/max/2000/1*SKELWrek4CV6jqDu91rr6w.png)

The important features contributing to the accuracy of the prediction are shown through the Heatmap in descending order. In silver color code, the most contributing feature, the chest pain types and maximum heart rate achieved proved to be more valuable by 1.28 to 1.03 units.

## FIN.

