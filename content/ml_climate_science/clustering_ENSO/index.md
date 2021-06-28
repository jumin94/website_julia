---
title: "El Niño Southern Oscillation flavors- Logistic Regression Example"
date: 2020-05-17T23:53:00+01:00
draft: false
hideLastModified: true
summary: "Here I show an example \
of using logistic regression\
to cluster ENSO flavors"
summaryImage: "ENSOclustering.jpg"
tags: ["logistic regression","research"]
---

The interannual variability of El Niño Southern Oscillation has huge impacts on South America, as many other regions of the world. To study changes in its impacts, changes in the frequency of the phases and changes in the amplitude of El Niño and La Niña events we can analyse trends, paleoclimatic record or climate simulations of different emissions scenarios performed with global climate models. In particular, different flavors of ENSO events have different impacts in South America. To sistematically study this, we need to find these events in our data. 

This example shows how based on three indices of temperature anomalies in the pacific these events can be distinguished. 

The method that we apply here is Logistic Regression, and it is a very simple analysis once the theory is understood, because it is implemented in python's Sci-kit learn library.

We apply this method to [Kaplan SST V2 data](https://psl.noaa.gov/data/gridded/data.kaplan_sst.html) provided by the NOAA/OAR/ESPL PSL, Boulder, Colorado, USA and the processed dataset can be downloaded [here](https://drive.google.com/file/d/1Jff_V9afOiTS3YdbvIndZqqTjdO8SzI0/view?usp=sharing). 

To train the logistic regression model, a dataset was generated where three indices were calculated for SST anomalies from all possible three-month seasons:


Additionally, the indices C-index: and E-index: were computed. 

With these indices, the following ENSO flavours were distinguished: 

Neutral ENSO

Central Niño ( Niño 3.4 > 0.5 , C-index > Eindex)

Eastern Niño ( Niño 3.4 > 0.5 , C-index < Eindex) 

Central Niña ( Niño 3.4 < -0.5 , C-index < Eindex)

Eastern Niña (Niño 3.4 < -0.5 , C-index > Eindex)


Code:

```python
import numpy as np 
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

NINO = pd.read_csv('../sst_kaplan_no_map.csv',  error_bad_lines=False)

```
We categorize the data:

```python

cat_list=pd.DataFrame(NINO.nino_type.value_counts())
cat_list=cat_list.index

NINO['cat']=np.zeros((len(NINO),1))+np.nan
NINO.cat[NINO['nino_type'].str.contains('Central Nino')]=1
NINO.cat[NINO['nino_type'].str.contains('Central Nina')]=-1
NINO.cat[NINO['nino_type'].str.contains('Eastern Nino')]=2
NINO.cat[NINO['nino_type'].str.contains('Eastern Nina')]=-2
NINO.cat[NINO['nino_type'].str.contains('Neutral')]=0
```

And finally train the classifier, separating the data in training and testing set and using C-index and E-index as features. 

```python

X = NINO[{"C_index", 'E_index'}] 
y = NINO["cat"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
softmax_reg.fit(X_train, y_train)


x0, x1 = np.meshgrid(
        np.linspace(-3, 4, 500).reshape(-1, 1),
        np.linspace(-3, 4, 200).reshape(-1, 1),
    )

X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
y_predict_test = softmax_reg.predict(X_test)

plt.plot(X_test.values[y_predict_test==1, 0], X_test.values[y_predict_test==1, 1], "k^", label="Central Niño")
plt.plot(X_test.values[y_predict_test==2, 0], X_test.values[y_predict_test==2, 1], "rs", label="Eastern Niño")
plt.plot(X_test.values[y_predict_test==-1, 0], X_test.values[y_predict_test==-1, 1], "bP", label="Central Niña")
plt.plot(X_test.values[y_predict_test==-2, 0], X_test.values[y_predict_test==-2, 1], "yD", label="Eastern Niña")
plt.plot(X_test.values[y_predict_test==0, 0], X_test.values[y_predict_test==0, 1], "go", label="Neutral")


from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0','orange','grey','pink'])
plt.contourf(x0, x1, zz, cmap=custom_cmap)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("E_index", fontsize=14)
plt.ylabel("C_index", fontsize=14)
plt.legend(bbox_to_anchor=(1.2,0.7), fontsize=14)
```

Results

# References
Géron, A. Hands-on Flow for Machine Learning with Scikit-Learn, Keras TensorFlow. Published by O’Reilly Media, Inc., 2019.


