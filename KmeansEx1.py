
# coding: utf-8

# In[35]:

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# In[37]:

#Load data
df=pd.read_csv('data_1024.csv',sep='\t')
print(df.head())
print(df.dtypes)
print(df.describe())


# In[38]:

### For the purposes of this example, we store feature data from our
### dataframe `df`, in the `f1` and `f2` arrays. We combine this into
### a feature matrix `X` before entering it into the algorithm.
f1 = df['Distance_Feature'].values
f2 = df['Speeding_Feature'].values



# In[39]:

X=np.matrix(zip(f1,f2))


# In[41]:

#kmeans = KMeans(n_clusters=2).fit(X)


# In[ ]:



