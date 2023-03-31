#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for plotting
import matplotlib.pyplot as plt # for plotting
import datetime


import os


# In[3]:


data = pd.read_csv('Total Emissions Per Country (2000-2020).csv')
data.head(224)


# In[4]:


data.columns


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


def set_frame_style(df, caption=""):
    """Helper function to set dataframe presentation style.
    """
    return df.style.background_gradient(cmap='Greens').set_caption(caption).set_table_styles([{
    'selector': 'caption',
    'props': [
        ('color', 'Green'),
        ('font-size', '18px'),
        ('font-weight','bold')
    ]}])

def bar_annot(ax, horizontal=True, annot_format='{:.2f}', fs=10):
    """Function to annotate bar plots. ax is the barplot axis object. annot_format is the annotation numeric display format.
    """
    if horizontal:
        for p in ax.patches:
            annot = annot_format.format(p.get_height())
            ax.annotate(annot, (p.get_x(), p.get_height()), fontsize=fs)
    else:
        for p in ax.patches:
            annot = annot_format.format(p.get_width())
            ax.annotate(annot, (p.get_width(), p.get_y()*1.01), fontsize=fs)


# In[8]:


set_frame_style(data.head(), "First Few Rows Of Data")


# In[9]:


set_frame_style(data.describe(),"Brief Overview Of Numerical Data")


# In[10]:


df_unique = pd.DataFrame({'Unique Value Counts':data[['Area', 'Item', 'Element', 'Unit']].nunique()})
set_frame_style(df_unique, "Unique Value Counts Per Categorical Column")


# In[11]:


years = [str(x) for x in range(2000,2021)]
df2 = data.melt(id_vars = ['Area','Item','Element'],value_vars=years, var_name='Year', value_name='Emissions')
df2.dropna(inplace=True)
set_frame_style(df2.head(10), "First Few Rows Of Data")
# Drop the redundant units column


# In[12]:


# Aggregate the data
df_agg = df2[['Item','Emissions']].groupby(by=['Item']).sum().reset_index().sort_values(by='Emissions', ascending=False)

plt.figure(figsize=(9,6))

ax = sns.barplot(data=df_agg, x='Emissions', y='Item', palette=sns.color_palette("tab20c"))
ax.xaxis.grid()
ax.set_ylabel('Source', weight='bold')
ax.set_xlabel('Emissions (kilotonnes)', weight='bold')
ax.set_title('Total Emissions By Source', weight='bold')

bar_annot(ax, False,'{0:.3g}',6) # Apply annotations

plt.yticks(fontsize=6)
plt.tight_layout()
plt.show()


# In[13]:


years = [str(x) for x in range(2000,2021)]
df3 = data.melt(id_vars = ['Area','Item','Element'],value_vars=years, var_name='Year', value_name='Emissions')
df3.dropna(inplace=True)
set_frame_style(df2.head(10), "First Few Rows Of Data")
# Drop the redundant units column


# In[14]:


# Aggregate the data
df_agg = df3[['Area','Emissions']].groupby(by=['Area']).sum().reset_index().sort_values(by='Emissions', ascending=False)

plt.figure(figsize=(10,9))

ax = sns.barplot(data=df_agg, x='Emissions', y='Area', palette=sns.color_palette("tab20c"))
ax.xaxis.grid()
ax.set_ylabel('Source', weight='bold')
ax.set_xlabel('Emissions (kilotonnes)', weight='bold')
ax.set_title('Total Emissions By Source', weight='bold')

bar_annot(ax, False,'{0:.3g}',6) # Apply annotations

plt.yticks(fontsize=6)
plt.tight_layout()
plt.show()
#hypothesis largest with densest population country will have the most emissions


# In[15]:


= [str(x) for x in range(2000,2021)]
df4 = data.melt(id_vars = ['Area','Item','Element'],value_vars=years, var_name='Year', value_name='Emissions')
df4.dropna(inplace=True)
set_frame_style(df2.head(10), "First Few Rows Of Data")
# Drop the redundant units column


# In[16]:


# Aggregate the data
df_agg = df3[['Element','Emissions']].groupby(by=['Element']).sum().reset_index().sort_values(by='Emissions', ascending=False)

plt.figure(figsize=(6,6))

ax = sns.barplot(data=df_agg, x='Emissions', y='Element', palette=sns.color_palette("tab20c"))
ax.xaxis.grid()
ax.set_ylabel('Element', weight='bold')
ax.set_xlabel('Emissions (kilotonnes)', weight='bold')
ax.set_title('Total Emissions By Source', weight='bold')

bar_annot(ax, False,'{0:.3g}',6) # Apply annotations

plt.yticks(fontsize=6)
plt.tight_layout()
plt.show()


#hypothesis most emissions will be from CO2eq.


# In[17]:


# Aggregate the data
df_agg = df2[['Year','Emissions']].groupby(by=['Year']).sum().reset_index()

plt.figure(figsize=(8,4))

ax = sns.lineplot(data=df_agg, x='Year', y='Emissions', color='red', marker='o')
ax.yaxis.grid()
ax.xaxis.grid()
ax.set_ylabel('Emissions (kilotonnes)', weight='bold')
ax.set_xlabel('Year', weight='bold')
ax.set_title('Total Emissions Over The Years', weight='bold')

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[18]:


df_agg = df2[['Year','Emissions']].groupby(by=['Year']).sum().reset_index()

plt.title("Clara Homework 1")

Year= [120,80]
labels= ['Sixty','Forty']


plt.pie(Year, labels=labels)


#hypothsis i cant get pie charts to work


# In[21]:


years = [str(x) for x in range(2000,2021)]
data2 = data.melt(id_vars = ['Area','Item','Element'],value_vars=years, var_name='Year', value_name='Emissions')
df_upd = data2[['Area','Emissions']].groupby(by=['Area']).sum().reset_index().sort_values(by='Emissions', ascending=False)
df_upd.head(25)
G14_countries = ['Australia', 'Canada', 'China', 'France', 'Germany', 'India', 'Indonesia', 'Italy',
                'Japan', 'Republic of Korea', 'Mexico', 'Russian Federation', 'United Kingdom of Great Britain and Northern Ireland',
                'United States of America']
G14_emissions = df_upd[df_upd['Area'].isin(G14_countries)]
G14_total_emissions = G14_emissions.sum(numeric_only = True)
World = df_upd.loc[275]['Emissions']
World_minus_G14 = World - G14_total_emissions
print(World)
print (G14_total_emissions)
print(World_minus_G14)
dict = {'Area': 'World', 'Emissions': World_minus_G14}
df2 = pd.DataFrame(dict)
G14_emissions_2 = pd.concat([G14_emissions, df2], ignore_index = True)
G14_emissions_2.reset_index()


# In[32]:


G14_emissions_indexed = G14_emissions_2.set_index('Area')
a = 0
b = 0.1
myexplode = [a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,b]
plot = G14_emissions_indexed.plot.pie(y='Emissions', 
                                      figsize=(20, 20),
                                      autopct='%1.1f%%', 
                                      colors=["red", "lightblue", "lightcoral", "lime","darkorange",
                                              "paleturquoise","plum","dodgerblue","yellowgreen","brown",
                                             "cyan","orange","green","olive"],
                                              textprops={'fontsize': 12},
                                     wedgeprops = {"edgecolor" : "black",
                                                    'linewidth': 2,
                                                    'antialiased': True},
                                     pctdistance=0.95, labeldistance=1, rotatelabels=True,
                                     explode = myexplode, title='Emissions of prominent countries')


# In[ ]:





# In[ ]:




