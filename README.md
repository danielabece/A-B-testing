# Introduction

In this project, we’ll explore whether there is a significant difference in conversion (number of purchases/number of visits) between two groups (Control and Treatment) using A/B testing. Using Python and statistical testing, we’ll compare conversion rates between both groups to determine if a new version of a landing page performs significantly better than the older one

The data comes from [Kaggle](https://www.kaggle.com/datasets/zhangluyuan/ab-testing])

# Background

This project is part of my way to improve my Python skills, specially to understand the manage of A/B tests on Python, which are esential for making decisions in various fields, particularly in digital marketing and product development.

## Goal of the Project

Discover if the new page helps us grow the conversion rate of 12% that the landing page has nowdays. In other words, **we want to prove if the group Treatment which it was presented the new version made more purchases or visited more the landpage than the other group Control which it was presented the old landpage.**

# Code Steps

## Import

```py
# Import the libraries.

import pandas as pd
import zipfile
import kaggle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.stats.api as sms
import scipy.stats as stats

# Download the dataset "ab-testing" from Kaggle, it will download a zip file.

!kaggle datasets download -d zhangluyuan/ab-testing

# Extract the content from the zip file.

zipfile_name = 'ab-testing.zip'
with zipfile.ZipFile(zipfile_name, 'r') as file:
    file.extractall()

# Read in the csv as a pandas dataframe.

df = pd.read_csv('ab_data.csv')
```
## Explore data

```py
# The dataframe contains the following columns: user_id, timestamp, group, landing_page, converted. This ones are explained in more detail below.

df.sample(10)

# Explore shape of the dataframe: 294,478 rows and 5 columns.

df.shape

# The key insight of this code is that the number of rows (294,478) is greater than the number of unique users (290,584). That means some users appear more than once.
# In an A/B test, each user should only be in one group, so duplicate users can bias the results.

df.nunique()

# There's no null values.

df.isna().sum()

# The content of the 5 columns are the following:
# 'user_id' contains user ids
# 'timestamp' is about when a session was
# 'group' contains 2 variables: control and treatment
# 'landing_page' is about what version of a site a user saw
# 'converted' says us about user's behavior: if a user made a purchase (1) or not (0)

df.info()
```
## Align groups to their landing pages.

```py
# The following code shows some mismatches; control group should only the old page and treatment group should only see the new page.

pd.crosstab(df['group'], df['landing_page'])

# Align groups to their specific landing pages.

df_clean = df[
    ((df['group'] == 'control') & (df['landing_page'] == 'old_page')) |
    ((df['group'] == 'treatment') & (df['landing_page'] == 'new_page'))
]

# Validate the groups are correctly align.

pd.crosstab(df_clean['group'], df_clean['landing_page'])
```

## Drop off duplicates.

```py
# Align groups helped with the duplicates, despite there's just one another duplicate.

df_clean[['user_id','timestamp']].nunique()

# The one that has more that one occurance it is the duplicated one.

df_clean.user_id.value_counts()

# Get the index of the duplicate. 

session_counts = df_clean.user_id.value_counts()
double_users = session_counts[session_counts>1].index

double_users

# Remove the duplicate.

df_clean = df_clean[~df_clean['user_id'].isin(double_users)]
df_clean.shape

## Balance the number of samples.

```py
# Validate the number of samples for each group.

df_clean.group.value_counts()

# Balance number of samples of each group.

treatment_group = df_clean[df_clean['group'] == 'treatment'] # Filter treatment group
treatment_trimmed = treatment_group.iloc[35:]    # Remove the first 35 values of treatment group
control_group = df_clean[df_clean['group'] == 'control']     # Filter control group
ab_test = pd.concat([control_group, treatment_trimmed], axis=0)
ab_test.reset_index(inplace=True, drop=True)

# Validate the number of samples for each group.

ab_test.group.value_counts()
```

## Conversion rates.

```py
# The control group shows 12.0% conversion rate and the treatment group 11.9% conversion rate.

conversion_rates = ab_test.groupby('group')['converted']

std_p = lambda x: np.std(x, ddof=1)              # Std. deviation of the proportion
se_p = lambda x: stats.sem(x, ddof=1)            # Std. error of the proportion (std / sqrt(n))

conversion_rates = conversion_rates.agg([np.mean, std_p, se_p])
conversion_rates.columns = ['conversion_rate', 'std_deviation', 'std_error']


conversion_rates.style.format('{:.3f}')
```
## Z-statistic and P value.

```py
# Count the number of occurrences of each group.

total_counts = ab_test.groupby('group')['converted'].count()
print(total_counts)

# Count the number of users converted (1) with method sum.

conversion_counts = ab_test.groupby('group')['converted'].sum()
print(conversion_counts)

# Interpretation:
# The p-value of 0.1866 is greater than the significance level of 0.05,
# so we fail to reject the null hypothesis. This suggests that the
# observed difference in conversion rates between groups A and B is not statistically significant.

z_stat, p_val = proportions_ztest(count=conversion_counts.values, nobs=total_counts.values)
print(f"Z-statistic: {z_stat:.3f}")
print(f"P-value: {p_val:.4f}")
```
## Graph total users vs converted users

```py
# Count users per group.

group_counts = ab_test['group'].value_counts().sort_index()
conversion_counts = ab_test[ab_test['converted'] == 1]['group'].value_counts().sort_index()

# Creat positions in X axis per bar.

x = np.arange(len(group_counts))  # [0, 1] for A and B groups
width = 0.35  # Width for each bar

# Graph bars

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, group_counts.values, width, label='Total users', color='#7FB3D5')     
bars2 = ax.bar(x + width/2, conversion_counts.values, width, label='Converted users', color='#2E86C1')  

# Labels and ticks

ax.set_xticks(x)
ax.set_xticklabels(group_counts.index)
ax.set_xlabel('Group')
ax.set_ylabel('Number of users')
ax.set_title('Total users vs. converted users by group')
ax.legend()

plt.show()
```

# Conclusions

The Z-statistic and p-value indicate that **the new page does not show a significant difference in conversion compared to the old page.** Therefore, it might be helpful to explore a different page design and run another A/B test to evaluate its effectiveness.





