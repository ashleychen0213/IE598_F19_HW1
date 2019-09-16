import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns
df = pd.read_csv('/Users/ashleychen/Desktop/UIUC/IE 598/HW3/HY_Universe_26k_basicliquidityscoring_INETF.csv', 
                 header = 0)

#Listing 2-1: Sizing up a new dataset
shape = df.shape
print('Shape = {}\n'.format(shape)) 

#Listing 2-2: Determining the Nature of Attributes
df.dtypes

#Listing 2-3, Statistical Summery, 2-5: Using Pandas to Summarize Data
data = df.to_numpy()

#CUSIP 
cusip_unique = np.unique(data[:,0])
print('There are ' + str(len(cusip_unique)) + ' bonds.')

#Ticker
print(df['Ticker'].describe())

#Issue Date
print(df['Issue Date'].describe())

#Maturity
print(df['Maturity'].describe())

#1st Call Date
print(df['1st Call Date'].describe())

#Moodys
print(df['Moodys'].value_counts())

#S_and_P
print(df['S_and_P'].value_counts())

#Fitch
print(df['Fitch'].value_counts())

#Bloomberg Composite Rating
print(df['Bloomberg Composite Rating'].value_counts())

#Coupon
print(df['Coupon'].describe())

#Issued Amount 
print(df['Issued Amount'].describe())

#Maturity Type 
print (df['Maturity Type'].value_counts())

#Coupon Type
print(df['Coupon Type'].value_counts())

#Industry
print(df['Industry'].value_counts())

#Months in JNK
print(df['Months in JNK'].describe())

#Months in HYG
print(df['Months in HYG'].describe())

#Months in Both
print(df['Months in Both'].describe())

#LIQ SCORE
print(df['LIQ SCORE'].describe())

#Listing 2-7: Cross Plotting Pairs of Attributes
#This scatter plot is used to show the relationship between Months in JNK and Months in HYG,
#The plot tends to be symmetric
plt.scatter(data[:,14],data[:,15], marker = '.')
plt.xlabel('Months in JNK')
plt.ylabel('Months in HYG')
plt.show()

#This scatter plot is used to show the relationship between Maturity Type and Coupon
plt.scatter(data[:,11],data[:,9], marker = '.')
plt.xlabel('Maturity Type')
plt.ylabel('Coupon')
plt.show()

#Listing 2-8: Target-attribute cross-plot
#This scatter plot is used to show the relation between an attribute (Maturity Type)
#and the output (LIQ SCORE), and we can see that callable and at maturity bond 
#has wider range of LIQ SCORE
plt.scatter(data[:,11], data[:,-1])
plt.xlabel('Maturity Type')
plt.ylabel('LIQ SCORE')
plt.show()

#Listing 2-10: Presenting Attribute Correlations Visually
# The heat map shows attribute cross-correlation
# and we can see that Months in HYG and the output are relatively highly correlated
corMat = DataFrame(df.corr())
print(corMat)
plt.pcolor(corMat)
plt.show()

#This is to give a better look of the relation between Months in HYG and output
#We can see that Months in HYG and LIQ SCORE has a corelation less than 1 but greater than 0.5
plt.scatter(data[:,-3], data[:,-1])
plt.ylim(0,70)
plt.xlabel('Months in HYG')
plt.ylabel('LIQ SCORE')
plt.show()

#Boxplot showing the relationship between Bloomberg Composite Rating and the output
plt.figure(figsize=(16, 6))
sns.boxplot(x=data[:,8], y= data[:,17],width=0.5)
plt.xlabel('Maturity Type')
plt.ylabel('LIQ SCORE')
plt.show()

#Boxplot showing the relationship between Maturity Type and the output
#Compare to bonds at maturity, callable bonds has lower mean LIQ SCORE.
sns.boxplot(x = data[:,11], y = data[:,-1])
plt.xlabel('Maturity Type')
plt.ylabel('LIQ SCORE')
plt.show()

print("My name is Yu Chi Chen")
print("My NetID is: yuchicc2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
