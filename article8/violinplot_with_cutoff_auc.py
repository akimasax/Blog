import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#Create training dataset
#Create training data
import random
random.seed(1234)
train_low_data  = pd.DataFrame({'Output':[random.uniform(0,0.6) for li in range(35)]})
train_high_data = pd.DataFrame({'Output':[random.uniform(0.5,1) for li in range(35)]})
train_data = pd.concat([train_low_data,train_high_data],axis=0)
#Create Label
train_low_label = pd.DataFrame({'Dataset':['Train']*35,'Grade':['Low']*35,'Label':[-1]*35})
train_high_label = pd.DataFrame({'Dataset':['Train']*35,'Grade':['High']*35,'Label':[1]*35})
train_label = pd.concat([train_low_label,train_high_label],axis=0)
#Combine data and label
train = pd.concat([train_data,train_label],axis=1)

#Calculate training auc, fpr, tpr, and thresholds
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(train['Label'],train['Output'])
train_auc = roc_auc_score(train['Label'],train["Output"])

#Decide optimal threshold (cutoff) value depend on Youden index
Youden_index_candidates = tpr-fpr
index = np.where(Youden_index_candidates==max(Youden_index_candidates))[0][0]
cutoff = thresholds[index]

#Create test dataset
#Create test data
random.seed(12345)
test_low_data   = pd.DataFrame({'Output':[random.uniform(0,0.6) for li in range(15)]})
test_high_data  = pd.DataFrame({'Output':[random.uniform(0.5,1) for li in range(15)]})
test_data = pd.concat([test_low_data,test_high_data],axis=0)
#Create Label
test_low_label = pd.DataFrame({'Dataset':['Test']*15,'Grade':['Low']*15,'Label':[-1]*15})
test_high_label = pd.DataFrame({'Dataset':['Test']*15,'Grade':['High']*15,'Label':[1]*15})
test_label = pd.concat([test_low_label,test_high_label],axis=0)
#Combine data and label
test = pd.concat([test_data,test_label],axis=1)
#Calculate test auc
test_auc = roc_auc_score(test['Label'],test["Output"])

#Combine training and test dataset for drawing violin plot
all_data = pd.concat([train,test],axis=0,join='inner')
all_data = all_data.reset_index(drop=True)

#Draw vioplin plot
#Figure captions
plt.figure(figsize=(10,10))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["figure.dpi"] = 300  
plt.rcParams["xtick.direction"] = "in" 
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["legend.edgecolor"] = "black" 
plt.rcParams["xtick.minor.size"] = 2
plt.rcParams["ytick.minor.size"] = 2
plt.rcParams["xtick.labelsize"] = 24
plt.rcParams["ytick.labelsize"] = 24
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

#Violin plot
v1 = sns.violinplot(x="Dataset",y="Output",inner='box',hue='Grade',palette='muted',data = all_data)
#Output plot
sns.swarmplot(x="Dataset",y="Output",hue="Grade",edgecolor="black",size=10,palette='muted',data=all_data,ax=v1)
#Cutoff line
plt.axhline(y=cutoff,color="black",label="Cutoff",linestyle="dashed",linewidth=2)
#Add new legend
labels = ["Low grade","High grade","Cutoff:0.5"]
handles,_ = v1.get_legend_handles_labels()
#Change legend order
handles = [handles[1],handles[2],handles[0]]
#Plot legned
plt.legend(handles = handles[0:],labels=labels,fontsize=20,loc=(0.02,0.82),frameon=False)
#ticks
v1.tick_params(direction='in',length=6,width=2,color='k')
plt.ylabel("Machine learning output",fontweight='bold',fontsize=28)
plt.xlabel("Dataset",fontweight='bold',fontsize=28)
#text of training and test auc
plt.text(-0.2,-0.2,"AUC:"+str(f'{train_auc:.2f}'), size=24)
plt.text(0.8,-0.2,"AUC:"+str(f'{test_auc:.2f}'),size=24)
#Save figure
plt.savefig("ViolonPlot_with_cutoff_auc.tif",format="tif",dpi=300)
