import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu 
import matplotlib.pyplot as plt

df=pd.read_csv("metabolites.csv",sep=',', encoding='cp1252')
numeric_columns = df.iloc[:, 10:1018]
print(numeric_columns)

##Apply normalization
scaler = StandardScaler()
df1 = scaler.fit_transform(numeric_columns)
df1


# control and disease groups
control_group = df[df['Group'] == 'Control']
disease_group = df[df['Group'] == 'Cases']

differential_results = []

metabolite_columns = numeric_columns.columns[0:]  
metabolite_columns

control_group[metabolite_columns].isna().mean()
disease_group[metabolite_columns].isna().mean()

##calculate t test and p values
for metabolite_name in metabolite_columns:
    t_stat, p_value = ttest_ind(control_group[metabolite_name], disease_group[metabolite_name])
    
    differential_results.append({'Metabolite': metabolite_name, 'T-Stat': t_stat, 'P-Value': p_value})

differential_results_df = pd.DataFrame(differential_results)
differential_results_df

print(t_stat)
fold_change_results = []

for metabolite_name in metabolite_columns:
    mean_disease = disease_group[metabolite_name].mean()
    mean_control = control_group[metabolite_name].mean()

    if mean_control == 0:
        fold_change = float('inf')  # Set fold change to infinity
    else:
        fold_change = mean_disease / mean_control
  # Perform log transformation
    log_fold_change = np.log(fold_change)

    fold_change_results.append({'Metabolite': metabolite_name, 'Fold Change': log_fold_change})

# DataFrame
fold_change_df = pd.DataFrame(fold_change_results)
fold_change_df

differential_results_df['Adjusted P-Value'] = multipletests(differential_results_df['P-Value'], method='Bonferroni')[1]

differential_results_df.sort_values(by='Adjusted P-Value', ascending=True, inplace=True)

combined_results = pd.merge(differential_results_df, fold_change_df, on='Metabolite')
combined_results

#write to files
combined_results.to_csv('/Users/Ash/Desktop/metabolites/foldchange_withbonferronicorrection.csv')

# significant upregulated metabolites
significant_metabolites_up = combined_results[(combined_results['P-Value'] < 0.05) & (combined_results['Fold Change'] > 1.0)]
print(significant_metabolites_up)
# significant downregulated metabolites
significant_metabolites_up = combined_results[(combined_results['P-Value'] < 0.05) & (combined_results['Fold Change'] < 1.0)]
print(significant_metabolites_up)


# code for Mann-Whitney U test 
group1 = df.groupby('Group').get_group('Control')
group2 = df.groupby('Group').get_group('Cases')

group1 = group1.iloc[:, 10:1025]
group1.fillna(0, inplace=True)


group2 = group2.iloc[:, 10:1025]
group2.fillna(0, inplace=True)

col_1= group1.columns
results =pd.DataFrame(mannwhitneyu(group1, group2))
results

df_mann = results.set_axis(col_1, axis=1)
df_mann

row_1=("Mann_stat", 'p-value')
df_mann = df_mann. set_axis(row_1, axis=0)
df_mann
transpose_df1_mann= df_mann.T

sign_mann= transpose_df1_mann.loc[transpose_df1_mann['p-value'] <= 0.05]
sign_mann

##to plot the important features
sign_mann.sort_values(by='p-value', inplace=True)
feature_scores = 10
# bar chart 
plt.figure(figsize=(12, 8))
plt.subplots_adjust(left=0.25)
plt.figure(figsize=(12, 6), facecolor='white')
plt.grid(False)
plt.barh(sign_mann['Features'], sign_mann['p-value'], color='grey')
plt.ylabel('Important Features', fontsize=12)
plt.xlabel('Mann Whitney p value', fontsize=12)
plt.gca().set_facecolor('white')
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.rc('axes',edgecolor='black')
plt.gca().patch.set_edgecolor('black')
plt.xticks(rotation=45, ha='center')  # You can adjust the rotation and horizontal alignment as needed
plt.title('Top 10 Selected Features', fontsize=12)
plt.savefig('features.png', bbox_inches='tight', pad_inches=0, dpi=600)
plt.show()