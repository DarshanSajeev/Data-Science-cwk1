import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
# Cleaned data, age groups and helper functions all come from cwk1.py (importing it runs the cleaning)
from cwk1 import df, age_labels, weighted_pct, topline
sns.set_theme(style='whitegrid')

# Helper for annotated heatmap tables
def heat(data, title, fname, fmt='.0f', cmap='viridis', center=None, size=(6, 8), cbar_label=''):
    plt.figure(figsize=size)
    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap, center=center, cbar_kws={'label': cbar_label})
    plt.title(title); plt.xlabel(''); plt.ylabel('')
    plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()

# Fig 1: exact ages (min and max in the title) and the number of respondents in each age group
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(df['age'].dropna(), bins=range(18, 102, 4), color='#4C72B0', ax=ax[0])
ax[0].set_title(f'Exact age (min {df["age"].min():.0f}, max {df["age"].max():.0f}); Australia has bands only')
ax[0].set_xlabel('Age'); ax[0].set_ylabel('Respondents')
counts = df['age_grp'].cat.add_categories('Missing').fillna('Missing').value_counts().reindex(age_labels + ['Missing'])
bars = ax[1].bar(counts.index.astype(str), counts.values, color=['#4C72B0'] * 4 + ['#C44E52'])
ax[1].bar_label(bars); ax[1].set_title('Respondents per age group'); ax[1].set_ylabel('Respondents')
plt.savefig('fig1_age.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 2: share of respondents with no age group, by country (a high bar means a problem with the age handling there)
missing_age = df.groupby('country_name')['age_grp'].apply(lambda s: s.isna().mean() * 100).sort_values()
plt.figure(figsize=(6, 7))
bars = plt.barh(missing_age.index, missing_age.values, color='#C44E52')
plt.bar_label(bars, fmt='%.1f'); plt.xlabel('% of respondents with no age group'); plt.title('Missing age group by country')
plt.savefig('fig2_missing_age.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 3: education codes in Argentina (NaN = DK/refused after recoding), to check that 98/99 are the DK/refused codes
arg = df.loc[df['country_name'] == 'Argentina', 'd_educ_argentina'].value_counts(dropna=False).sort_index()
plt.figure(figsize=(7, 4))
bars = plt.bar([str(i) for i in arg.index], arg.values, color='#4C72B0')
plt.bar_label(bars); plt.xlabel('d_educ_argentina code (nan = DK/refused)'); plt.ylabel('Respondents'); plt.title('Education codes in Argentina')
plt.savefig('fig3_educ_argentina.png', dpi=150, bbox_inches='tight'); plt.close()

# Fig 4: reproducing the published topline. Each cell shows my figure (published figure in brackets), colour = my figure minus the published one in percentage points
check_countries = ['Canada', 'India', 'Japan']
published_q30 = pd.DataFrame({'A lot': [41, 14, 53], 'A little': [51, 32, 36], 'Nothing': [8, 35, 11], 'DK/Refused': [0, 19, 0]}, index=check_countries)
published_q31 = pd.DataFrame({'Excited': [9, 16, 16], 'Concerned': [45, 19, 28], 'Equally': [45, 39, 55], 'DK/Refused': [1, 26, 1]}, index=check_countries)
mine_q30 = topline('ai_heard_raw', [1, 2, 3], {1: 'A lot', 2: 'A little', 3: 'Nothing'}, check_countries)
mine_q31 = topline('ai_cncexc_raw', [1, 2, 3], {1: 'Excited', 2: 'Concerned', 3: 'Equally'}, check_countries)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.4))
for a, mine, pub, title in [(ax[0], mine_q30, published_q30, 'Q30: heard about AI'), (ax[1], mine_q31, published_q31, 'Q31: excited vs concerned')]:
    mine = mine.reindex(index=pub.index, columns=pub.columns).fillna(0)
    labels = (mine.round(0).astype(int).astype(str) + ' (' + pub.astype(str) + ')').values
    sns.heatmap(mine - pub, annot=labels, fmt='', cmap='coolwarm', center=0, vmin=-5, vmax=5, cbar=False, ax=a)
    a.set_title(title + ': mine (published)'); a.set_xlabel(''); a.set_ylabel('')
plt.savefig('fig4_topline_check.png', dpi=150, bbox_inches='tight'); plt.close()

# Weighted and unweighted % who've heard "a lot" about AI, per country and age group
grp = df.groupby(['country_name', 'age_grp'], observed=True)
heard_weighted = grp[['ai_heard', 'weight']].apply(lambda g: weighted_pct(g, 'ai_heard', 1)).unstack()
heard_unweighted = grp['ai_heard'].apply(lambda s: (s.dropna() == 1).mean() * 100).unstack()
group_sizes = grp.size().unstack(fill_value=0)

# Fig 5: weighted % heard "a lot", countries ordered by the gap between the youngest and oldest group
order = (heard_weighted['18-29'] - heard_weighted['65+']).sort_values().index
heat(heard_weighted.loc[order], '% who have heard "a lot" about AI (weighted)', 'fig5_heard_a_lot.png', cbar_label='%')

# Fig 6: weighted minus unweighted, in percentage points (shows what the weights change)
heat(heard_weighted - heard_unweighted, 'Weighted minus unweighted (percentage points)', 'fig6_weight_effect.png', fmt='.1f', cmap='coolwarm', center=0, cbar_label='points')

# Fig 7: respondents per country and age group (small groups mean wide uncertainty later)
heat(group_sizes, 'Respondents per country and age group', 'fig7_group_sizes.png', fmt='d', cmap='Blues', cbar_label='respondents')

# Tables for the report
heard_weighted.round(1).to_csv('table_heard_weighted.csv')
group_sizes.to_csv('table_group_sizes.csv')
(mine_q30 - published_q30).to_csv('table_topline_diff_q30.csv')
(mine_q31 - published_q31).to_csv('table_topline_diff_q31.csv')
