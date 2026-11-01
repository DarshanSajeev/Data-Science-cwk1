import pandas as pd, numpy as np, matplotlib, matplotlib.pyplot as plt, seaborn as sns
columns = ['country', 'weight', 'age', 'age_australia', 'ai_heard', 'ai_cncexc', 'aireg_us', 'aireg_china', 'aireg_eu', 'aireg_eu', 'aireg_country', 'd_educ_argentina']
country_codes = {1: "U.S.", 2: "Canada", 3: "France", 4: "Germany", 5: "Greece", 6: "Hungary", 7: "Italy", 8: "Netherlands", 9: "Poland", 10: "Spain", 11: "Sweden", 12: "UK", 13: "Australia", 14: "India", 15: "Indonesia", 16: "Japan", 17: "South Korea", 18: "Israel", 19: "Turkey", 20: "Kenya", 21: "Nigeria", 22: "South Africa", 23: "Argentina", 24: "Brazil", 25: "Mexico"}
df = pd.read_csv('Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv', usecols=columns, low_memory=False)

for c in columns:
    if c != 'country':
        df[c] = pd.to_numeric(df[c], errors='coerce')
df['country_name'] = df['country'].map(country_codes)
 
# Keep raw copies of Q30 and Q31 before recoding
df['ai_heard_raw'] = df['ai_heard']
df['ai_cncexc_raw'] = df['ai_cncexc']
 
for c in columns:
    if c != 'country' and c != 'weight':
        if c == 'd_educ_argentina':
            missing_codes = [98, 99]
        elif c.startswith('aireg_'):
            missing_codes = [9, 98, 99]
        elif c == 'age':
            missing_codes = [98, 99]
        else:
            missing_codes = [8, 9, 98, 99]
        df[c] = df[c].replace(missing_codes, np.nan)


# Split data into generations and use the Australian age bands where exact age is unavailable.
age_labels = ['18-29', '30-49', '50-64', '65+']
df['age_grp'] = pd.cut(df['age'], bins=[17, 29, 49, 64, 200], labels=age_labels)
australian_age_grp = {1: '18-29', 2: '30-49', 3: '50-64', 4: '65+'}
df['age_grp'] = df['age_grp'].astype(object).fillna(df['age_australia'].map(australian_age_grp))
df['age_grp'] = pd.Categorical(df['age_grp'], categories=age_labels, ordered=True)
print(df['age_grp'].value_counts(dropna=False))
#print(df.groupby('country_name')['age_grp'].apply(lambda s: s.isna().mean()).round(3))
 
def weighted_pct(d, var, value):
    d = d.dropna(subset=[var, 'weight'])
    return d.loc[d[var] == value, 'weight'].sum() / d['weight'].sum() * 100

def topline(var_raw, codes, names, countries):
    d = df[df['country_name'].isin(countries)].copy()
    d['cat'] = d[var_raw].where(d[var_raw].isin(codes), 99)
    t = d.pivot_table(index='country_name', columns='cat', values='weight', aggfunc='sum')
    t = t.div(t.sum(axis=1), axis=0) * 100
    t.columns = [names.get(c, 'DK/Refused') for c in t.columns]
    return t.round(0)
 
check_countries = ['Canada', 'India', 'Japan']
print(topline('ai_heard_raw', [1, 2, 3], {1: 'A lot', 2: 'A little', 3: 'Nothing'}, check_countries))
print(topline('ai_cncexc_raw', [1, 2, 3], {1: 'Excited', 2: 'Concerned', 3: 'Equally'}, check_countries))
 
# % who've heard "a lot" about AI per country and age group, weighted vs unweighted (the difference shows what weighting changes)
grp = df.groupby(['country_name', 'age_grp'], observed=True)
heard_weighted = grp[['ai_heard', 'weight']].apply(lambda g: weighted_pct(g, 'ai_heard', 1)).unstack()
heard_unweighted = grp['ai_heard'].apply(lambda s: (s.dropna() == 1).mean() * 100).unstack()
print(heard_weighted.round(1))
print((heard_weighted - heard_unweighted).round(1))
 
# Number of respondents in each country x age group (small groups mean wide uncertainty later)
print(grp.size().unstack())