# %%
from utils import *

from utils_feature_extraction import *
# %%
with open('output/reading_times_dataset_splitted.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame.from_dict(data)
# see if we have nan values
print(df.isna().sum())
df.columns
# %%
df.head()
# %%
# see if there is a correlation between sent SD and norm_story_sentence_rt
feat1 = 'SD_sent'
feat2 = 'NORM_STORY_RT_PER_SENTENCE'

corr, p_value = spearmanr(df[feat1], df[feat2])
    
print(f"CORR {feat1} & {feat2}: {round(corr,4)}", f"P-value: {p_value}")

# plot it
plt.figure(figsize=(5, 3))
sns.set_style("whitegrid")
sns.scatterplot(x=feat1, y=feat2, data=df)

# %%
# Let's try to predict reading time w. sentiment features
# using linreg

difficulty_features = ['average_wordlen', 'msttr', 'average_sentlen', 'bzipr',
       'bigram_entropy', 'word_entropy', 'flesch_ease', 'dale_chall_new', 
       'nominal_ratio', 'NDD_mean', 'NDD_std', 'TTR_VERB', 'TTR_NOUN', 'FREQ_OF', 'FREQ_THAT']

sentiment_features = ['SD_sent','approx_entropy', 'hurst']

all_features = difficulty_features + sentiment_features

lindf = df.copy()

# scale features
for feat in all_features:
    if lindf[all_features].isnull().values.any():
        print("Missing values found in the features.")
    scaler = MinMaxScaler()
    lindf[feat] = scaler.fit_transform(lindf[feat].values.reshape(-1, 1))

print('df prepared, features scaled')

# %%
# plot all features against reading time
scores_list = sentiment_features
scores_list2 = difficulty_features

lindf['RT'] = lindf['NORM_STORY_RT_PER_SENTENCE']
plot_scatters(lindf, scores_list, 'RT', 'blue', 14, 4, remove_outliers=False, outlier_percentile=100, show_corr_values=True)
plot_scatters(lindf, scores_list2[:5], 'RT', 'blue', 20, 4, remove_outliers=False, outlier_percentile=100, show_corr_values=True)
plot_scatters(lindf, scores_list2[5:10], 'RT', 'blue', 20, 4, remove_outliers=False, outlier_percentile=100, show_corr_values=True)
plot_scatters(lindf, scores_list2[10:], 'RT', 'blue', 20, 4, remove_outliers=False, outlier_percentile=100, show_corr_values=True)

# %%
# plot a custom scatter
custom_scores = ['msttr', 'nominal_ratio', 'SD_sent']
plot_scatters(lindf, custom_scores, 'RT', 'blue', 12, 3.5, remove_outliers=False, outlier_percentile=100, show_corr_values=False)

# %%

# linreg experiment
use_type_features = 'all' # set this to 'stylistics', 'sentiment', or 'all'

# define what feats to use
if use_type_features == 'sentiment':
    use_features = sentiment_features
    print('using sentiment features to predict')
elif use_type_features == 'stylistics':
    use_features = difficulty_features
    print(f'using styl/syntactic features, n={len(difficulty_features)}, so performing a selection using recursive feature elimination')
elif use_type_features == 'all':
    use_features = all_features
    print(f'using ALL features, n={len(difficulty_features)}, so performing a selection using recursive feature elimination')

X = lindf[use_features]
y = lindf['NORM_STORY_RT_PER_SENTENCE']

# check how many features we are using, since we have so few datapoints
if len(use_features) > 3:
    # then we need to select 3
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=3)
    X_rfe = rfe.fit_transform(X, y)
    selected_features = [feature for feature, rank in zip(use_features, rfe.ranking_) if rank == 1]
    print(f"Selected features after RFE: {selected_features}")

    # Refit with selected features
    X = lindf[selected_features]
else:
    print(f'Using all features without RFE (number of features used={len(use_features)})')


lm = pg.linear_regression(X, y, relimp=True)
temp = lm[["coef", "r2", "se", "adj_r2", "pval", "relimp"]].iloc[1, :].to_dict()
temp = {key: round(value, 3) for key, value in temp.items()}
if len(use_features) < 4: 
    label = 'SENTIMENT features'
else:
    label = 'STYL/SYNTACTIC features'
print('OVERALL results for', label, '::', temp)
print(lm)
print()

for feature in use_features:
    # set the variables
    X = lindf[feature]
    y = lindf['NORM_STORY_RT_PER_SENTENCE']

    # Re-run regression with the selected features individually
    lm = pg.linear_regression(X, y, relimp=True)
    temp = lm[["coef", "r2", "se", "adj_r2", "pval", "relimp"]].iloc[1, :].to_dict()
    temp = {key: round(value, 3) for key, value in temp.items()}
    print('INDIVIDUAL', feature, '::', temp)


# %%
