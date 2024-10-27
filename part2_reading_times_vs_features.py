# %%
from utils import *

from utils_feature_extraction import *
# %%
with open('output/reading_times_dataset.json', 'r') as f:
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
# plot histogram
# make a histogram of the scores
plt.figure(figsize=(6, 3), dpi=500)
sns.histplot(df['NORM_STORY_RT_PER_SENTENCE'], color='purple', kde=True)
plt.xlabel('Reading Times')


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

# we choose feature category then format the strings to make into a patsy matrix

# linreg experiment
use_type_features = 'stylistics' # set this to 'stylistics', 'sentiment', or 'all'

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


# %%

rt_linreg_results = {}

# we use RFE to select features
X_b = lindf[use_features]
y_b = lindf['NORM_STORY_RT_PER_SENTENCE']

# check how many features we are using, since we have so few datapoints
if len(use_features) > 3:
    # then we need to select 3
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=3)
    X_rfe = rfe.fit_transform(X_b, y_b)
    selected_features = [feature for feature, rank in zip(use_features, rfe.ranking_) if rank == 1]
    print(f"Selected features after RFE: {selected_features}")

    selected_features_formatted = ' + '.join(selected_features)
    use_features_formatted = selected_features_formatted

else:
    print(f'Using all features without RFE (number of features used={len(use_features)})')
    selected_features_formatted = use_features
    use_features_formatted = ' + '.join(sentiment_features)


print("Linreg experiment. Patsy strings made")

# %%

y, X = dmatrices(f'NORM_STORY_RT_PER_SENTENCE ~ {use_features_formatted}', data=lindf, return_type='dataframe')
mod = sm.OLS(y, X)
res = mod.fit()
print(f"{use_features_formatted, use_type_features}")
print(res.summary())

# add results to dict
rt_linreg_results[use_type_features] = res.summary()

# get the predicted values with sm
y_pred = res.predict(X)

# plot it
plt.figure(figsize=(7, 5), dpi=500)
sns.scatterplot(lindf, x =y_pred, y=lindf['NORM_STORY_RT_PER_SENTENCE'], color='teal', alpha=0.2)
plt.plot(y, y, color='red')
plt.xlabel('Predicted RT')
plt.ylabel('Actual RT')
plt.title(f'Predicted vs actual RT - {use_type_features}')
plt.show()

with open(f'output/results/linreg_pred_based_on_{use_type_features}.txt', 'w') as f:
    f.write(str(rt_linreg_results))


# %%
