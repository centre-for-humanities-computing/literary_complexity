# %%
from utils import *

# %%

df = pd.read_excel('data/chicago/chicago_with_sentiment_apen.xlsx')
df.head()

# %%
cols_to_use = ['BOOK_ID', 'TITLE', 'AUTH_LAST', 'AUTH_FIRST', 'PUBL_DATE', 'GENDER',
 # sentiment features
 #'HURST', 'APPENT', # vader based sentiment features
 'MEAN_SENT', 'STD_SENT', 'STD_SENT_SYUZHET', 'MEAN_SENT_SYUZHET',

 'HURST_SYUZHET', 'APPENT_SYUZHET','APEN_SYUZHET_SLIDING',
 # stylistic features
  'WORDCOUNT', 'SENTENCE_LENGTH', 'AVG_WORDLENGTH', # simple stylistics
   'READABILITY_FLESCH_EASE', 'READABILITY_DALE_CHALL_NEW', 
 'SPACY_FUNCTION_WORDS', 'FREQ_OF', 'FREQ_THAT', 'PASSIVE_ACTIVE_RATIO', 'NOMINAL_VERB_RATIO', # syntactics
 'TTR_VERB', 'TTR_NOUN', 'MSTTR', # TTRs
 'SELF_MODEL_PPL', 
 'NDD_NORM_MEAN', 'NDD_NORM_STD', 'NDD_RAW_MEAN', 'NDD_RAW_STD',
 'BZIP_TXT','BIGRAM_ENTROPY', 'WORD_ENTROPY' # 'entropy' measures
 ]

# rename msttr
df.rename(columns={'MSTTR-100': 'MSTTR'}, inplace=True)

# normalize the function words by wordcount
df['SPACY_FUNCTION_WORDS'] = df['SPACY_FUNCTION_WORDS'] / df['WORDCOUNT']

df = df[cols_to_use]

print(len(df))
df.head()

# %%

difficulty_features = [
    'SENTENCE_LENGTH', 'AVG_WORDLENGTH', # simple stylistics
   'READABILITY_FLESCH_EASE', 
   'READABILITY_DALE_CHALL_NEW', 
 'SPACY_FUNCTION_WORDS', 
 'FREQ_OF', 'FREQ_THAT', 
 'NOMINAL_VERB_RATIO', # syntactics
 'TTR_VERB', 'TTR_NOUN', 
 'MSTTR', # TTRs
 'SELF_MODEL_PPL', 
 'NDD_NORM_MEAN', 'NDD_NORM_STD', 
 'BZIP_TXT','BIGRAM_ENTROPY', 'WORD_ENTROPY'
 ]

sentiment_features = ['STD_SENT_SYUZHET', 'HURST_SYUZHET', 'APEN_SYUZHET_SLIDING']

all_feats = difficulty_features + sentiment_features

lindf = df[difficulty_features + sentiment_features + ['AUTH_LAST', 'TITLE']].dropna()
print('number of titles after drop na:', len(lindf))

# normalize the range of all features using min max scaling
for feat in all_feats:
    scaler = MinMaxScaler()
    lindf[feat] = scaler.fit_transform(lindf[feat].values.reshape(-1, 1))
print('Regression experiments, df prepared, features scaled')



# show
lindf.head()
# %%
# Experiment: predict reader level score using stylistic features

# We load and merge the Dalvean list of books and his assigned complexity scores
dalvean_list = pd.read_excel('data/reader_level/dalvean_list.xlsx')
# make the author last name column
dalvean_list['AUTH_LAST'] = dalvean_list['AUTHOR'].str.split(',').str[0]
# lowercase titles
dalvean_list['TITLE'] = dalvean_list['TEXT'].str.lower()

# get the overlap w chicago
merged = lindf.merge(dalvean_list, on=['AUTH_LAST', 'TITLE'], how='inner')
print('number of Dalvean datapoints in Chicago:', len(merged))

# %%

# normalize the scores 0 to 1
#merged['SCORE'] = (merged['SCORE'] - merged['SCORE'].min()) / (merged['SCORE'].max() - merged['SCORE'].min())

# now we want to predict merged['SCORE'] using the stylistic features
plot = True
carryout_RFE = True

reader_level_scores_linreg_results = {}

feature_sets = [difficulty_features, sentiment_features, difficulty_features + sentiment_features]
labels = ['styl/syntactic features', 'sentiment features', 'all features']
colors = ['teal', 'purple', 'orange']

# %%

for i, feature_set in enumerate(feature_sets):

    print(f'Running linear regression for {labels[i]}')

    # check how many features we are using, since we have so few datapoints
    if carryout_RFE == True and len(feature_set) > 3:
        # then we need to select fetaures
        model = LinearRegression()
        rfe = RFE(model, n_features_to_select=3)
        X_rfe = rfe.fit_transform(X, y)
        selected_features = [feature for feature, rank in zip(feature_set, rfe.ranking_) if rank == 1]
        print(f"Selected features after RFE: {selected_features}")

        # Refit with selected features
        selected_features_formatted = ' + '.join(selected_features)
        X = merged[selected_features]
    else:
        print(f'Using all features without RFE (number of features used={len(feature_set)})')
        selected_features_formatted = ' + '.join(feature_set)

    # linreg proper
    y, X = dmatrices(f'SCORE ~ {selected_features_formatted}', data=merged, return_type='dataframe')
    mod = sm.OLS(y, X)
    res = mod.fit()
    print(f"{selected_features_formatted, labels[i]}")
    print(res.summary())

    # add results to dict
    reader_level_scores_linreg_results[labels[i]] = res.summary()

    # get the predicted values with sm
    y_pred = res.predict(X)

    # plot it
    if plot == True:
        plt.figure(figsize=(7, 5), dpi=500)
        sns.scatterplot(merged, x =y_pred, y=merged['SCORE'], color='teal', alpha=0.2)
        plt.plot(y, y, color='red')
        plt.xlabel('Predicted DR')
        plt.ylabel('Actual DR')
        plt.title(f'Predicted vs actual RT - {labels[i]}')
        plt.show()

    with open(f'output/results/linreg_pred_DR.txt', 'w') as f:
        f.write(str(reader_level_scores_linreg_results))



# %%