# %%
from utils import *

# %%
df = pd.read_excel('data/chicago/chicago_with_sentiment_apen.xlsx')
df.head()

# %%
plt.figure(figsize=(6, 3), dpi=500)
sns.histplot(df['PUBL_DATE'], color='purple', kde=True)
plt.xlabel('Year')

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

# rename mstrr-100 to msttr
df.rename(columns={'MSTTR-100': 'MSTTR'}, inplace=True)

# normalize the function words by wordcount
df['SPACY_FUNCTION_WORDS'] = df['SPACY_FUNCTION_WORDS'] / df['WORDCOUNT']

df = df[cols_to_use]

print(len(df))
df.head()


#df.to_csv('data/chicago/cvs_for_mart.csv', index=False)
# %%
## Experiment 1: examine relations between features

# heatmap
# Compute the correlation matrix
# get rid of the first columns
dt = df.drop(columns=['BOOK_ID', 'TITLE', 'AUTH_LAST', 'AUTH_FIRST', 'PUBL_DATE','GENDER'])

corr = dt.corr(method='spearman')
plt.figure(figsize=(20, 20))
sns.clustermap(corr,  linewidths=0.5, cbar=False, annot=True, figsize=(20, 15), method='ward')
# %%
# let's correleate where we have the apent, hurst and std_sent on the yaxis and the rest on the xaxis
  # Compute the correlation matrix for these columns
correlation_matrix = dt.corr(method='spearman')

# define the vertical columns
selected_rows = ['STD_SENT_SYUZHET', 'HURST_SYUZHET', 'APEN_SYUZHET_SLIDING']#, 'EMOTION_ENTROPY', 'SENT ENTROPY',] #,'self_model_ppl', , 'gpt2-xl_ppl']

# define all features
select = ['STD_SENT_SYUZHET','HURST_SYUZHET', 'APEN_SYUZHET_SLIDING', #'EMOTION_ENTROPY','SENT ENTROPY', # sentiment
          'SENTENCE_LENGTH', 'AVG_WORDLENGTH', # simple stylistics
   'READABILITY_FLESCH_EASE', 'READABILITY_DALE_CHALL_NEW', 
 'SPACY_FUNCTION_WORDS', 'FREQ_OF', 'FREQ_THAT', 'NOMINAL_VERB_RATIO', 'NDD_NORM_MEAN', 'NDD_NORM_STD', # dependency length # syntactics
 'TTR_VERB', 'TTR_NOUN', 'MSTTR', # TTRs
 'SELF_MODEL_PPL', # perplexity
 'BZIP_TXT','BIGRAM_ENTROPY', 'WORD_ENTROPY'] # entropy features

x_labels_nice = ['SD sent', 'Hurst', 'ApEn', #'emotion entropy', # sentiment
                  'Sentence length', 'Wordlength',
                  'R Flesch ease', 'R Dale chall',
                  'Function words', 'Freq \"of\"', 'Freq \"that\"', 'Nominal verb ratio','NDD mean', 'NDD SD', # syntactics
                  'TTR verb', 'TTR noun', 'MSTTR-100', # TTRs
                  'Perplexity',
                  'Compressibility','Bigram entropy', 'Word entropy']

y_labels = x_labels_nice[:3]

# defne matrix
data_for_correlation = dt[select]
selected_data = round(correlation_matrix[select].loc[selected_rows], 2)

# Plot the heatmap
plt.figure(figsize=(13, 1.5), dpi=300)
sns.heatmap(selected_data, cbar=False, annot=True, xticklabels=x_labels_nice, yticklabels=y_labels) 
           # mask=mask, linewidths=0.5)
# Rotate x-axis labels and set alignment
plt.xticks(rotation=80, ha='left',  ticks=range(len(select)), labels=x_labels_nice)

# Add a vertical line at the 4th column
plt.axvline(x=3, color='white', linestyle='-', linewidth=3.5)

plt.tight_layout()
plt.show()
# %%
# we want to inspect the relationship between the sentiment and the stylistic features

# for visualization, let's just remove the crazy outlier in ppl
scatter_df = df.loc[df['SELF_MODEL_PPL'] < 1000]
# and remove the low values of hurst 
scatter_df = scatter_df.loc[scatter_df['HURST_SYUZHET'] > 0.4]
# and remove the really high sd sent values
scatter_df = scatter_df.loc[scatter_df['STD_SENT_SYUZHET'] < 3]
print('len original df:', len(df), 'len filtered for ppl:', len(scatter_df), '\n')

# rename the columns
scatter_df['SD sent'] = scatter_df['STD_SENT_SYUZHET']
scatter_df['ApEn'] = scatter_df['APEN_SYUZHET_SLIDING']
scatter_df['Hurst'] = scatter_df['HURST_SYUZHET']
scatter_df.rename(columns={'AVG_WORDLENGTH': 'Avg. word length', 'NOMINAL_VERB_RATIO': 'Nominal Verb Ratio', 'READABILITY_FLESCH_EASE': 'Flesch-ease readability', 'NDD_NORM_MEAN': 'NDD mean', 'SELF_MODEL_PPL': 'Perplexity'}, inplace=True)

# just a selection of some features to visualize the correlation with std sent
scores_list = ['Avg. word length', 'Nominal Verb Ratio', 'Flesch-ease readability', 'NDD mean', 'Perplexity']


sns.set_style('whitegrid')
plot_scatters(scatter_df, scores_list, 'SD sent', '#385f71', 20, 4, remove_outliers=False, outlier_percentile=100, show_corr_values=True)

scores_list_2 = scores_list
plot_scatters(scatter_df, scores_list_2, 'ApEn', '#385f71', 20, 4, remove_outliers=False, outlier_percentile=100, show_corr_values=True)

scores_list_2 = scores_list
plot_scatters(scatter_df, scores_list_2, 'Hurst', '#385f71', 20, 4, remove_outliers=False, outlier_percentile=100, show_corr_values=True)




# %%
# Prediction experiments, using Linear regression

# Preparing df for regression experiments

difficulty_features = [
    'SENTENCE_LENGTH', 'AVG_WORDLENGTH', # simple stylistics
   'READABILITY_FLESCH_EASE', 
   'READABILITY_DALE_CHALL_NEW', 
 'SPACY_FUNCTION_WORDS', 
 'FREQ_OF', 'FREQ_THAT', 'NOMINAL_VERB_RATIO', # syntactics
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

# we convert data to a patsy frame
# for this we need to convert features to strings so we can pass those in the loop

# make difficutly features a string seperated by +
difficulty_features_formatted = ' + '.join(difficulty_features)
# same for sentiment features
sentiment_features_formatted = ' + '.join(sentiment_features)
# and all feats
all_feats_formatted = ' + '.join(all_feats)

print("patsy strings made")
# %%

sent_feats_linreg_results = {}

# now loop and do linreg

# predict sentiment features from styl/syntactic features
for feature in sentiment_features:

    y, X = dmatrices(f'{feature} ~ {difficulty_features_formatted}', data=lindf, return_type='dataframe')
    mod = sm.OLS(y, X)
    res = mod.fit()
    print(f"{feature}")
    print(res.summary())

    # add results to dict
    sent_feats_linreg_results[feature] = res.summary()

    # get the predicted values with sm
    y_pred = res.predict(X)

    # plot it
    plt.figure(figsize=(7, 5), dpi=500)
    sns.scatterplot(lindf, x =y_pred, y=lindf[feature], color='teal', alpha=0.2)
    plt.plot(y, y, color='red')
    plt.xlabel('Predicted Sentiment SD')
    plt.ylabel('Sentiment SD')
    plt.title('Predicted vs actual sentiment SD')
    plt.show()

with open('output/results/linreg_pred_sentiment.txt', 'w') as f:
    f.write(str(sent_feats_linreg_results))


# %%
# now loop
# predict stylistic features from sentiment features

stylistic_feats_linreg_results = {}

for feature in difficulty_features:

    y, X = dmatrices(f'{feature} ~ {sentiment_features_formatted}', data=lindf, return_type='dataframe')
    mod = sm.OLS(y, X)
    res = mod.fit()
    print(f"{feature}")
    print(res.summary())

    # add results to dict
    stylistic_feats_linreg_results[feature] = res.summary()

    # get the predicted values with sm
    y_pred = res.predict(X)

    # plot it
    plt.figure(figsize=(7, 5), dpi=500)
    sns.scatterplot(lindf, x =y_pred, y=lindf[feature], color='teal', alpha=0.2)
    plt.plot(y, y, color='red')
    plt.xlabel(f'Predicted {feature}')
    plt.ylabel(f'Actual {feature}')
    plt.title(f'Predicted vs actual {feature}')
    plt.show()

with open('output/results/linreg_pred_styl-syntactic.txt', 'w') as f:
    f.write(str(stylistic_feats_linreg_results))




# %%
print('Done with correlations and linreg in Chicago Corpus')


# %%
print('Additional plots')


# make an annotated plot with readability on one axis and the sentiment measures on the other
cols = ['READABILITY_DALE_CHALL_NEW','NOMINAL_VERB_RATIO']

measure = 'STD_SENT_SYUZHET'

# make a full name column
df['FULL_NAME'] = df['AUTH_FIRST'].str.split(' ').str[0] + ' ' + df['AUTH_LAST']

threshold = 100000  # Rating count threshold

for col in cols:
    plt.figure(figsize=(18, 8))
    
    sns.scatterplot(data=df, x=measure, y=col, hue='AUTH_LAST', palette='rocket', s=100, alpha=0.2)
    
    # Filter the DataFrame for annotation
    annot_df = df[df['RATING_COUNT'] > threshold].drop_duplicates(subset=['FULL_NAME'])
    
    # Annotate only those points with RATING_COUNT above the threshold
    for i in range(len(annot_df)):
        plt.text(annot_df[measure].iloc[i], annot_df[col].iloc[i], annot_df['AUTH_LAST'].iloc[i], fontsize=9, rotation=20, bbox=dict(facecolor='white', alpha=0.4, edgecolor='none'))
    
    plt.legend('')
    plt.xlabel(measure.replace('_', ' ').title())
    plt.ylabel(col.replace('_', ' ').title())
    plt.xlim(0, 2)
    #plt.ylim(0,4)
    plt.show()

# %%
