# %%
from utils import *

# %%

df = pd.read_excel('data/chicago/chicago_with_sentiment_apen.xlsx')
df.head()

# %%
cols_to_use = ['BOOK_ID', 'TITLE', 'AUTH_LAST', 'AUTH_FIRST', 'PUBL_DATE', 'GENDER',
 # sentiment features
 #'HURST', 'APPENT', # vader based sentiment features
 'MEAN_SENT', 'STD_SENT', 'STD_SENT_SYUZHET',
 'HURST_SYUZHET', 'APEN_SYUZHET_SLIDING',
 # stylistic features
  'WORDCOUNT', 'SENTENCE_LENGTH', 'AVG_WORDLENGTH', # simple stylistics
   'READABILITY_FLESCH_EASE', 'READABILITY_DALE_CHALL_NEW', 
 'SPACY_FUNCTION_WORDS', 'FREQ_OF', 'FREQ_THAT', 'PASSIVE_ACTIVE_RATIO', 'NOMINAL_VERB_RATIO', # syntactics
 'TTR_VERB', 'TTR_NOUN', 'MSTTR', # TTRs
 'SELF_MODEL_PPL', 
 'NDD_NORM_MEAN', 'NDD_NORM_STD',
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
# save the list of novels in a txt file in data, including AUTH_LAST and TITLE and the assigned score
with open('data/reader_level/dalvean_list_in_chicago.txt', 'w') as f:
    for i, row in merged.iterrows():
        f.write(f"{row['AUTH_LAST']} - {row['TITLE']} - {row['SCORE']}\n")


# make a histogram of the scores
plt.figure(figsize=(6, 3), dpi=500)
sns.histplot(merged['SCORE'], color='purple', kde=True)
plt.xlabel('Difficult Rank')

# %%

# now we want to predict merged['SCORE'] using the different feature categories
plot = True

carryout_feat_selection = None # 'PFA' or 'RFE' else None
correlation_threshold = 0.5 # for PFA


reader_level_scores_linreg_results = {}

feature_sets = [difficulty_features, sentiment_features, all_feats]
labels = ['styl/syntactic features', 'sentiment features', 'all features']
colors = ['teal', 'purple', 'orange']

# %%

for i, feature_set in enumerate(feature_sets):

    print(f'Running linear regression for {labels[i]}')
    print(f'Features used: {feature_set}')

    # check how many features we are using, since we have so few datapoints
    # to do RFE
    if carryout_feat_selection == 'RFE' and len(feature_set) > 3:
        # then we need to select fetaures
        model = LinearRegression()
        rfe = RFE(model, n_features_to_select=3)
        y = merged['SCORE']
        X = merged[feature_set]
        X_rfe = rfe.fit_transform(X, y)
        selected_features = [feature for feature, rank in zip(feature_set, rfe.ranking_) if rank == 1]
        print(f"Selected features after RFE: {selected_features}")
        print('')

        # Refit with selected features
        selected_features_formatted = ' + '.join(selected_features)

    # to do PFA
    elif carryout_feat_selection == 'PFA' and len(feature_set) > 3:
        y = merged['SCORE']
        X = merged[feature_set]
        
        # Compute the correlation matrix and distances
        corr_matrix = X.corr().abs()  # Use absolute values for clustering
        dist_matrix = 1 - corr_matrix  # Distance = 1 - correlation
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(squareform(dist_matrix), method='average')
        
        # Form clusters based on the updated threshold
        cluster_labels = fcluster(linkage_matrix, correlation_threshold, criterion='distance')
        
        # Select one representative feature from each cluster based on highest variance
        selected_features = []
        for cluster in set(cluster_labels):
            cluster_indices = [i for i, x in enumerate(cluster_labels) if x == cluster]
            cluster_features = [feature_set[idx] for idx in cluster_indices]
            
            # Choose the feature with the highest variance within the cluster
            if len(cluster_features) > 1:
                representative_feature = X[cluster_features].var().idxmax()
            else:
                representative_feature = cluster_features[0]
            
            selected_features.append(representative_feature)
        
        print(f"Selected features after PFA: {selected_features}")
        selected_features_formatted = ' + '.join(selected_features)

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
        sns.set_style('whitegrid')
        plt.figure(figsize=(6, 4), dpi=500)
        color = colors[i]
        sns.scatterplot(merged, x =y_pred, y=merged['SCORE'], color=color, alpha=0.4)
        plt.plot(y, y, color='red')
        # annotate the points with titles
        for j, txt in enumerate(merged['TITLE']):
            plt.annotate(txt, (y_pred[j], merged['SCORE'].iloc[j]), fontsize=6)
        plt.xlabel('Predicted DR')
        plt.ylabel('Actual DR')
        plt.title(f'Predicted vs actual DR - {labels[i]}')
        plt.show()

    save_extension = ''
    if carryout_feat_selection != None:
        save_extension = '_w_feat_selection' + f'{carryout_feat_selection}'

    with open(f'output/results/linreg_pred_DR{save_extension}.txt', 'w') as f:
        f.write(str(reader_level_scores_linreg_results))

# %%

# try but instead of RFE use PCA to reduce collinearity
from sklearn.decomposition import PCA

carryout_PCA = True
reader_level_scores_linreg_results = {}
plot = True

for i, feature_set in enumerate(feature_sets):
    
    print(f'Running linear regression for {labels[i]}')
    print(f'Features used: {feature_set}')
    
    # Check how many features we are using; apply PCA if necessary
    if carryout_PCA and len(feature_set) > 3:
        model = LinearRegression()
        pca = PCA(n_components=3)  # Choose number of components to retain
        y = merged['SCORE']
        X = merged[feature_set]
        X_pca = pca.fit_transform(X)
        
        # Update for summary display purposes
        selected_features_formatted = 'PCA Components'
        
        # Use statsmodels with transformed PCA components
        X_pca = sm.add_constant(X_pca)  # Adding a constant for intercept
        mod = sm.OLS(y, X_pca)
        res = mod.fit()
        
        # Display PCA summary
        print(f"Selected features after PCA: {selected_features_formatted}")
        print(res.summary())
        
    else:
        # Use original features without PCA
        selected_features_formatted = ' + '.join(feature_set)
        y, X = dmatrices(f'SCORE ~ {selected_features_formatted}', data=merged, return_type='dataframe')
        mod = sm.OLS(y, X)
        res = mod.fit()
        
        # Display non-PCA summary
        print(f"Using all features without PCA (number of features used={len(feature_set)})")
        print(res.summary())
    
    # Add results to dictionary
    reader_level_scores_linreg_results[labels[i]] = res.summary()
    
    # Get predictions
    if carryout_PCA and len(feature_set) > 3:
        y_pred = res.predict(X_pca)
    else:
        y_pred = res.predict(X)
    
    # Plot predictions vs. actuals if plotting is enabled
    if plot:
        sns.set_style('whitegrid')
        plt.figure(figsize=(6, 4), dpi=500)
        color = colors[i]
        sns.scatterplot(x=y_pred, y=merged['SCORE'], color=color, alpha=0.4)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='-')
        # annotate the points with titles
        for j, txt in enumerate(merged['TITLE']):
            plt.annotate(txt, (y_pred[j], merged['SCORE'].iloc[j]), fontsize=6)

        plt.xlabel('Predicted DF')
        plt.ylabel('Actual DF')
        plt.title(f'Predicted vs Actual DF - {labels[i]}')
        plt.show()
    
# Save results to file
with open('output/results/linreg_pred_DR_w_PCA.txt', 'w') as f:
    for label, summary in reader_level_scores_linreg_results.items():
        f.write(f"{label}:\n{summary}\n\n")
# %%
