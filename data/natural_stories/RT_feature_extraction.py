#%%
from utils import *
from utils_feature_extraction import *

# %%
# Load the data
df = pd.read_csv('data/natural_stories/processed_RTs.tsv', sep='\t')

# make a zone/item column
df['zone_item'] = df['zone'].astype(str) + '_' + df['item'].astype(str)

df.head(10)
# %%
print(len(df))

mean_RTS = df[['item', 'zone', 'word', 'meanItemRT', 'sdItemRT', 'gmeanItemRT', 'gsdItemRT', 'zone_item']].drop_duplicates(subset=['zone_item', 'word', 'meanItemRT']).reset_index(drop=True)
print(len(mean_RTS))
mean_RTS.head(10)

# %%
# get the full texts
stories_full = pd.read_csv('data/all_stories.tok', sep='\t')
stories_full['zone_item'] = stories_full['zone'].astype(str) + '_' + stories_full['item'].astype(str)
stories_full.head()

# merge with reading times
stories_full = stories_full.merge(mean_RTS, on=['zone_item', 'word', 'zone', 'item'], how='left')
print(len(stories_full))
stories_full.head(5)

# %%

# make a loop to divide each story, get the sentences and corresponding reading times

# since we cannot use a tokenize function, we want to tokenize based on punctuation
# however, we want to minimize errors

# remove all . after mr/mrs etc
replacements = {
    'mr.': 'Mr',
    'mrs.': 'Mrs',
    'dr.': 'Dr',
    'prof.': 'Prof',
    'st.': 'St',
    'lt.': 'Lt',
    'gen.': 'Gen',
    'rev.': 'Rev',
    'col.': 'Col',
    'sgt.': 'Sgt',
    'capt.': 'Capt',
    'jr.': 'Jr',
    'sr.': 'Sr',
    'vs.': 'vs',
    'etc.': 'etc',
    'inc.': 'Inc',
    'ltd.': 'Ltd',
    'co.': 'Co'
}

# Iterate over the dataframe and replace words
for i, row in stories_full.iterrows():
    word = row['word'].lower()  # Convert to lowercase for comparison
    if word in replacements:
        # Update the word in the dataframe
        stories_full.at[i, 'word'] = replacements[word]

print('text is clean')
# %%

# now the loop
stories_dict = {}
range_1_10 = range(1, 10)

for j in range_1_10:
    word_list = []  # List to store sentences for the current story
    rt_list = []    # List to store RTs for the current story

    prv_wordlist = []  # To collect words for a sentence
    prv_rtlist = []    # To collect RTs for a sentence

    # Filter the rows for the current story (item)
    prv_df = stories_full.loc[stories_full['item'] == j]

    for i, row in prv_df.iterrows():
        word = row['word']
        prv_wordlist.append(word)
        rt = row['meanItemRT']
        prv_rtlist.append(rt)

        # If a sentence-ending punctuation is found
        if word.endswith('.') or word.endswith('?') or word.endswith('!') or word.endswith('...'):
            word_list.append(prv_wordlist)  # Add sentence to word_list
            rt_list.append(prv_rtlist)      # Add RTs to rt_list

            # Reset for the next sentence
            prv_wordlist = []
            prv_rtlist = []

    sentence_list = [' '.join(i) for i in word_list]

    # make sure to check short sentences
    for x in sentence_list:
        threshold = 20
        if len(x) < threshold:
            print(f'NB sentence < {threshold} toks::', x)

    mean_rt = [np.mean(x) for x in rt_list]

    # Store the collected sentences and RTs for the story
    stories_dict[j] = {'sentences': sentence_list, 'rts': mean_rt}

stories_dict

# %%
# make df
stories_rts = pd.DataFrame.from_dict(stories_dict, orient='index').reset_index()
#stories_rts['story_mean_rt_per_sentence'] = stories_rts['rts'].apply(lambda rts: np.mean(rts) if len(rts) > 0 else np.nan)
# rename index

stories_rts['story_total_rt'] = stories_rts['rts'].apply(lambda rts: np.sum(rts))
# count the number of sentences per story
stories_rts['n_sents'] = stories_rts['sentences'].apply(lambda sents: len(sents))
# get the normalized rt for story
stories_rts['NORM_STORY_RT_PER_SENTENCE'] = stories_rts['story_total_rt'] / stories_rts['n_sents']

stories_rts

# %%

## we want to create another dataset where the stories are split in two
# however, we want to keep as much data as possible

# add overlap between the two splits
split_dict = {}
split_dict2 = {}

# Define overlap size (for example, 20% of the midpoint)
overlap_ratio = 0.9

# Loop through each key and split its contents with overlap
for key, value in stories_dict.items():
    # Get the length of sentences and rts
    num_sentences = len(value['sentences'])
    num_rts = len(value['rts'])
    
    # Calculate the midpoint
    mid_point = num_sentences // 2  # Integer division for the middle index

    # Calculate the overlap size based on the midpoint
    overlap_size = int(mid_point * overlap_ratio)
    
    # Create two overlapping splits for each key
    split_dict[key] = {
        'sentences': value['sentences'][:mid_point + overlap_size],
        'rts': value['rts'][:mid_point + overlap_size]
    }

    key_2_name = str(key) + '_2'

    split_dict2[key_2_name] = {
        'sentences': value['sentences'][mid_point - overlap_size:],
        'rts': value['rts'][mid_point - overlap_size:]
    }

# Convert the two split dictionaries to DataFrames
split_df = pd.DataFrame.from_dict(split_dict, orient='index').reset_index()
#split_df['story_mean_rt_per_sentence'] = split_df['rts'].apply(lambda rts: np.mean(rts)) # why
split_df['story_total_rt'] = split_df['rts'].apply(lambda rts: np.sum(rts))
split_df['n_sents'] = split_df['sentences'].apply(lambda sents: len(sents))
split_df['NORM_STORY_RT_PER_SENTENCE'] = split_df['story_total_rt'] / split_df['n_sents']


split_df2 = pd.DataFrame.from_dict(split_dict2, orient='index').reset_index()
#split_df2['story_mean_rt_per_sentence'] = split_df2['rts'].apply(lambda rts: np.mean(rts)) # why
split_df2['story_total_rt'] = split_df2['rts'].apply(lambda rts: np.sum(rts))
split_df2['n_sents'] = split_df2['sentences'].apply(lambda sents: len(sents))
split_df2['NORM_STORY_RT_PER_SENTENCE'] = split_df2['story_total_rt'] / split_df2['n_sents']

# print the average number of sentences in each
print(f"Average number of sentences in the first split: {split_df['n_sents'].mean()}")
print(f"Average number of sentences in the second split: {split_df2['n_sents'].mean()}")
# and original sentence len
print(f"Average number of sentences in the original stories: {stories_rts['n_sents'].mean()}")

# Merge the two DataFrames
stories_split_rts = pd.concat([split_df, split_df2], ignore_index=True)

stories_split_rts


# %%
# Now we get features
## NB, set this to true if you want to extract data from the split stories

split = False

# define use_df (unsplit/split version of corpus)
if split == True:
    use_df = stories_split_rts
    print(f'OBS. Making a split of the stories, at {overlap_ratio}')
else:
    use_df = stories_rts
    print('OBS. Using the full stories')

# Make SA
syuzhet = importr('syuzhet')

# CHANGE THESE TO YOUR PREFERENCES
if len(use_df) > 10:
    out_dir = "output/split"
else:
    out_dir = "output"

language = "english"
#sentiment_method = "vader"

nlp = spacy.load("en_core_web_sm")

nltk.download("punkt")

# %%

# Get the texts into a dictionary with running text
text_dict = {}
for i, row in use_df.iterrows():
    content = " ".join(row['sentences'])  # Join the list of sentences into a single string
    ids = row['index']

    text_dict[ids] = content

text_dict
# %%

# Create the master dictionary
master_dict = {}


# Iterate over the dictionary items
for key, text in tqdm(text_dict.items(), total=len(text_dict)):
    temp = {}

    file_id = str(key)
    filename = Path(file_id)

    # Remove newline characters from the text
    cleaned_text = text.replace('\n', ' ')
    
    # Tokenize the text into sentences and words
    sents = sent_tokenize(cleaned_text, language='english')
    words = word_tokenize(cleaned_text, language='english')
    
    # get spacy attributes
    spacy_attributes = []
    for token in nlp(cleaned_text):
        token_attributes = get_spacy_attributes(token)
        spacy_attributes.append(token_attributes)

    spacy_df = create_spacy_df(spacy_attributes)

    save_spacy_df(spacy_df, filename, out_dir)

    # stylometrics
    # for words
    temp["word_count"] = len(words)
    temp["average_wordlen"] = avg_wordlen(words)
    temp["msttr"] = ld.msttr(words, window_length=100)

    # for sentences
    if len(sents) < 10:
        print(f"\n{key}")
        print("text not long enough for stylometrics\n")
        pass
    else:
        temp["average_sentlen"] = avg_sentlen(sents)
        temp["gzipr"], temp["bzipr"] = compressrat(sents)

    # bigram and word entropy
    try:
        temp["bigram_entropy"], temp["word_entropy"] = text_entropy(
            text, language=language, base=2, asprob=False
        )
    except:
        print(f"\n{key}")
        print("error in bigram and/or word entropy\n")
        pass

    #arc = get_sentarc(sents, sent_method=sentiment_method, lang=language)

    # doing the things that only work in English
    if language == "english":
        # readability
        try:
            (
                temp["flesch_grade"],
                temp["flesch_ease"],
                temp["smog"],
                temp["ari"],
                temp["dale_chall_new"],
            ) = text_readability(text)

        except:
            print(f"\n{key}")
            print("error in readability\n")
            pass
        
        # sentiment analysis
        valences = list(syuzhet.get_sentiment(sents, method='syuzhet'))
        temp['valences_syuzhet'] = valences
        temp['SD_sent'] = np.std(valences)
        temp['mean_sent'] = np.mean(valences)

        # we add apen and hurst

        # approximate entropy w sliding window
        try:
            temp["approx_entropy"] = calculate_approx_entropy_sliding(valences, window_size=20, step_size=10, dimension=2, tolerance='sd')
        except:
            print(f"\n{key}")
            print("error with approximate entropy\n")
            pass

        # hurst
        # we use the saffine package for this
        try:
            temp["hurst"] = get_hurst(valences)
        except:
            print(f"\n{key}")
            print("error with hurst\n")

    print(key)
    # saving it all
    master_dict[str(key)] = temp

  
# %%
# check
features_df = pd.DataFrame.from_dict(master_dict, orient='index').reset_index()
features_df
# %%
# merge features with stories
features_df['index'] = features_df['index'].astype(str)
use_df['index'] = use_df['index'].astype(str)
use_df.head(20)
# %%
merged = pd.merge(use_df, features_df, on='index', how='left')

merged.head(30)
# %%
# now we want to add the syntactic derived features
# get nominal_verb_ratio

if use_df.shape[0] > 10:
    path = 'output/split/spacy_books/'
else:
    path = 'output/spacy_books/'

for file in os.listdir(path):
    if file.endswith(".csv"):
        print('filename treated:', file)


# %%
spacy_dict = {}

for file in os.listdir(path):
    if file.endswith(".csv"):
        temp = {}

        data = pd.read_csv(path + file)

        new_list = data['token_pos_'].to_list()
        n_noun = new_list.count('NOUN')
        n_verb = new_list.count('VERB')
        n_adj = new_list.count('ADJ')

        temp['nominal_ratio'] = (n_noun + n_adj) / n_verb
        temp['nominal_ratio_inverse'] = n_verb / (n_noun + n_adj)

        filename = file.split('.')[0].split('_spacy')[0]

        # and dependency distances (NDDs)
        print(len(data))

        if len(data)>40:
            full_stop_indices = data[data['token_text'].str.strip() == '.'].index
            # Adding the last index of the DataFrame to handle the last sentence
            full_stop_indices = list(full_stop_indices) + [data.index[-1]]
        
            temp['NDD_mean'], temp['NDD_std'], temp['DD_mean'], temp['DD_std'] = calculate_dependency_distances(data, full_stop_indices)

        # and TTRs
            split = data.loc[10:1010]
            
            # Get TTR of verbs
            df_verbs = split.loc[split['token_pos_'] == 'VERB']
            no_verbs = len(df_verbs)
            raw_verb_tokens = list(df_verbs['token_text'].str.lower().astype(str))
            no_types_verbs = len(set(raw_verb_tokens))
            # print(no_verbs, no_types_verbs)
            temp['TTR_VERB'] = no_types_verbs/no_verbs

            # Get TTR of nouns
            df_nouns = split.loc[split['token_pos_'] == 'NOUN']
            no_nouns = len(df_nouns)
            raw_noun_tokens = list(df_nouns['token_text'].str.lower().astype(str))
            no_types_nouns = len(set(raw_noun_tokens))
            #print(no_nouns, no_types_nouns)
            temp['TTR_NOUN'] = no_types_nouns/no_nouns

            # Get no. of specific function words
            tokens = list(split['token_text'])
            # lowercase
            tokens_lower = [str(x).lower() for x in tokens]
            # total len
            no_tokens = len(tokens)

            # Get no. of "of"
            no_of_of = len([x for x in tokens_lower if str(x) == 'of'])
            normalized_no_of = no_of_of/no_tokens
            temp['FREQ_OF'] = normalized_no_of

            # Get no. of "that"
            no_of_that = len([x for x in tokens_lower if str(x) == 'that'])
            normalized_no_that = no_of_that/no_tokens
            temp['FREQ_THAT'] = normalized_no_that


        spacy_dict[filename] = temp


spacy_df = pd.DataFrame.from_dict(spacy_dict, orient='index').reset_index()
spacy_df.head(30)



# %%
# merge with everything else
spacy_df['index'] = spacy_df['index'].astype(str)
everything = pd.merge(merged, spacy_df, on='index', how='left')
everything

# %%

if len(use_df) > 10:
    path = 'output/reading_times_dataset_splitted.json'
else:
    path = 'output/reading_times_dataset.json'

with open(path, 'w') as f:
    f.write(everything.to_json(orient='records', indent=4))

# see if we can open
with open(path, 'r') as f:
    data = json.load(f)

pd.DataFrame.from_dict(data)

# beatiful
# %%
print('done with features')
# %%


# mean sentiment/sentence