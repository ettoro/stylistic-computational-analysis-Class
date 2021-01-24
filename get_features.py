def getfeatures(corpus):
  
  tokens_nopunct = []
  lower = []
  lemma = []
  pos = []
  lemma_nopunct = []
  stopwords = []
  stopwords_words = []
  len_words = []
  sent_count = []
  dependency = []
  bigrams_ww = []
  trigrams_ww = []
  bigrams_pos = []
  trigrams_pos = []
  typetoken_ratio = []
  
  for doc in nlp.pipe(corpus['text'].astype('unicode').values):
      if doc.is_parsed:
          tokens_nopunct.append([n.text for n in doc if n.is_punct != True])
          lower.append([n.lower_ for n in doc])
          lemma.append([n.lemma_ for n in doc])
          pos.append([n.pos_ for n in doc])
          lemma_nopunct.append([n.lemma_ for n in doc if n.is_punct != True])
          stopwords.append([n.is_stop == True for n in doc if n.is_punct != True])
          len_words.append([len(n) for n in doc])
          sent_count.append(len(list(doc.sents)))
          dependency.append([n.dep_ for n in doc])
      else:
          tokens_nopunct.append(None)
          lower.append(None)
          lemma.append(None)
          pos.append(None)
          lemma_nopunct.append(None)
          stopwords.append(None)
          len_words.append(None)
          sent_count.append(None)
          dependency.append(None)
          
  corpus['text_tokens_nopunct'] = tokens_nopunct
  corpus['text_lower'] = lower
  corpus['text_lemma'] = lemma
  corpus['text_pos'] = pos
  corpus['lemma_nopunct'] = lemma_nopunct
  corpus['stopwords'] = stopwords
  corpus['pos_count'] = corpus.text_pos.apply(lambda x: len(str(x).split(',')))
  corpus['stop_count'] = [sum(i) for i in corpus['stopwords']]
  corpus['len_words'] = len_words
  corpus['total_len_words'] = [sum(i) for i in corpus['len_words']]
  corpus['word_count'] = [len(i) for i in corpus['text_tokens_nopunct']]
  corpus['sent_count'] = sent_count
  corpus['prop_stopwords'] = corpus['stop_count'] / corpus['word_count']
  corpus['avg_word_length'] = corpus['total_len_words'] / corpus['word_count']
  corpus['avg_sent_length'] = corpus['word_count'] / corpus['sent_count']
  corpus['avg_sent_length_char'] = corpus['total_len_words'] / corpus['sent_count']
  corpus['dependency'] = dependency
  
  for doc in nlp.pipe(corpus['text_lower'].astype('unicode').values):
      if doc.is_parsed:
          stopwords_words.append([n for n in doc if n.is_stop == True])
      else:
          stopwords_words.append(None)
  corpus['stopwords_words'] = stopwords_words
      
  for line in corpus['text_pos']:
      bigrams_pos.append(list(nltk.bigrams(line)))
      trigrams_pos.append(list(nltk.trigrams(line)))
  corpus['bigrams_pos'] = bigrams_pos
  corpus['trigrams_pos'] = trigrams_pos
  
  for line in corpus['lemma_nopunct']:
      linetext = " ".join(str(x) for x in line)
      lexrich_line = LexicalRichness(linetext)
      typetoken_ratio.append(lexrich_line.ttr)

  corpus['typetoken_ratio'] = typetoken_ratio
  
getfeatures(corpus)

# selection of the features that will be used for classification
corpus = corpus[['character', 'avg_word_length', 'avg_sent_length', 'avg_sent_length_char', 'prop_stopwords', 'typetoken_ratio', 'text_pos', 'bigrams_pos', 'trigrams_pos']].copy()

# transform string-type features
mlb = MultiLabelBinarizer()
corpus = corpus.join(pd.DataFrame(mlb.fit_transform(corpus.pop('text_pos')),
                          columns=mlb.classes_,
                          index=corpus.index))

corpus = corpus.join(pd.DataFrame(mlb.fit_transform(corpus.pop('bigrams_pos')),
                          columns=mlb.classes_,
                          index=corpus.index))

corpus = corpus.join(pd.DataFrame(mlb.fit_transform(corpus.pop('trigrams_pos')),
                          columns=mlb.classes_,
                          index=corpus.index))

# adjust features in order to maintain only those that are 
features = corpus.drop(['character'],axis=1)
features = features[corpusfeaturecorpuss.columns[features.sum()>10]]
features['character'] = corpus['character']
