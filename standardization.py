data = features.values[:,:-1]

scalar=StandardScaler()
data = scalar.fit_transform(data)
data = DataFrame(data)
data['character'] = features['character']

