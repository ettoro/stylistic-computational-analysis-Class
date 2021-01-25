# PCA representation
df_graf = data.copy()
y = df_graf.iloc[:,-1]

np.random.seed(42)
rndperm = np.random.permutation(df_graf.shape[0])

pca = PCA(n_components=3)
pca_result = pca.fit_transform(df_graf.iloc[:,:-1].values)
df_graf['pca-one'] = pca_result[:,0]
df_graf['pca-two'] = pca_result[:,1] 
df_graf['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_)) # suggests which are the most relevant components

plt.figure(figsize=(8,5)) #2D
sns.scatterplot(
    x="pca-one", 
    y="pca-two",
    hue = 'character',
    palette=sns.color_palette('Dark2',6),
    data=df_graf.loc[rndperm,:],
    legend="full",
    alpha=0.3
)

ax = plt.figure(figsize=(10,7)).gca(projection='3d') #3D, no string-type for characters' names
ax.scatter(
    xs=df_graf.loc[rndperm,:]["pca-one"], 
    ys=df_graf.loc[rndperm,:]["pca-two"], 
    zs=df_graf.loc[rndperm,:]["pca-three"], 
    c=df_graf.loc[rndperm,:]["character_num"], 
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()


#T-SNE
df_tsne = features.copy()
df_tsne_f = df_tsne.iloc[:,:-2]
df_tsne_t = df_tsne.iloc[:,-1]


plt.figure(figsize = (8,4))
plt.subplots_adjust(top = 1.5)

for index, p in enumerate([1, 10, 25, 75]): # t-sne across several perplexity states

    tsne = TSNE(n_components = 2, perplexity = p, random_state=10)
    tsne_results = tsne.fit_transform(df_tsne_f)
    
    tsne_results=pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
    
    plt.subplot(2,2,index+1)
    plt.scatter(tsne_results['tsne1'], tsne_results['tsne2'], c=df_tsne_t, s=30)
    plt.title('Perplexity = '+ str(p))
plt.show()
