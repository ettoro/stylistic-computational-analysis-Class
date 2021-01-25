# specific feature across the whole corpus, not character related

sns.set(rc={'figure.figsize':(11.7,8.27)})

n_bins = 10
fig, axs = plt.subplots(2, 3)
axs[0,0].hist(df_grafici['feature'], bins = n_bins)
axs[0,0].set_title('feature')
axs[0,1].hist(df_grafici['feature2'], bins = n_bins)
axs[0,1].set_title('Feature2')


# feature for each character

fig, axs = plt.subplots(3,2)
fn = ['feature', 'feature2', 'feature3', 'feature4']
cn = ['character1', 'character2', 'character3', 'character4', 'character5']
sns.boxplot(x = 'character', y = 'feature', data = df, order = cn, ax = axs[0,0]);
sns.boxplot(x = 'character', y = 'feature2', data = df, order = cn, ax = axs[0,1])


# pairplot of features

sns.pairplot(df[['feature', 'feature2', 'feature3', 'feature4', 'character']], hue='character', height = 2, palette = 'colorblind')

