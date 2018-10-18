
def model_selection_scores(data, mpl, plt, pca=True, x=9, y=5.5, n=.3125, fs=16, cmap='viridis'):
    results = data[0]
    x_train_acc = []
    x_test_acc = []
    x_test_fbeta = []
    x_test_precision = []
    time = []
    for key in results.keys():
        x_train_acc.append(results[key][0])
        x_test_acc.append(results[key][1])
        x_test_fbeta.append(results[key][2])
        x_test_precision.append(results[key][3])
        time.append(results[key][4])

    '''
    f, axes = plt.subplots(3, 2, figsize=(20,20))
    plt.setp(axes, xticks=[i for i in range(len(x_train_acc))], xticklabels=[i for i in results.keys()])
    axes[0][0].bar([i for i in range(len(x_train_acc))], x_train_acc, color=['b','r','g','k','y'])
    axes[0][0].set_title('X Train Accuracy')
    axes[0][0].set_ylabel('Percent')
    for i, j in enumerate(x_train_acc):
        axes[0][0].text(i-.3, j, '{:4f}'.format(j), fontsize=12)
        
    axes[0][1].bar([i for i in range(len(x_test_acc))], x_test_acc, color=['b','r','g','k','y'])
    axes[0][1].set_title('X Test Accuracy')
    axes[0][1].set_ylabel('Percent')
    for i, j in enumerate(x_test_acc):
        axes[0][1].text(i-.3, j, '{:4f}'.format(j))
        
    axes[1][0].bar([i for i in range(len(x_test_fbeta))], x_test_fbeta, color=['b','r','g','k','y'])
    axes[1][0].set_title('X Test Fbeta Score')
    axes[1][0].set_ylabel('Percent')
    for i, j in enumerate(x_test_fbeta):
        axes[1][0].text(i-.3, j, '{:4f}'.format(j))
        
    axes[1][1].bar([i for i in range(len(x_test_precision))], x_test_precision, color=['b','r','g','k','y'])
    axes[1][1].set_title('X Test Precision Score')
    axes[1][1].set_ylabel('Percent')
    for i, j in enumerate(x_test_precision):
        axes[1][1].text(i-.3, j, '{:4f}'.format(j))
        
    axes[2][0].bar([i for i in range(len(time))], time, color=['b','r','g','k','y'])
    axes[2][0].set_title('Time To Train')
    axes[2][0].set_ylabel('Sec')
    for i, j in enumerate(time):
        axes[2][0].text(i-.3, j, '{:4f}'.format(j))

    axes[2][1].axis('off')
    '''

    #x, y = (13, 10)

    labels = ['XGB\nClassifier', 'SVC', 'AdaBoost\nClassifier', 'GaussianNB', 'Logistic\nRegression']
    save_labels = ['train_acc', 'test_acc', 'fbeta', 'precision', 'time']

    mpl_colors = [mpl.cm.get_cmap(cmap)((i+1)/5) for i in range(5)]

    plt.figure(1)
    fig, ax = plt.subplots(figsize=(x, y))
    plt.setp(ax, xticks=[i for i in range(len(x_train_acc))], xticklabels=labels)
    plt.bar([i for i in range(len(x_train_acc))], x_train_acc, color=mpl_colors)
    #plt.bar([i for i in range(len(x_train_acc))], x_train_acc, color=colors)
    plt.ylabel('Percent')
    for i, j in enumerate(x_train_acc):
        ax.text(i-n, j, '{:.5f}'.format(j), fontsize=fs)
    if pca:
        plt.savefig(save_labels[0] + '_pca.png', dpi=300)
    else:
        plt.savefig(save_labels[0] + '_notpca.png', dpi=300)

    plt.figure(2)
    fig, ax = plt.subplots(figsize=(x, y))
    plt.setp(ax, xticks=[i for i in range(len(x_train_acc))], xticklabels=labels)
    plt.bar([i for i in range(len(x_test_acc))], x_test_acc, color=mpl_colors)
    plt.ylabel('Percent')
    for i, j in enumerate(x_test_acc):
        ax.text(i-n, j, '{:.5f}'.format(j), fontsize=fs)
    if pca:
        plt.savefig(save_labels[1] + '_pca.png', dpi=300)
    else:
        plt.savefig(save_labels[1] + '_notpca.png', dpi=300)

    plt.figure(3)
    fig, ax = plt.subplots(figsize=(x, y))
    plt.setp(ax, xticks=[i for i in range(len(x_train_acc))], xticklabels=labels)
    plt.bar([i for i in range(len(x_test_fbeta))], x_test_fbeta, color=mpl_colors)
    plt.ylabel('Percent')
    for i, j in enumerate(x_test_fbeta):
        ax.text(i-n, j, '{:.5f}'.format(j), fontsize=fs)
    if pca:
        plt.savefig(save_labels[2] + '_pca.png', dpi=300)
    else:
        plt.savefig(save_labels[2] + '_notpca.png', dpi=300)

    plt.figure(4)
    fig, ax = plt.subplots(figsize=(x, y))
    plt.setp(ax, xticks=[i for i in range(len(x_train_acc))], xticklabels=labels)
    plt.bar([i for i in range(len(x_test_precision))], x_test_precision, color=mpl_colors)
    plt.ylabel('Percent')
    for i, j in enumerate(x_test_precision):
        ax.text(i-n, j, '{:.5f}'.format(j), fontsize=fs)
    if pca:
        plt.savefig(save_labels[3] + '_pca.png', dpi=300)
    else:
        plt.savefig(save_labels[3] + '_notpca.png', dpi=300)

    plt.figure(5)
    fig, ax = plt.subplots(figsize=(x, y))
    plt.setp(ax, xticks=[i for i in range(len(x_train_acc))], xticklabels=labels)
    plt.bar([i for i in range(len(time))], time, color=mpl_colors)
    plt.ylabel('Sec')
    for i, j in enumerate(time):
        if len(str(int(j))) == 2:
            format_fun = lambda x: '{:.4f}'.format(x)
        else:
            format_fun = lambda x: '{:.5f}'.format(x)
        ax.text(i-n, j, format_fun(j), fontsize=fs)
    if pca:
        plt.savefig(save_labels[4] + '_pca.png', dpi=300)
    else:
        plt.savefig(save_labels[4] + '_notpca.png', dpi=300)

    plt.show()

def model_selection_pca_plots(data, mpl, plt):
    f, axes = plt.subplots(3, 2, figsize=(20,20))
    plt.setp(axes, ylabel='PCA2', xlabel='PCA1', yscale='symlog', xscale='symlog')
    linthreshx = 0.25
    linthreshy = 0.3
    axes[0][0].scatter(data[2][:,0], data[2][:,1], c=data[3])
    axes[0][0].set_title('True Values')
    axes[0][0].set_xscale('symlog', linthreshx=linthreshx)
    axes[0][0].set_yscale('symlog', linthreshy=linthreshy)

    axes[0][1].scatter(data[2][:,0], data[2][:,1], c=data[1]['AdaBoostClassifier'])
    axes[0][1].set_title('AdaBoostClassifier')
    axes[0][1].set_xscale('symlog', linthreshx=linthreshx)
    axes[0][1].set_yscale('symlog', linthreshy=linthreshy)

    axes[1][0].scatter(data[2][:,0], data[2][:,1], c=data[1]['GaussianNB'])
    axes[1][0].set_title('GaussianNB')
    axes[1][0].set_xscale('symlog', linthreshx=linthreshx)
    axes[1][0].set_yscale('symlog', linthreshy=linthreshy)

    axes[1][1].scatter(data[2][:,0], data[2][:,1], c=data[1]['Logistic Regression'])
    axes[1][1].set_title('Logistic Regression')
    axes[1][1].set_xscale('symlog', linthreshx=linthreshx)
    axes[1][1].set_yscale('symlog', linthreshy=linthreshy)

    axes[2][0].scatter(data[2][:,0], data[2][:,1], c=data[1]['SVC'])
    axes[2][0].set_title('SVC')
    axes[2][0].set_xscale('symlog', linthreshx=linthreshx)
    axes[2][0].set_yscale('symlog', linthreshy=linthreshy)

    axes[2][1].scatter(data[2][:,0], data[2][:,1], c=data[1]['XGB Classifier'])
    axes[2][1].set_title('XGB Classifier')
    axes[2][1].set_xscale('symlog', linthreshx=linthreshx)
    axes[2][1].set_yscale('symlog', linthreshy=linthreshy)

    plt.show()

def GNB_predictions(data, mpl, plt, week, with_rs):
    from numpy import equal

    diff1 = equal(data[7], data[5])
    diff2 = equal(data[7], data[6])
    diff3 = equal(data[5], data[6])

    '''
    f, axes = plt.subplots(3, 2, figsize=(12,14))
    f.subplots_adjust(hspace=.25, wspace=.25)
    plt.setp(axes, ylabel='PCA2', xlabel='PCA1', yscale='symlog', xscale='symlog')
    linthreshx = 0.25
    linthreshy = 0.3

    axes[0][0].scatter(data[4][:,0], data[4][:,1], c=data[7])
    axes[0][0].set_title('True Values')
    axes[0][0].set_xscale('symlog', linthreshx=linthreshx)
    axes[0][0].set_yscale('symlog', linthreshy=linthreshy)

    axes[0][1].scatter(data[4][:,0], data[4][:,1], c=diff3)
    axes[0][1].set_title('Difference Between GaussianNB and FIRST Values')
    axes[0][1].set_xscale('symlog', linthreshx=linthreshx)
    axes[0][1].set_yscale('symlog', linthreshy=linthreshy)

    axes[1][0].scatter(data[4][:,0], data[4][:,1], c=data[5])
    axes[1][0].set_title('GaussianNB Predictions')
    axes[1][0].set_xscale('symlog', linthreshx=linthreshx)
    axes[1][0].set_yscale('symlog', linthreshy=linthreshy)

    axes[1][1].scatter(data[4][:,0], data[4][:,1], c=diff1)
    axes[1][1].set_title('Difference Between GaussianNB and True Values')
    axes[1][1].set_xscale('symlog', linthreshx=linthreshx)
    axes[1][1].set_yscale('symlog', linthreshy=linthreshy)

    axes[2][0].scatter(data[4][:,0], data[4][:,1], c=data[6])
    axes[2][0].set_title('FIRST Predictions')
    axes[2][0].set_xscale('symlog', linthreshx=linthreshx)
    axes[2][0].set_yscale('symlog', linthreshy=linthreshy)

    axes[2][1].scatter(data[4][:,0], data[4][:,1], c=diff2)
    axes[2][1].set_title('Difference Between FIRST and True Values')
    axes[2][1].set_xscale('symlog', linthreshx=linthreshx)
    axes[2][1].set_yscale('symlog', linthreshy=linthreshy)
    '''

    add_on = "_Week_" + str(week) + "_With_RS_" + str("True" if with_rs else "False") + ".png"

    plt.figure(1, figsize=(8,5.5))
    plt.scatter(data[4][:,0], data[4][:,1], c=data[7])
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.xscale('symlog', linthreshx=.25)
    plt.yscale('symlog', linthreshy=.3)
    plt.savefig('results/GaussianNB_True_Values'+add_on, dpi=300)

    plt.figure(2, figsize=(8,5.5))
    plt.scatter(data[4][:,0], data[4][:,1], c=diff3)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.xscale('symlog', linthreshx=.25)
    plt.yscale('symlog', linthreshy=.3)
    plt.savefig('results/GaussianNB_Diff_Gauss_FIRST'+add_on, dpi=300)

    plt.figure(3, figsize=(8,5.5))
    plt.scatter(data[4][:,0], data[4][:,1], c=data[5])
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.xscale('symlog', linthreshx=.25)
    plt.yscale('symlog', linthreshy=.3)
    plt.savefig('results/GaussianNB_Gauss'+add_on, dpi=300)

    plt.figure(4, figsize=(8,5.5))
    plt.scatter(data[4][:,0], data[4][:,1], c=diff1)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.xscale('symlog', linthreshx=.25)
    plt.yscale('symlog', linthreshy=.3)
    plt.savefig('results/GaussianNB_Diff_Gauss_True'+add_on, dpi=300)

    plt.figure(5, figsize=(8,5.5))
    plt.scatter(data[4][:,0], data[4][:,1], c=data[6])
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.xscale('symlog', linthreshx=.25)
    plt.yscale('symlog', linthreshy=.3)
    plt.savefig('results/GaussianNB_FIRST'+add_on, dpi=300)

    plt.figure(6, figsize=(8,5.5))
    plt.scatter(data[4][:,0], data[4][:,1], c=diff2)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.xscale('symlog', linthreshx=.25)
    plt.yscale('symlog', linthreshy=.3)
    plt.savefig('results/GaussianNB_Diff_FIRST_True'+add_on, dpi=300)

    plt.show()

def XGB_predictions(data, mpl, plt, week, with_rs):
    from numpy import equal

    diff1 = equal(data[7], data[5])
    diff2 = equal(data[7], data[6])
    diff3 = equal(data[5], data[6])

    '''
    f, axes = plt.subplots(3, 2, figsize=(12,14))
    f.subplots_adjust(hspace=.25, wspace=.25)
    plt.setp(axes, ylabel='PCA2', xlabel='PCA1', yscale='symlog', xscale='symlog')

    axes[0][0].scatter(data[4][:,0], data[4][:,1], c=data[7])
    axes[0][0].set_title('True Values')

    axes[0][1].scatter(data[4][:,0], data[4][:,1], c=diff3)
    axes[0][1].set_title('Difference Between XGB and FIRST Values')

    axes[1][0].scatter(data[4][:,0], data[4][:,1], c=data[5])
    axes[1][0].set_title('XGB Predictions')

    axes[1][1].scatter(data[4][:,0], data[4][:,1], c=diff1)
    axes[1][1].set_title('Difference Between XGB and True Values')

    axes[2][0].scatter(data[4][:,0], data[4][:,1], c=data[6])
    axes[2][0].set_title('FIRST Predictions')

    axes[2][1].scatter(data[4][:,0], data[4][:,1], c=diff2)
    axes[2][1].set_title('Difference Between FIRST and True Values')
    '''

    add_on = "_Week_" + str(week) + "_With_RS_" + str("True" if with_rs else "False") + ".png"

    plt.figure(1, figsize=(8,5.5))
    plt.scatter(data[4][:,0], data[4][:,1], c=data[7])
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.xscale('symlog', linthreshx=2.0)
    plt.yscale('symlog', linthreshy=1.5)
    plt.savefig('results/XGB_True_Values'+add_on, dpi=300)

    plt.figure(2, figsize=(8,5.5))
    plt.scatter(data[4][:,0], data[4][:,1], c=diff3)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.xscale('symlog', linthreshx=2.0)
    plt.yscale('symlog', linthreshy=1.5)
    plt.savefig('results/XGB_Diff_XGB_FIRST'+add_on, dpi=300)

    plt.figure(3, figsize=(8,5.5))
    plt.scatter(data[4][:,0], data[4][:,1], c=data[5])
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.xscale('symlog', linthreshx=2.0)
    plt.yscale('symlog', linthreshy=1.5)
    plt.savefig('results/XGB_XGB'+add_on, dpi=300)

    plt.figure(4, figsize=(8,5.5))
    plt.scatter(data[4][:,0], data[4][:,1], c=diff1)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.xscale('symlog', linthreshx=2.0)
    plt.yscale('symlog', linthreshy=1.5)
    plt.savefig('results/XGB_Diff_XGB_True'+add_on, dpi=300)

    plt.figure(5, figsize=(8,5.5))
    plt.scatter(data[4][:,0], data[4][:,1], c=data[6])
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.xscale('symlog', linthreshx=2.0)
    plt.yscale('symlog', linthreshy=1.5)
    plt.savefig('results/XGB_FIRST'+add_on, dpi=300)

    plt.figure(6, figsize=(8,5.5))
    plt.scatter(data[4][:,0], data[4][:,1], c=diff2)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.xscale('symlog', linthreshx=2.0)
    plt.yscale('symlog', linthreshy=1.5)
    plt.savefig('results/XGB_Diff_FIRST_True'+add_on, dpi=300)

    plt.show()