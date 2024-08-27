import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score

def plot_categories(df, features, target):

    num_cols = min(2, len(features))
    num_rows = (len(features) - 1) // 2 + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(9, 3*num_rows))

    if num_rows == 1:
        axes = axes.reshape(1, -1)

    colors = ['tab:green', 'tab:red']

    for i, column in enumerate(features):
        row_idx = i // num_cols
        col_idx = i % num_cols
        sns.countplot(x=column, data=df, hue=target, ax=axes[row_idx, col_idx], palette=colors, saturation=0.5, width=0.75)
        axes[row_idx, col_idx].set_xlabel(column)
        axes[row_idx, col_idx].set_ylabel('Count')
        axes[row_idx, col_idx].legend(title=target, loc='upper right', facecolor='white')

    for i in range(len(features), num_rows * num_cols):
        row_idx = i // num_cols
        col_idx = i % num_cols
        fig.delaxes(axes[row_idx, col_idx])

    plt.tight_layout()
    plt.show()




def plot_histograms(df, features, target):

    num_cols = min(2, len(features))
    num_rows = (len(features) - 1) // 2 + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(9, 3*num_rows))

    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for i, column in enumerate(features):
        row_idx = i // num_cols
        col_idx = i % num_cols
        colors = ['tab:green', 'tab:red']
        for i, v in enumerate(sorted(df[target].unique())):
            sns.histplot(df.loc[df[target] == v][column], ax=axes[row_idx, col_idx], color=colors[i])
        axes[row_idx, col_idx].set_xlabel(column) 

    for i in range(len(features), num_rows * num_cols):
        row_idx = i // num_cols
        col_idx = i % num_cols
        fig.delaxes(axes[row_idx, col_idx])

    plt.legend([0, 1], title=target, facecolor='white', loc='upper right')
    plt.tight_layout()
    plt.show()




def plot_roc_curve(models_dict, y_test):

    colors = ['tab:blue','tab:red','tab:orange','tab:green','tab:purple','tab:gray','tab:cyan','tab:pink']
    plt.figure(figsize=(8, 6))
    for (model_name, model_details), color in zip(models_dict.items(), colors):
        y_proba = model_details[0].predict_proba(model_details[1])[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, color=color, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='lightgray', linestyle='--', label='Random Model')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curves')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.show()