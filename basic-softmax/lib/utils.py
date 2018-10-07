import os
import pandas as pd
from time import gmtime, strftime

SUMMARY_PATH ='./summaries'


def createDirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_summary(epochs, lr, optimizer, batch_sizes, val_losses, val_accuracies):
    table = {
        "Epoch": epochs,
        "Optimizer": [optimizer.__class__.__name__] * len(epochs),
        "Learning rate": [lr] * len(epochs),
        "Batch size": batch_sizes,
        "Val Loss": val_losses,
        "VAL ACCURACY": val_accuracies
    }
    summary = pd.DataFrame(table)
    summary.index = list(range(len(batch_sizes)))
    return summary


def max_acc(summary): return summary[["VAL ACCURACY"]].values.max()


def current_time(): return strftime("%Y-%m-%d_%H:%M:%S", gmtime())


def save_results(summary, test_labels, current_time):    
    createDirectory(SUMMARY_PATH)
    
    filename_prefix = f'{SUMMARY_PATH}/{max_acc(summary)}__{current_time}'

    summary.to_csv(f'{filename_prefix}__results.csv', sep=',',index=True,  index_label='Index')

    df = pd.DataFrame(data={"Category": test_labels}).astype(int)
    df.to_csv(f'{filename_prefix}__submission.csv', sep=',',index=True,  index_label='Id')    
    return summary