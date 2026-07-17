import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("dataset")
parser.add_argument("--histogram", action='store_true')
parser.add_argument("--scatter_plot", action='store_true')
parser.add_argument("--pair_plot", action="store_true")
parser.add_argument('--count', action='store_true')
parser.add_argument('--check_null', action='store_true')
parser.add_argument('--describe', action='store_true')

args = parser.parse_args()

COLUMNS = ['ID', 'diagnosis', 'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 
                 'compactness1', 'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1',
                 'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2', 'concavity2', ''
                 'concave_points2', 'symmetry2', 'fractal_dimension2',
                 'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3', 'compactness3', 'concavity3', 
                 'concave_points3', 'symmetry3', 'fractal_dimension3']


SIGNIFICANT_COLUMNS = ['diagnosis', 'radius1', 'perimeter1', 'area1', 'compactness1', 'concavity1', 'concave_points1']


try:
    dataset = pd.read_csv(args.dataset, names=COLUMNS, header=None)
    benign = dataset[dataset['diagnosis'] == 'B']
    malign = dataset[dataset['diagnosis'] == 'M']
    if dataset.isnull().values.any():
        raise ValueError
except (ValueError, pd.errors.EmptyDataError):
    print(f"error: could not read data or empty data detected\n{dataset.isnull().sum()}")

def histogram(feature):
    plt.hist(dataset[dataset['diagnosis'] == 'B'][feature], label='bening', alpha=0.1, color='green', density=True, bins='auto')
    plt.hist(dataset[dataset['diagnosis'] == 'M'][feature], label='malign', alpha=0.5, color='red', density=True, bins='auto')

    plt.legend(loc='upper right')
    plt.title(feature)
    plt.savefig('immages/histogram.png')
    print("histogram done!")

def pair_plot(dataset, COLUMNS):
    sns.pairplot(dataset, vars=COLUMNS[1:], hue="diagnosis")

    plt.tight_layout()
    plt.savefig('immages/pair_plot.png', dpi=300)
    plt.close()

if args.count:
    print(dataset['diagnosis'].value_counts())

if args.check_null:
    print(dataset.isnull().sum())

if args.describe:
    print(dataset[COLUMNS[2:]].describe())

if args.histogram:
    feature = str(input("insert a feature: "))
    histogram(feature=feature)

if args.pair_plot:
    significant_data = dataset[SIGNIFICANT_COLUMNS]
    pair_plot(significant_data, SIGNIFICANT_COLUMNS)