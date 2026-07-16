import pandas as pd
import matplotlib.pyplot as plt
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

columns_names = ['ID', 'diagnosis', 'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 
                 'compactness1', 'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1',
                 'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2', 'concavity2', ''
                 'concave_points2', 'symmetry2', 'fractal_dimension2',
                 'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3', 'compactness3', 'concavity3', 
                 'concave_points3', 'symmetry3', 'fractal_dimension3']


try:
    dataset = pd.read_csv(args.dataset, names=columns_names, header=None)
    if dataset.isnull().values.any():
        raise ValueError
except (ValueError, pd.errors.EmptyDataError):
    print(f"error: could not read data or empty data detected\n{dataset.isnull().sum()}")

if args.count:
    print(dataset['diagnosis'].value_counts())

if args.check_null:
    print(dataset.isnull().sum())

if args.describe:
    print(dataset.describe())


