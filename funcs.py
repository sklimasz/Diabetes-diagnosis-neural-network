import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def leakyRELU(input):
    return np.maximum(0.01*input , 1*input)
    
def leakyRELU_der(x):
    data = [1 if value>0 else 0.01 for value in x]
    return np.array(data, dtype=float)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_der(input):
    dummy1 = sigmoid(input)
    dummy2 = (1-sigmoid(input))
    return dummy1*dummy2

def normalize(frame):
    dummy=[]
    for header in frame.columns.values:
        col = frame[header]
        min = frame[header].min()
        max = frame[header].max()
        dummy.append((col-min)/(max-min))
    return dummy

def visualize_results(data):
    # Convert data into a DataFrame
    df = pd.DataFrame.from_dict(data, orient='index', columns=['Accurate Diagnoses','Sensivitity', 'Specifity', 'Mean Squared Error'])
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Iterations'}, inplace=True)

    # Melt the DataFrame to convert multiple columns into one
    df_melted = pd.melt(df, id_vars='Iterations', var_name='Metrics', value_name='Values')

    # Plot using Seaborn
    sns.set(style="darkgrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Iterations', y='Values', hue='Metrics', data=df_melted)
    sns.scatterplot(x='Iterations', y='Values', hue='Metrics', data=df_melted, marker='o', s=70, legend=False)

    # Add title and labels
    plt.title('Neural Network Training Metrics')
    plt.xlabel('Iterations')
    plt.ylabel('Values')

    # Show the plot
    plt.show()