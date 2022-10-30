import matplotlib.pyplot as plt
import seaborn as sns

def main(df):
    '''
    Quicklook at the dataframe and its summary information.
    Info shown:
        - first 5 rows
        - shape (num rows, num cols)
        - data types of attributes (with attribute names)
        - class distribution
        - visualization of class distribution
    :param
        df: dataframe to be looked at
    :return:
        None
    '''

    # quicklook and summary info
    print('DATASET AT A GLANCE: \n', df.head())
    print('\nDATASET SHAPE: ', df.shape)
    print('\nDATASET ATTRIBUTES: \n', df.dtypes)
    print('\nDATASET TARGET DISTRIBUTION: \n', df.groupby(['isFraud']).count().iloc[:, 0])

    # # class distribution graph
    # fig = plt.figure(figsize=(16,9))
    # # graph labels, values, colors
    # labels = ['Valid', 'Fraud']
    # sizes = df.groupby(['isFraud']).count().iloc[:, 0].to_numpy()
    # colors = sns.color_palette('pastel')[0:2]
    # explode = (0.2, 0.0)
    # # plot graph
    # plt.pie(sizes, labels=labels,
    #         explode=explode, colors=colors, autopct='%1.2f%%', startangle=60)
    # plt.axis('equal')
    # plt.show()

    return