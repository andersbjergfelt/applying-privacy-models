from numpy.core import numeric
import pandas as pd
import matplotlib.pylab as pl
import matplotlib.patches as patches
import csv
categorical = set((
    'learned_language',
    'native_language',
    'topic',
))

df = pd.read_csv('learned_language_topic_new.csv', sep=';',quotechar='"', engine='python');

for name in categorical:
    df[name] = df[name].astype('category')

def get_spans(df, partition, scale=None):
   
    spans = {}
    for column in df.columns:
        if column in categorical:
            span = len(df[column][partition].unique())
        else:
            span = df[column][partition].max()-df[column][partition].min()
        if scale is not None:
            span = span/scale[column]
        spans[column] = span
    return spans
    
full_spans = get_spans(df, df.index)
print(full_spans)



def split(df, partition, column):

    dfp = df[column][partition]
    if column in categorical:
        values = dfp.unique()
        lv = set(values[:len(values)//2])
        rv = set(values[len(values)//2:])
        return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
    else:        
        median = dfp.median()
        dfl = dfp.index[dfp < median]
        dfr = dfp.index[dfp >= median]
        return (dfl, dfr)

def is_k_anonymous(df, partition, sensitive_column, k=1):

    if len(partition) < k:
        # we cannot split this partition further...
        return False
    return True

def partition_dataset(df, feature_columns, sensitive_column, scale, is_valid):
    finished_partitions = []
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns], partition, scale)
        for column, span in sorted(spans.items(), key=lambda x:-x[1]):
            #we try to split this partition along a given column
            lp, rp = split(df, partition, column)
            if not is_valid(df, lp, sensitive_column) or not is_valid(df, rp, sensitive_column):
                continue
            # the split is valid, we put the new partitions on the list and continue
            partitions.extend((lp, rp))
            break
        else:
            # no split was possible, we add the partition to the finished partitions
            finished_partitions.append(partition)
    return finished_partitions

feature_columns = ['learned_language', 'topic']
sensitive_column = 'native_language'

finished_partitions = partition_dataset(df, feature_columns, sensitive_column, full_spans, is_k_anonymous)    


def agg_categorical_column(series):
    ## should be something else for numerical
    ##print([series])
    return series

def agg_numerical_column(series):
    return [series.mean(numeric_only=None)]

def build_anonymized_dataset(df, partitions, feature_columns, sensitive_column, max_partitions=None):
    aggregations = {}
    for column in feature_columns:
        if column in categorical:
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column
    rows = []
    for i, partition in enumerate(partitions):
        ##print(partition)
        if i % 100 == 1:
            print("Finished {} partitions...".format(i))
        if max_partitions is not None and i > max_partitions:
            break
        grouped_columns = df.loc[partition].agg(aggregations, squeeze=False)
        sensitive_counts = df.loc[partition].groupby(sensitive_column).agg({sensitive_column : 'count'})
        values = grouped_columns.iloc[0].to_dict()
        
        for sensitive_value, count in sensitive_counts[sensitive_column].items():
            if count == 0:
                continue
            values.update({
                sensitive_column : sensitive_value,
                'count' : count,
            })
            rows.append(values.copy())
        
    return pd.DataFrame(rows)


dfn = build_anonymized_dataset(df, finished_partitions, feature_columns, sensitive_column)
#### EXPORT TO CSV 
dfn.to_csv(r'export_anonymized-data_k_anonymity_group_users.csv_', index = False, header=True)
print("########K-ANONYMITY########")
print(dfn.sort_values(feature_columns+[sensitive_column]))

#----------------l-diversity----------------#
def diversity(df, partition, column):
    return len(df[column][partition].unique())


def is_l_diverse(df, partition, sensitive_column, l=2):
    return diversity(df, partition, sensitive_column) >= l


finished_l_diverse_partitions = partition_dataset(df, feature_columns, sensitive_column, full_spans, lambda *args: is_k_anonymous(*args) and is_l_diverse(*args))

column_x, column_y = feature_columns[:2]

print(len(finished_l_diverse_partitions))

dfl = build_anonymized_dataset(df, finished_l_diverse_partitions, feature_columns, sensitive_column)
#### EXPORT TO CSV 
##dfl.to_csv(r'export_anonymized-data_l_diversity_group_users.csv', index = False, header=True)
print("########l-DIVERSITY########")
print(dfl.sort_values([column_x, column_y, sensitive_column]))

global_freqs = {}
total_count = float(len(df))
group_counts = df.groupby(sensitive_column)[sensitive_column].agg('count')
for value, count in group_counts.to_dict().items():
    p = count/total_count
    global_freqs[value] = p


def t_closeness(df, partition, column, global_freqs):
    total_count = float(len(partition))
    d_max = None
    group_counts = df.loc[partition].groupby(column)[column].agg('count')
    for value, count in group_counts.to_dict().items():
        p = count/total_count
        d = abs(p-global_freqs[value])
        if d_max is None or d > d_max:
            d_max = d
    return d_max

def is_t_close(df, partition, sensitive_column, global_freqs, p=0.2):
    if not sensitive_column in categorical:
        raise ValueError("this method only works for categorical values")
    return t_closeness(df, partition, sensitive_column, global_freqs) <= p

finished_t_close_partitions = partition_dataset(df, feature_columns, sensitive_column, full_spans, lambda *args: is_k_anonymous(*args) and is_t_close(*args, global_freqs))

print(len(finished_t_close_partitions))

dft = build_anonymized_dataset(df, finished_t_close_partitions, feature_columns, sensitive_column)

print("########t-CLOSENESS########")
print(dft.sort_values([column_x, column_y, sensitive_column]))

#### EXPORT TO CSV 
dft.to_csv(r'export_anonymized-data_t_closeness.csv', index = False, header=True)

def build_indexes(df):
    indexes = {}
    for column in categorical:
        values = sorted(df[column].unique())
        indexes[column] = { x : y for x, y in zip(values, range(len(values)))}
    return indexes

def get_coords(df, column, partition, indexes, offset=0.1):
    if column in categorical:
        sv = df[column][partition].sort_values()
        l, r = indexes[column][sv[sv.index[0]]], indexes[column][sv[sv.index[-1]]]+1.0
    else:
        sv = df[column][partition].sort_values()
        next_value = sv[sv.index[-1]]
        larger_values = df[df[column] > next_value][column]
        if len(larger_values) > 0:
            next_value = larger_values.min()
        l = sv[sv.index[0]]
        r = next_value
    l -= offset
    r += offset
    return l, r

def get_partition_rects(df, partitions, column_x, column_y, indexes, offsets=[0.1, 0.1]):
    rects = []
    for partition in partitions:
        xl, xr = get_coords(df, column_x, partition, indexes, offset=offsets[0])
        yl, yr = get_coords(df, column_y, partition, indexes, offset=offsets[1])
        rects.append(((xl, yl),(xr, yr)))
    return rects

def get_bounds(df, column, indexes, offset=1.0):
    if column in categorical:
        return 0-offset, len(indexes[column])+offset
    return df[column].min()-offset, df[column].max()+offset

indexes = build_indexes(df)

def plot_rects(df, ax, rects, column_x, column_y, edgecolor='black', facecolor='none'):
    for (xl, yl),(xr, yr) in rects:
        ax.add_patch(patches.Rectangle((xl,yl),xr-xl,yr-yl,linewidth=1,edgecolor=edgecolor,facecolor=facecolor, alpha=0.5))
    ax.set_xlim(*get_bounds(df, column_x, indexes))
    ax.set_ylim(*get_bounds(df, column_y, indexes))
    ax.set_xlabel(column_x)
    ax.set_ylabel(column_y)


