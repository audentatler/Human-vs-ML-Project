from data.fetch_data import get_data
from getting_started.make_plots import df
# Human Algorithm
def humanAlgorithm(val1, val2, val3):
    hNormal = 0
    hAltered = 0
    for feature in val1:
        if feature == 0.0:
            for feature2 in val2:
                if feature2 == 1.0:
                    for feature3 in val3:
                        if feature3 == 0.0:
                            break
                    break
                break
            hAltered += 1
        else:
            hNormal += 1

    return hNormal, hAltered

hN, hA = humanAlgorithm(df['accident'], df['surgical_intervention'], df['smoking'])
#print(hN)
#print(hA)

# Actual Result
def actualOutput(df):
    normal = 0
    altered = 0
    # iterrows returns (index, Series) pairs
    for _, row in df.iterrows(): 
        if row['diagnosis'] == 'N':
            normal += 1
        else:
            altered += 1
    return normal, altered

# Results/Accuracy
print(hN)
print(hA)
n, a = actualOutput(df)
hAccuracy = (hN + a)/100

print(f'Accuracy: {hAccuracy: .2%}\n')

print(f'- HUMAN PREDICTION -\nNormal: {hN}\nAltered: {hA}\n')

print(f'- ACTUAL DATA -\nNormal: {n}\nAltered: {a}')