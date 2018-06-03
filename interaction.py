import pandas as pd

def interact(df):
    numerics = df[['Names_scaled', 'SibSp_scaled', 'Parch_scaled', 'TicketPrefix_id_scaled', 'Fare_scaled', 'CabinNumber_scaled',
         'Pclass_scaled', 'Title_id_scaled', 'TicketNumber_scaled', 'CabinLetter_scaled', 'Embarked_scaled',
         'Age_scaled']]
    # Add your the terms you want here
    new_fields_count = 0
    for i in range(0, numerics.columns.size - 1):
        for j in range(0, numerics.columns.size - 1):
            if i <= j:
                name = str(numerics.columns.values[i]) + '*' + str(numerics.columns.values[j])
                df = pd.concat([df, pd.Series(numerics.iloc[:, i] * numerics.iloc[:, j], name=name)], axis=1)
                new_fields_count += 1
            if i < j:
                name = str(numerics.columns.values[i]) + "+" + str(numerics.columns.values[j])
                df = pd.concat([df, pd.Series(numerics.iloc[:, i] + numerics.iloc[:, j], name=name)], axis=1)
                new_fields_count += 1

            if not i == j:
                name = str(numerics.columns.values[i]) + "/" + str(numerics.columns.values[j])
                df = pd.concat([df, pd.Series(numerics.iloc[:, i] / numerics.iloc[:, j], name=name)], axis=1)

                name = str(numerics.columns.values[i]) + "-" + str(numerics.columns.values[j])
                df = pd.concat([df, pd.Series(numerics.iloc[:, i] - numerics.iloc[:, j], name=name)], axis=1)
                new_fields_count += 2
    return df