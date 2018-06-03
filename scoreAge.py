from sklearn.model_selection import cross_val_predict
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def scoreAge(path):
    a = pd.read_csv(path)
    knownAge=a[a.Age.notnull()]
    unknownAge=a[a.Age.isnull()]
    y = knownAge['Age'].values
    del knownAge['Age']
    aa = knownAge.drop('Survived',axis=1)
    X = aa
    X_test = unknownAge.drop(['Survived','Age'],axis=1)

    rfr=RandomForestRegressor(n_estimators=2000,n_jobs=-1)
    rfr.fit(X,y)
    print(cross_val_score(rfr,X,y,cv=3))

    from sklearn.metrics import accuracy_score