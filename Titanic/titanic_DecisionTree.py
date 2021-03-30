# 导入库
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# 导入数据并清洗
def read_dataset(data_link):
    data = pd.read_csv(data_link, index_col=0)  # 读取数据集，取第一列为索引
    data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True) # 删除掉三个无关紧要的特征
    labels = data['Sex'].unique().tolist()
    data['Sex'] = [*map(lambda x: labels.index(x), data['Sex'])] # 将字符串数值化
    labels = data['Embarked'].unique().tolist()
    data['Embarked'] = data['Embarked'].apply(lambda n: labels.index(n)) # 将字符串数值化
    data = data.fillna(0) # 将其余缺失值填充为0
    return data

train = read_dataset('data/train.csv')
y = train['Survived'].values  # 类别标签
x = train.drop(['Survived'], axis=1).values # 所有样本特征

# 对样本进行随意切割，得到训练集和验证集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.6)


clf = DecisionTreeClassifier()  # 先不进行调参，看训练样本和测试样本分数如何
clf.fit(x_train, y_train)
print('train score:', clf.score(x_train, y_train))
print('test score:', clf.score(x_test, y_test))


param = [{'criterion':['gini'],'max_depth': np.arange(20,50,10),'min_samples_leaf':np.arange(2,8,2),'min_impurity_decrease':np.linspace(0.1,0.9,10)},
             {'criterion':['gini','entropy']},
             {'min_impurity_decrease':np.linspace(0.1,0.9,10)}]
clf = GridSearchCV(DecisionTreeClassifier(),param_grid=param,cv=10)
clf.fit(x_train,y_train)
print(clf.best_params_,clf.best_score_)


# 按最优参数生成决策树
model = DecisionTreeClassifier(max_depth=20,min_impurity_decrease=0.1, min_samples_leaf=2)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('train score:', clf.score(x_train, y_train))
print('test score:', clf.score(x_test, y_test))
print("查准率：", metrics.precision_score(y_test,y_pred))
print('召回率:',metrics.recall_score(y_test,y_pred))
print('f1分数:', metrics.f1_score(y_test,y_pred)) #二分类评价标准

# 生成混淆矩阵
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
C2= confusion_matrix(y_test, y_pred)
sns.heatmap(C2,annot=True)
plt.show()