from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import display,Image
from prettytable import PrettyTable

iris=load_iris()
x=iris.data  #鸢尾花特征
y=iris.target#鸢尾花类型

'''按比例划分为训练集和测试集
其中x为要划分的鸢尾花样本特征集
y为要划分的鸢尾花分类结果
测试集样本占比20%
random_state随机数种子为1保证重复实验时可以得到一样的随机数'''
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# 调用tree.DecisionTreeClassifier()建立决策树，并采用entropy信息熵算法
clf=tree.DecisionTreeClassifier(criterion='entropy')
# 拟合x_train，y_train
clf.fit(x_train,y_train)
# 在测试集上进行测试并计算得分
y_pre=clf.predict(x_test)
score=clf.score(x_test,y_test)
print('score:',score)

table=PrettyTable(['序号',iris.feature_names[0],iris.feature_names[1],
                   iris.feature_names[2],iris.feature_names[3],
                   'y_true', 'y_pre'])
class_names=['Iris-setosa','Iris-versicolor','Iris-virginica']
for i in range(len(x_test)):
    row=[i+1]
    for j in x_test[i]:
        row.append(j)
    row.append(class_names[y_test[i]])
    row.append(class_names[y_pre[i]])
    table.add_row(row)
print(table)


# 画出决策树
dot_data=StringIO()
tree.export_graphviz(clf,out_file=dot_data,
               feature_names=iris.feature_names,  # 属性名称
               class_names=iris.target_names,  # 分类名称
               filled=True,# 由颜色标识不纯度
               rounded=True,  # 圆角框
               special_characters=True)  # 特殊字符
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('iris.png')
display(Image(graph.create_png()))


