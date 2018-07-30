
# coding: utf-8

# In[17]:


from sklearn import tree
'''DATA is been stored or given here'''
x=[[181,80,44],[177,70,43],[161,70,44],[176,80,40],[160,50,30],[165,60,34],[141,60,44],[185,80,43],[178,76,42],[171,60,44],[176,65,42],[178,87,44],[181,70,40],[154,77,43],[181,80,44],[187,88,46]]
y=['male','female','female','male','female','male','male','female','male','female','male','female','female','male','male','female']

'''defining the variable to store the decision tree model'''

clf=tree.DecisionTreeClassifier()

'''trains the data'''
clf=clf.fit(x,y) 
predicition=clf.predict([[190,70,44]])
print(predicition)


# In[18]:


from sklearn.neighbors import KNeighborsClassifier
clf=clf.fit(x,y) 
predicition=clf.predict([[190,70,44]])
print(predicition)


# In[16]:


from sklearn.svm import SVC
clf = SVC()
clf.fit(x, y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
print(clf.predict([[190,70,44]]))

