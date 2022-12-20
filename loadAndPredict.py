import trainTree as tt
import pickle

def convert_to_text(y):
    if y[0]==0:
        y='Iris-Setosa'
    elif y[0]==1:
        y='Iris-Versicolour'
    elif y[0]==2:
      y = 'Iris-Virginica'
    return y


if __name__=='__main__':
    model=pickle.load(open('decision_tree.model','rb'))
    parameters=['sepal length','sepal width','petal length','Wind']
    inputs=[]
    for i in parameters: 
        inputs.append(input('Enter value for '+i+': '))
    
    inputs=[inputs]
    print (inputs)
    y=convert_to_text(model.predict(inputs))

    print(y)