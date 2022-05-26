print("hi")
def multi_run_wrapper(args):
    print("ok")
    return add(*args)


def add(x,y):
    X=[]
    Y=[]
    X.append(x)
    Y.append(y)
    return X,Y
