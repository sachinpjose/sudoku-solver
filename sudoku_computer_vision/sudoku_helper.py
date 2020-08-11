def convert(question):
    output = []
    temp = []
    for q in range(1 , len(question)+1):
        if q % 9 == 0:
            temp.append(question[q-1])
            output.append(temp)
            temp = []
        else :
            temp.append(question[q-1])
    return output


def re_convert(output):
    result = []
    print("*********************************")
    print(output)
    for out in output:
        print(out)
        for x in out:
            result.append(x)
    return result


def verify(actual, user):
    if sum(actual) == sum(user) :
        return True
    return  False