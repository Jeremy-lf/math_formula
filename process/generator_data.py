import random

f = open('/Users/momo/Desktop/generator_latex.txt','a')




for i in range(100):


    #运算符


    a = random.choice([i for i in range(0, 10)])
    a1 = random.choice([i for i in range(0, 10)])
    a2 = random.choice([i for i in range(0, 10)])
    a3 = random.choice([i for i in range(0, 10)])
    a4 = random.choice([i for i in range(0, 10)])
    a5 = random.choice([i for i in range(0, 10)])
    b = random.choice(['a', 'b', 'm', 'n', 'q', 'p', 's', 't', 'r', 'f', 'g', 'k', 'l', 'v', 'z', 'c', 'd', 'y', 'i', 'm', 'n', 'w'])
    b1 = random.choice(['a', 'b', 'm', 'n', 'q', 'p', 's', 't', 'r', 'f', 'g', 'k', 'l', 'v', 'z', 'c', 'd', 'y', 'i', 'm', 'n', 'w'])
    b2 = random.choice(['a', 'b', 'm', 'n', 'q', 'p', 's', 't', 'r', 'x','y','f', 'g', 'k',  'z', 'c', 'd', 'y', 'i', 'm', 'n', 'w'])
    c = random.choice(['+', '-', r'\times', '\div'])
    c1 = random.choice(['+', '-', r'\times', '\div'])
    c2 = random.choice(['+', '-', r'\times', '\div'])
    e = random.choice(['\sin', '\cos', r'\tan', '\lg','\sin','\cos'])




    # if i % 2 == 0 and i not in [j for j in range(6,100,6)]:
    if i >50:
        d = b1
    else:
        d =''

    q1 = '\left\{'
    q2 = r'\begin{array}{ l }'

    t = random.random()
    if t>0.3:

        w1 =  q1+' '+ q2 + ' { { '+ str(a3) +' '+ str(c) +' '+ str(a) + ' '+ str(d) + ' ' + '=' + ' ' + str(a1) + ' ' + '} } ' + r'\\' + ' ' + '{ { ' + str(b2) +' '+ str(c1)+ ' '+ str(b1) + ' ' + '=' + ' ' + str(a2) + ' } }' + r' \\' +' '+ r'\end{array}' +' '+ r'\right.'
        f.write(w1+'\n')
        print(w1)
    if 0.3<t<0.5:
        w1 =  q1+' '+ q2 + ' { { '+ str(a3)+str(a1) +' '+ str(c1) +' '+ str(a) + ' '+ str(d) + ' ' + '=' + ' ' + str(e)+str(a1) + ' ' + '} } ' + r'\\' + ' ' + '{ { ' + str(b2) +' '+ str(c)+ ' '+ str(b1) + ' ' + '=' + ' ' + str(e) +str(a4)+'-'+str(b)+ ' } }' + r' \\' +' '+ r'\end{array}' +' '+ r'\right.'
        f.write(w1+'\n')
        print(w1)
    if 0.5<t<0.8:
        w1 =  q1+' '+ q2 + ' { { '+ str(a3)+str(a1) +' '+ str(c1) +' '+ str(a) +str(a2) +' '+ str(d) + ' ' +str(c2)+' '+ str(a4)+' '+'=' + ' ' + str(a1) + ' ' + '} } ' + r'\\' + ' ' + '{ { ' + str(a5) +' '+ str(c)+ ' '+ str(b1) + ' ' + '=' + ' ' + str(a2) + str(a1)+' '+str(c2)+str(a4)+ ' } }' + r' \\' +' '+ r'\end{array}' +' '+ r'\right.'

        f.write(w1+r'\\'+'\n')
        print(w1)

#     if i % 3 == 0:
#
#
#
#
#     if i % 5 == 0:
#
#
# a1 ='\left\{ \begin{array} { l } { { x = 0 } } \\ { { y = - \frac { 1 } { 2 } } } \\ \end{array} \right.'