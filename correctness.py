import random

M = pow(2,64)
delta = 2458341979796443375
delta_1 = 3966477760629218831

def Mul_DH(x, y):
    a = 17390643323242993303
    b = 11079525186552562668
    ab = 7322647857831352372
    print("a * b = ", a * b % M, " ab = ", ab)
    a1 = 15747976944791849518
    b1 = 6251972030160369573
    c1 = 15924838271767560556
    ab1 = 14336853182062326182
    ac1 = 1366764957930748264
    bc1 = 15195638085638512284
    abc1 = 3276958408196036616
    print("a1*b1 = ", a1*b1%M, " a1 * c1 = ", a1*c1%M, " b1*c1 = ", b1*c1%M, " a1*b1*c1 = ", a1*b1*c1%M)
    print("ab1 = ", ab1, " ac1 = ", ac1, " bc1 = ", bc1, " abc1 = ", abc1)
    d_x = delta * x % M
    d_y = delta * y % M
    xy = ((x - a)*(y-b) + a * (y-b) + b*(x-a) + ab) % M
    d_xy =  ((d_x - a1)*(d_y - b1)*(delta_1-c1) + \
            c1*(d_x - a1)*(d_y - b1) + a1*(d_y - b1)*(delta_1-c1) + b1*(d_x - a1)*(delta_1-c1)  \
            + ac1*(d_y - b1) + bc1 * (d_x - a1) + ab1*(delta_1-c1) + abc1) % M
    print("x*y = ", x*y%M, "xy = ", xy)
    print("d_x*d_y*d_1 = ", d_x * d_y * delta_1% M, "d_xy = ", d_xy)

#Mul_DH(1, 2)

def Trunc_DH(x, m, k = 64, lx = 60, ):
    r = random.randint(0, pow(2, lx-m)-1)
    r1 = random.randint(0, pow(2, m)-1)
    b = random.randint(0, 1)
    d_r = delta * r % M
    d_r1 = delta * r1 % M
    d_b = delta * b

    d_x = delta * x % M
    c = pow(2, k-lx-1) * (x + pow(2, lx)*b + pow(2, m)*r + r1) % M
    d_c = pow(2, k-lx-1) * (d_x + pow(2, lx)*d_b + pow(2, m)*d_r + d_r1) % M
    #print("c*delta = ", c*delta%M, " d_c = ", d_c)

    c1 = c >> (k - lx - 1)
    cl1 = 1 & (c1 >> (lx))
    #print("bin c: ", bin(c), " ", cl1)
    d_cl1 = delta * cl1
    v = b + cl1 - 2*cl1*b
    d_v = d_b + d_cl1 - 2*cl1*d_b
    #print("delta*v = ", delta*v%M, " d_v = ", d_v%M)
    x_2m = (((c1 % pow(2, lx)) >> m) - r + pow(2, lx-m)*v) % M
    d_x_2m = delta* ((c1 % pow(2, lx)) >> m) - d_r + pow(2, lx-m)*d_v
    #print("delta * x/2^m = ", delta*x_2m%M, " d_x_2m = ", d_x_2m%M)
    #print(x_2m)
    return x_2m

#Trunc_DH(12888, 1)
#Trunc_DH(12888, 2)
#Trunc_DH(12888, 3)
#Trunc_DH(12888, 4)
#Trunc_DH(12888, 5)
#Trunc_DH(12888, 60)

def GTEZ(x, lx = 60):
    random_integers = [random.randint(1, 100) for _ in range(lx + 3)]
    t = random.randint(0, 1)
    x = pow(-1, t) * x
    u_ = pow(-1, t)
    u0 = x
    uj = [x]
    for i in range(lx):
        x = Trunc_DH(x, 1, k = 64, lx = 60, )
        uj.append(x)
    print(uj)
    v_ = u_ + 3 * u0 - 1
    vj = [v_]
    for j in range(len(uj)):
        temp = 0
        for k in range(j, len(uj)):
            temp += uj[k]
        temp -= 1
        vj.append(temp)
    for i in range(len(vj)):
        vj[i] *= random_integers[i]
    gtez1 = 0
    if 0 in vj:
        gtez1 = 1
    gtez = t + gtez1 - 2 * t * gtez1
    print(gtez)
    return [gtez1, t]

GTEZ(12888)

def EQ(x=1, y=2):
    x_y = x-y
    y_x = y-x
    [getz_xy1, t0] = GTEZ(x_y)
    [getz_yx1, t1] = GTEZ(y_x)
    eq_xy1 = getz_xy1 ^ getz_yx1
    t = t0 ^ t1
    eq_xy = 1 - (t + eq_xy1 - 2*t*eq_xy1)
    print(eq_xy)

EQ(1, 2)