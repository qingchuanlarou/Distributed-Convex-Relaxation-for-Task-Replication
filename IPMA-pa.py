# 最终版 ：相比最初版修改了函数形式，把anm提到了指数的位置
# anm解主要取决于Fx、Cn、Pm
import winsound
import numpy as np
import datetime
import scipy.special
# np.set_printoptions(precision = 8)       # 浮点数组输出的精度位数，即小数点后位数(默认为8)。
np.set_printoptions(threshold = 1000000)   # 元素门槛值。数组个数沒有超过设置的阈值，NumPy就会将Python將所有元素列印出來。
from matplotlib import pyplot as plt
# 最小化凸函数，即对凹函数取负
# np.seterr(divide='ignore',invalid='ignore')  忽略所有警告
# e是障碍法中的参数1/t，c为回溯直线搜索中α，β取为0.5
# Ynm的累计分布函数服从指数分布
N = 40        # N个用户，M个服务器，
M = 10
anm = np.full((1,N*M), 0)                # 初始化自变量,含有N*M个元素,0.5为严格可行点
C1 = np.zeros((N, N*M))                  # 不等式1的系数矩阵，N行（N*M）列
C2 = np.zeros((M, N*M))                  # 不等式2的系数矩阵，M行（N*M)列
Cn = np.zeros((1,N))                     # 每个子任务的总成本限制，
Pm = np.zeros((1,M))                     # 直接初始化每个服务器上处理器数量限制，对应运行副本数(变为二维数组才能转置）
Cnm = np.zeros((N,M))                    # 单个副本的成本，均为1
obj = []
# 随机初始化Fx=Pr(Ynm<=dt)值

wt = 0.5                                                      # 公共延迟分量
tau = 1                                                      # 期望阈值为2至少大于公共延迟分量

# 帕累托分布
np.random.seed(1)                                           # 使用相同的seed()值，保证同一程序每次运行结果一样
# r = np.random.choice(N*M,size=N*M,replace=False,p=None)     # 在[0,N*M)中随机选取
r = np.random.randint(0,101,size=(1, N*M))
#Fx= np.random.randint(80, 100, size=(1, N*M)) / 100
lam = np.array([0.5+0.03*i for i in r])                     # xm和lam对应帕累托分布的两个参数
Fx = np.array(1 - np.power(wt/tau,lam)).reshape((1,N*M))        # Ynm的累积分布函数，Pr(Ynm<=dt)的取值
print("随机数r为:",r)
print("帕累托分布参数lam取值：", np.array([0.5+0.03*i for i in r]))
print("帕累托分布参数lam增序取值：", np.sort(lam,axis=None))
print("Pr(Ynm<=dt)取值:",Fx)
print("Pr(Ynm<=dt)增序取值:",np.sort(Fx,axis=None))              # 帕累托分布


# 初始化赋值Cnm，C1，C2，Cn
def init():
    for i in range(N):
        for j in range(M):
            Cnm[i][j] = 1                                 # 单个副本均为1
            C1[i][j + i * M] = Cnm[i][j]                  # 使非N所在系数为0

    for i in range(M):
        for j in range(N):
            C2[i][i + j * M] = 1                          # 使非M所在系数为0

    np.random.seed(1)
    for i in range(N):
        Cn[0,i] = np.random.randint(5,20)                  # 随机初始化每个子任务n的成本
    np.random.seed(1)
    for j in range(M):
        Pm[0,j] = np.random.randint(12,18)                # 随机初始化每个服务器的处理器数


# 求加上对数障碍的目标函数值
def f(x, e):                # 输入x为列向量
    global anm
    anm = x.T
    temp1 = 0
    for i in range(N):
        anm1 = anm[:,i*M:M+i*M]
        Fx1 = Fx[:,i*M:M+i*M]
        a = np.array(1 - Fx1)**anm1
        temp1 += np.log(1-np.prod(a))   # *是对应元素相乘,dot是矩阵相乘,prod将所有元素相乘
        #print(1-np.prod(1 - anm1*Fx1))
    temp2 = -np.sum(np.log(Cn.T-np.dot(C1,anm.T)))\
            -np.sum(np.log(Pm.T-np.dot(C2,anm.T)))-np.sum(np.log(anm)+np.log(1-anm))
    #print(Cn,Cn.T)
    #print(Cn.T-np.dot(C1,anm.T), np.sum(np.log(Pm.T-np.dot(C2,anm.T))))  #出现nan
    return -temp1/e + temp2  # 新的函数值
    # 警告：log中遇到无效值，比如 inf 或者 nan 等
    #        log0，inf超出浮点数的表示范围（溢出，即阶码部分超过其能表示的最大值）
    #       log负数，nan not a number对浮点数进行了未定义的操作


# 求原凸函数值(我们将凹函数取反求最小化）
def f1(x):
    global anm
    anm = x.T
    temp1 = 0
    for i in range(N):
        anm1 = anm[:, i * M:M + i * M]
        Fx1 = Fx[:, i * M:M + i * M]
        a = np.array(1 - Fx1)**anm1      # Fx1为底，anm1为幂
        temp1 += np.log(1 - np.prod(a))  # *是对应元素相乘,dot是矩阵相乘,prod将所有元素相乘
    return -temp1


# Gradient of f,求目标函数在x点的梯度值
def gradient(x, e):        # 输入x为列向量
    global anm
    anm = x.T
    grad = np.zeros(N*M)
    for i in range(N*M):  # 求梯度中的每一个元素
        n = i // M        # 确定i在数组哪一行,即哪个子任务(求商)
        m = i % M         # 确定i在数组哪一列,即哪一个服务器(求余)
        anm1 = anm[:,n*M:M+n*M]   # 取出与子任务n对应的的M个服务器
        Fx1 = Fx[:,n*M:M+n*M]     # 取出子任务n与M个服务器相连所用时间分布
        gd1 = -np.prod(np.array(1 - Fx1)**anm1)*np.log(1-Fx[0,i])/(1-np.prod(np.array(1 - Fx1)**anm1))    # 原函数梯度
        # C1[n,i],C2[m,i]写成了C1[n,m],C2[m,n]导致了这两个月停滞不前，致命错误！！！
        gd2 = C1[n,i]/(Cn[0,n]-np.dot(C1[n,:],anm.T))  # 不等式1的梯度
        if (Cn[0,n]-np.dot(C1[n,:],anm.T)) == 0:   # 分母为0
            gd2 = -99999999
        gd3 = C2[m,i]/(Pm[0,m]-np.dot(C2[m,:],anm.T))  # 不等式2的梯度
        if (Pm[0,m]-np.dot(C2[m,:],anm.T)) == 0:
            gd3 = -99999999
        gd4 = -1/anm[0,i] + 1/(1-anm[0,i])
        grad[i] = -gd1/e + (gd2 + gd3 + gd4)
    return grad.reshape((N*M,1))              # 返回梯度


# 求原函数梯度向量，输入x为列向量
def gradient2(x):
    global anm
    anm = x.T
    grad = np.zeros(N*M)
    for i in range(N*M):  # 求梯度中的每一个元素
        n = i // M        # 确定i在数组哪一行,即哪个子任务(求商)
        m = i % M         # 确定i在数组哪一列,即哪一个服务器(求余)
        anm1 = anm[:,n*M:M+n*M]   # 取出与子任务n对应的的M个服务器
        Fx1 = Fx[:,n*M:M+n*M]     # 取出子任务n与M个服务器相连所用时间分布
        gd5 = -np.prod(np.array(1-Fx1)**anm1)*np.log(1-Fx[0,i])/(1-np.prod(np.array(1-Fx1)**anm1))   # 原函数梯度
        # 注意anm[0,i]而不是anm[i],这是1行50列数组，而不是一维数组！
        grad[i] = -gd5
    return grad.reshape((N*M,1))              # 返回梯度


# Hessian Matrix of f,目标函数在x点的海森矩阵值
def hessian(x, e):
    global anm
    anm = x.T
    H = np.zeros((N*M,N*M))
    for i in range(N*M):
        for j in range(N*M):
            n1 = i // M     # 确定i在数组哪一行,即哪个子任务(求商)
            m1 = i % M      # 确定i在数组哪一列,即哪一个服务器(求余)
            n2 = j // M     # 遇上同理
            m2 = j % M
            if n1 == n2:     # 判断i与j是否为同一子任务,原函数与不等式1的二次导
                anm2 = anm[:,n1 * M:M + n1 * M]    # 取出与子任务n1对应的的M个服务器
                Fx2 = Fx[:,n1 * M:M + n1 * M]      # 取出子任务n1与M个服务器相连所用时间分布
                temp1 = -np.prod(np.array(1-Fx2)**anm2)*np.log(1-Fx[0,i])*np.log(1-Fx[0,j])*(1-np.prod(np.array(1-Fx2)**anm2))
                temp2 = -np.power(np.prod(np.array(1-Fx2)**anm2),2)*np.log(1-Fx[0,i])*np.log(1-Fx[0,j])
                temp3 = np.power(1-np.prod(np.array(1-Fx2)**anm2),2)
                H1 = (temp1+temp2)/temp3
                H2 = C1[n1,i]*C1[n1,j]/(np.power(Cn[0,n1]-np.dot(C1[n1,:],anm.T),2))
                #if float(np.power(Cn[0,n1]-np.dot(C1[n1,:],anm.T),2)) == 0:
                    #H2 = -99999999
            else:           # n1不等于n2则H1=0，H2=0，i的梯度不包含第j项
                H1 = 0
                H2 = 0
            if m1 == m2:    # 计算不等式2的二次导
                H3 = C2[m1,i]*C2[m1,j]/(np.power(Pm[0,m1]-np.dot(C2[m1,:],anm.T),2))
                #if float(np.power(Pm[0,m1]-np.dot(C2[m1,:],anm.T),2)) == 0:
                    #H3 = -99999999
            else:
                H3 = 0
            if i == j:     # 计算不等式3的二次导
                H4 = 1/np.power(anm[0,i],2) + 1/np.power((1-anm[0,i]),2)
            else:
                H4 = 0
            H[i,j] = -H1/e + (H2 + H3 + H4)
    # print(H,H.shape)
    return H


def h2(x):      # 原目标函数海森矩阵
    global anm
    anm = x.T
    H = np.zeros((N * M, N * M))
    for i in range(N * M):
        for j in range(N * M):
            n1 = i // M  # 确定i在数组哪一行,即哪个子任务(求商)
            n2 = j // M
            if n1 == n2:  # 判断i与j是否为同一子任务,原函数与不等式1的二次导
                anm2 = anm[:, n1 * M:M + n1 * M]  # 取出与子任务n1对应的的M个服务器
                Fx2 = Fx[:, n1 * M:M + n1 * M]  # 取出子任务n1与M个服务器相连所用时间分布
                temp1 = -np.prod(np.array(1 - Fx2) ** anm2) * np.log(1 - Fx[0, i]) * np.log(1 - Fx[0, j]) * (
                            1 - np.prod(np.array(1 - Fx2) ** anm2))
                temp2 = -np.power(np.prod(np.array(1 - Fx2) ** anm2), 2) * np.log(1 - Fx[0, i]) * np.log(1 - Fx[0, j])
                temp3 = np.power(1 - np.prod(np.array(1 - Fx2) ** anm2), 2)
                H5 = (temp1 + temp2) / temp3
            else:           # n1不等于n2则H1=0，H2=0，i的梯度不包含第j项
                H5 = 0
            H[i,j]=-H5
    return H


def h3(x,n):   # 局部变量海森矩阵
    global anm
    anm = x.T
    H = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            anm2 = anm[:, n * M:M + n * M]  # 取出与子任务n1对应的的M个服务器
            Fx2 = Fx[:, n * M:M + n * M]  # 取出子任务n1与M个服务器相连所用时间分布
            temp1 = -np.prod(np.array(1 - Fx2) ** anm2) * np.log(1 - Fx[0, i]) * np.log(1 - Fx[0, j])
            H[i,j] = temp1
    # print("P(x<t)的海森矩阵：",H)
    return H


# P444，回溯直线搜索
def backtracking(x0, delta, e, c):
    t = 1
    while True:
        x1 = x0 + t * delta
        if f(x1, e) <= (f(x0, e) + c * t * gradient(x0, e).T @ delta):  # c为α，取为0.01，x1-x0=t*delta,@为矩阵乘法
            print("回溯搜索步长为:",t)
            break
        else:
            t = t*0.5                # β取为0.5
    return x1,t                   # 返回更新后x值和步长t


# P465牛顿法，tol为误差阈值
def newton(x0, e, c, tol=1e-6):
    assert (0 < c < 0.5)          # 判断一个表达式，在表达式条件为 false 的时候触发异常
    num1 = 1                      # Newton迭代
    while True:
        delta = - np.linalg.inv(hessian(x0, e)).dot(gradient(x0, e))   # @为矩阵乘法，linalg.inv为矩阵求逆
        #print("牛顿步径及其形状:",delta.T,delta.shape)
        print("第%d次内部迭代:" % num1)
        x1,t1 = backtracking(x0, delta, e, c=c) # 返回列向量
        #print("更新前x值:",x0)
        print("更新后x值:",x1.T)
        #print(f(x0, e))
        #print("求两次x距离:",np.linalg.norm(x1 - x0))
        #if np.linalg.norm(x1 - x0) < tol:    # 默认求牛顿步径二次范数,解可能不是最优，但接近最优
        l2 = gradient(x0, e).T @ np.linalg.inv(hessian(x0, e)) @ gradient(x0, e)
        print("停止准则量:",l2[0,0]/2)
        if l2[0,0]/2 < tol:  # λ的平方     不采用牛顿减量,精度问题,回溯法中步径t很小时t*delta==0,x0+t*delta=x0,陷入死循环
            print("本轮内部迭代结束，找到x(e)")
            break
        else:
            x0 = x1
        num1 += 1
    return x0


# P543内点法
def interior_point(x0, e, c, tol=1e-6):
    assert (tol > 0)
    num2 = 1                         # 外部迭代次数
    while True:
        print("第%d次外部迭代:" % num2)
        #print("anm=", x0.T)
        #print("e=", e)
        xe = newton(x0, e, c=c)      # 返回列向量
        if not np.all((Cn.T - np.dot(C1, xe)) >= 0):  # 如果数组中的元素全部满足>=0 则返回False，否则返回True
            print("不满足条件1")
        if not np.all((Pm.T - np.dot(C2, xe)) >= 0):
            print("不满足条件2")
        if 2*N*M*e <= tol:                 # 相当于t>=1/tol
            break
        else:
            x0 = xe
            e /= 3                  # 相当于t=μt，μ=3
            obj.append(Pr(x0))
            num2 += 1
    print("求解结束，返回最优值！")
    return x0, num2


# 论文里面完成工作时间太于τ的概率
def Pr(x):
    p=1 - np.exp(-f1(x))
    return p


def pr_task(x,i):    # 计算任务i完成概率Pr(Xn<=dt)
    anm = x.T.copy()  # 行向量
    anmi = anm[:,i*M:M+i*M]
    Fxi = Fx[:,i*M:M+i*M]
    temp = 1 - np.prod(np.array(1 - Fxi) ** anmi)
    return temp


def variance(x):         # 求任务的方差
    arr = np.zeros(N)
    for n in range(N):   # 计算每个任务完成概率并存入数组
        arr[n] = pr_task(x,n)
    average = np.sum(arr)/N
    var = 0
    for n in range(N):
        var += np.square(arr[n]-average)
    print("算法任务方差为:",var*1e3)


# 局部优化交换法(按值的降序选择未选择的anm,并按值的升序选择所选的anm)
# 如果不满足条件则将剩余资源服务器和超出资源服务器进行交换
def opt(x0, k):
    arg = np.argsort(-x0.T)      # 取负使降序排序并提取对应索引
    arg = arg[0, :]
    k1 = k                       # 整数为1的个数
    # 可能的k1*(N*M-k1)次交换
    t1 = 0  # 最优的交换
    t2 = 0
    t3 = 0  # 最优的交换值
    t4 = 0  # P(X>τ)最优值
    xe = np.full((1, N * M), 0)
    for m in range(k1):  # 求最优解
        xe[0, arg[m]] = 1  # 每次重置为原始解
    # print("取整后任务中断概率:",Pr(xe.T))
    # print("取整后值:",xe)
    if np.all((Cn.T - np.dot(C1, xe.T)) >= 0) and np.all((Pm.T - np.dot(C2, xe.T)) >= 0):  # 当原始整数解满足条件时
        print("满足约束条件")
        for i in range(k1):          # 升序选择所选的anm
            for j in range(N*M-k1):  # 降序选择未选择的anm
                a = xe.copy()
                temp1 = f1(a.T)
                if i==0 and j==0:
                    t3=temp1         # 赋初值为原始解
                    t4 = Pr(a.T)
                a[0, arg[k1+j]], a[0, arg[k1-i-1]] = a[0, arg[k1-i-1]], a[0, arg[k1+j]]  # 交换两个anm值
                # print("arg中第%d与第%d个值交换，原函数最优值为:" % (k1 - i - 1, k1 + j), f1(a.T))
                # 判断是否满足约束条件,不满足则直接下一轮循环
                if not np.all((Cn.T - np.dot(C1, a.T)) >= 0):
                    # print("arg中第%d与第%d个值交换后不满足条件1" % (k1 - i - 1, k1 + j))
                    # print(Cn.T - np.dot(C1, a.T))
                    continue
                if not np.all((Pm.T - np.dot(C2, a.T)) >= 0):
                    # print("arg中第%d与第%d个值交换后不满足条件2" % (k1 - i - 1, k1 + j))
                    # print(Pm.T - np.dot(C2, a.T))
                    continue
                temp2 = f1(a.T)       # 满足约束条件交换后值
                if temp2 < temp1:    # 判断是否小于原函数值
                    print("arg中第%d与第%d个值交换后更优，原函数最优值更新为:" % (k1 - i - 1, k1 + j), f1(a.T))
                else:
                    # print("arg中第%d与第%d个值交换后没有更优" % (k1 - i - 1, k1 + j))
                    continue         # 交换没有更优则直接下次循环
                if temp2 < t3:       # 判断是否小于目前最小值
                    t1 = k1 - i - 1
                    t2 = k1 + j
                    t3 = f1(a.T)
                    t4 = Pr(a.T)
        xe[0, arg[t2]], xe[0, arg[t1]] = xe[0, arg[t1]], xe[0, arg[t2]]  # 交换两个anm值
        print("最终x的整数解和为:", k1)
        print("arg中第%d与第%d个值交换后最优，原函数最优值更新为:" % (t1, t2), t3)
        print("公式7 P(X>τ)最优值更新为", t4)
    else:   # 当原始整数解不满足条件时，先使之满足条件，再局部交换优化
        # l1 = Cn - np.dot(A1, xe.T)  # [N,1]   # 不等式1约束右边减左边
        l2 = Pm.T - np.dot(C2, xe.T)  # [M,1]   # 不等式2约束右边减左边
        if not np.all((Cn.T - np.dot(C1, xe.T)) >= 0):  # 如果数组中的元素全部满足>=0 则返回False，否则返回True
            print("不满足条件1,Cn-A1@x.T=", (Cn.T - np.dot(C1, xe.T)).T)
        if not np.all((Pm.T - np.dot(C2, xe.T)) >= 0):
            print("不满足条件2,Pm-A2@x.T=", (Pm.T - np.dot(C2, xe.T)).T)
        t1 = [i for i in range(M) if l2[i, 0] < 0]  # 求出为-1的索引号(服务器超出最大资源),t1和t2的长度相同
        t4 = [l2[i, 0] for i in range(M) if l2[i, 0] < 0]  # 求出小于于0的服务器对应超出资源数
        t2 = [i for i in range(M) if l2[i, 0] > 0]  # 求出大于0的索引号(服务器资源没用完)
        t3 = [l2[i, 0] for i in range(M) if l2[i, 0] > 0]  # 求出大于0的服务器对应剩余资源
        print("循环交换次数:",int(-sum(t4)))
        for i in range(int(-sum(t4))):          # 循环交换使所以服务器满足资源预算
            for n in range(k1):  # 升序将满足条件的anm变为0
                # n1 = arg[k1-n-1] // M         # 确定当前元素属于哪个子任务(求商)
                m1 = arg[k1-n-1] % M            # 确定当前元素属于哪个服务器(求余)
                if xe[0,arg[k1-n-1]] == 0:      # 已经变为0则直接跳过
                    continue
                if m1 == t1[0]:
                    xe[0,arg[k1-n-1]] = 0
                    # print("当前赋0结束,第%d个服务器满足约束条件" % t1[0])
                    t4[0] += 1     # 注意这里不是-=
                    if t4[0] == 0:  # 当前还有空闲服务器资源用完
                        t1.pop(0)  # 弹出当前服务器号
                        t4.pop(0)  # 弹出当前服务器
                    break
            l1t = Cn.T - np.dot(C1, xe.T)  # 每次循环更新一下信息
            for n in range(N*M-k1):     # 降序将属于还有余额服务器的第一个anm变为1
                n1 = arg[k1 + n] // M   # 确定当前元素属于哪个子任务(求商)
                m2 = arg[k1 + n] % M    # 确定当前元素属于哪个服务器(求余)
                if xe[0,arg[k1+n]] == 1:   # 已经变为1则直接跳过
                    continue
                if m2 == t2[0] and l1t[n1, 0] > 0:        # 取某个元素，所属服务器资源剩余，所属任务没达到最大约束
                    xe[0,arg[k1+n]] = 1
                    # print("当前赋1结束,第%d个服务器剩余资源-1" % t2[0])
                    t3[0] -= 1
                    if t3[0] == 0:       # 当前还有空闲服务器资源用完
                        t2.pop(0)        # 弹出当前服务器号
                        t3.pop(0)        # 弹出当前服务器
                    break
        print("约束条件1,Cn-A1@x.T=", (Cn.T - np.dot(C1, xe.T)).T)
        print("约束条件2,Pm-A2@x.T=", (Pm.T - np.dot(C2, xe.T)).T)
        print("最终x的整数解和为:", k1)
        print("公式7 P(X>τ)最优值更新为", Pr(xe.T))
    variance(xe.T)


def cpopt(x0,k):       # 优化cvxpy中结果，没用到
    k1 = k  # 整数为1的个数
    t1 = 0  # 最优的交换
    t2 = 0
    t3 = f1(x0)  # 最优的交换值
    t4 = 1  # P(x>τ)
    # 可能的k1*(N*M-k1)次交换
    for i in range(k1):  # 升序选择所选的anm
        for j in range(N * M - k1):  # 降序选择未选择的anm
            a = x0.copy().reshape((1, N * M))
            temp1 = f1(a.T)
            a[0,k1+j],a[0,k1-i-1]=a[0,k1-i-1],a[0,k1+j]   # 交换两个anm值
            # print("arg中第%d与第%d个值交换，原函数最优值为:" % (k1 - i - 1, k1 + j), f1(a.T))
            # 判断是否满足约束条件,不满足则直接下一轮循环
            if not np.all((Cn.T - np.dot(C1, a.T)) >= 0):  # 如果数组中的元素全部满足>=0 则返回False，否则返回True
                print("第%d与第%d个值交换后不满足条件1" % (k1 - i - 1, k1 + j))
                # print((Cn.T - np.dot(C1, anm3.T)), anm3)
                continue
            if not np.all((Pm.T - np.dot(C2, a.T)) >= 0):
                print("第%d与第%d个值交换后不满足条件2" % (k1 - i - 1, k1 + j))
                # print((Pm.T-np.dot(C2,anm3.T)), anm3)
                continue
            temp2 = f1(a.T)  # 满足约束条件交换后值
            if temp2 < temp1:
                print("第%d与第%d个值交换后更优，原函数最优值更新为:" % (k1 - i - 1, k1 + j), f1(a.T))
            else:
                print("第%d与第%d个值交换后没有更优" % (k1 - i - 1, k1 + j))
                continue  # 交换没有更优则直接下次循环
            if t3 > temp2:
                t1 = k1 - i - 1
                t2 = k1 + j
                t3 = f1(a.T)
                t4 = Pr(a.T)
    print("第%d与第%d个值交换后最优，原函数最优值更新为:" % (t1, t2), t3)
    print("局部优化后P(x>τ)值:", t4)


# 可视化绘图
def visual(x,y):
    # 绘制图形(采用指数分布,横坐标为外部迭代次数，纵坐标为）
    # 创建画布
    fig = plt.figure(num=1, figsize=(12, 6))
    # 字体设置为中文黑体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
    # 设置子图
    plt.subplot(111)
    plt.title("帕累托分布下内点法", fontsize=25, backgroundcolor='#3c7f99',
              fontweight='bold', color='white', verticalalignment="baseline")
    plt.xlabel("外部迭代次数", fontsize=15)
    plt.ylabel("任务中断概率", fontsize=15)
    # plt.semilogy(np.arange(1, x + 1), y1, "r")
    plt.plot(np.arange(1, x), y, "r")
    plt.show()


# 运行主程序
if __name__ == '__main__':
    start = datetime.datetime.now()
    init()
    xe, n = interior_point(x0=np.full((N * M, 1), 0.1), e=1, c=0.01)            # 内点法搜索最优值，返回列向量和外部迭代次数
    if not np.all((Cn.T - np.dot(C1, xe)) > 0):  # 如果数组中的元素全部满足>=0 则返回False，否则返回True
        print("不满足条件1")
    if not np.all((Pm.T - np.dot(C2, xe)) > 0):
        print("不满足条件2")
    print("未整数化时最优解anm =:", xe.T,"其形状:", xe.shape)            # 输出转置后的行向量
    print("小数解之和:",np.sum(xe))
    print("未整数化时原函数最优值:", f1(xe))                             # 原目标函数最小值（小数）
    print("未整数化时公式P(X>τ)最优值:", Pr(xe))                      # 完成时间大于t的概率
    end1 = datetime.datetime.now()
    print('未取整时程序运行时间: %s 秒' % (end1 - start))
    print("传入的x求和值:", int(np.around(np.sum(xe))))
    opt(xe, int(np.around(np.sum(xe))))  # k为x求和后四舍五入取整
    end2 = datetime.datetime.now()
    print('程序运行时间: %s 秒' % (end2 - start).total_seconds())
    duration = 2000         # 持续时间以毫秒为单位，这里是5秒
    freq = 440              # Hz
    winsound.Beep(freq, duration)
    np.savetxt('data/50/IPMA-decimal-obj.txt', obj, fmt="%.4f")
    visual(n,obj)