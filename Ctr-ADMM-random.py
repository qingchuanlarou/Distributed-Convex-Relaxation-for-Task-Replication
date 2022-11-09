# 将函数改为不分块形式，均能达到收敛,用梯度下降不行还是要用内点法
# 约束为Ax+Bz=c  B取单位方阵，g(z)为指示函数，定义域0<=z<=c1(c1见181行)
import numpy as np
import scipy.special
from matplotlib import pyplot as plt
import datetime
np.set_printoptions(precision = 8)       # 浮点数组输出的精度位数，即小数点后位数(默认为8)。
np.set_printoptions(threshold = 1000000) # 元素门槛值。数组个数沒有超过设置的阈值，NumPy就会将Python將所有元素列印出來。


# 最小化凸函数，即对凹函数取负
# np.seterr(divide='ignore',invalid='ignore')  忽略所有警告
# Ynm的累计分布函数服从指数分布
# Ax+z=c
N = 40        # N个汽车，M个服务器，
M = 10

Cnm = np.zeros((N, M))  # 单个副本的成本，均为1
A1 = np.zeros((N, N*M))                  # 不等式1的系数矩阵，N行（N*M）列
A2 = np.zeros((M, N*M))                  # 不等式2的系数矩阵，M行（N*M)列
A3 = np.zeros((2*N*M,N*M))               # 不等死3的系数矩阵，2*N*M行（N*M)列
A = np.zeros((N+M+2*N*M,N*M))
Cn = np.zeros((N,1))                     # 每个子任务的总成本限制，
Pm = np.zeros((M,1))                     # 直接初始化每个服务器上处理器数量限制，对应运行副本数(变为二维数组才能转置）
C3 = np.zeros((2*N*M,1))                 # 约束3的右边项
c = np.zeros((N+M+2*N*M,1))              # 所有约束的右边项
z = np.zeros((N+M+2*N*M,1))              # 第二个变量

y = np.full((N+M+2*N*M,1),0.01)              # 拉格朗日乘子
step = 1             # 对偶变量更新使用步长
rho = 1            # 二次罚项的系数(很多时候步长step步长取为1）
u = y / rho        # 标准对偶变量 u=y/rho
x = np.full((N * M, 1), 0.1)
obj = []
res = []
to = []  # 选择迭代次数的运行时间和目标函数值

# 随机初始化Fx=Pr(Ynm<=dt)值

# # 随机分布
np.random.seed(1)                                           # 使用相同的seed()值，保证同一程序每次运行结果一样
# r = np.random.choice(N*M,size=N*M,replace=False,p=None)   # 在[0,N*M)中随机选取
r = np.random.randint(50,100,size=(1, N*M))/100
Fx = np.array(r).reshape((1,N*M))                           # Ynm的分布函数，Pr(Ynm<=dt)的取值
print("随机数r为:",r)
print("Pr(Ynm<=dt)取值:", Fx)
print("Pr(Ynm<=dt)增序取值:", np.sort(Fx, axis=None))       # 随机分布


# 初始化赋值并统一约束为Ax+z=C
def init():
    for i in range(N):
        for j in range(M):
            Cnm[i][j] = 1                                 # 单个副本均为1
            A1[i][j + i * M] = Cnm[i][j]                  # 使非N所在系数为0
    for i in range(M):
        for j in range(N):
            A2[i][i + j * M] = 1                          # 使非M所在系数为0
    for i in range(2*N*M):
        if i < N*M:
            A3[i][i] = 1
        else:
            A3[i][i-N*M] = -1
    global A
    A = np.vstack((np.vstack((A1,A2)),A3))

    np.random.seed(1)
    for i in range(N):
        Cn[i, 0] = np.random.randint(5, 20)  # 随机初始化每个子任务n的成本预算为[5,20)
    np.random.seed(1)
    for j in range(M):
        Pm[j,0] = np.random.randint(14,20)                # 随机初始化每个服务器的处理器数

    for i in range(N*M):      # anm<1,C3
         # 剩下的-anm<0
        C3[i][0] = 1
    global c
    c = np.vstack((np.vstack((Cn,Pm)),C3))

    for i in range(N+M+2*N*M):
        if i < N:
            z[i,0] = Cn[i,0]-A1[i,:]@x
        elif i < (N+M):
            z[i,0] = Pm[i-N,0]-A2[i-N,:]@x
        elif i < (N+M+N*M):
            z[i,0] = 1 - x[i-N-M,0]
        else:
            z[i,0] = 0 + x[i-N-M-N*M,0]


# 求原函数值(我们将凹函数取反求最小化）
def f(xk):        # 输入为列向量
    anm = xk.T.copy()
    temp1 = 0
    for i in range(N):
        anm1 = anm[:,i*M:M+i*M]
        Fx1 = Fx[:,i*M:M+i*M]
        # print("anm1和FX1的形状:",anm1.shape,Fx1.shape)
        a = np.array(1 - Fx1) ** anm1
        temp1 += np.log(1 - np.prod(a))  # *是对应元素相乘,dot是矩阵相乘,prod将所有元素相乘
    return -temp1


def Pr(xk):
    p = 1 - np.exp(-f(xk))
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


def hx(xk,zk,uk):                     # 拉格朗日函数对x求值  fx+二次惩罚项
    fx = 0
    for i in range(N):
        xi = xk[i*M:M+i*M,:]
        Fxi = Fx[:,i*M:M+i*M]
        # print("anm1和FX1的形状:",anm1.shape,Fx1.shape)
        fx -= np.log(1-np.prod(np.array(1 - Fxi)**xi.T))   # *是对应元素相乘,dot是矩阵相乘,prod将所有元素相乘
    ptx = ((A@xk+zk-c+uk) ** 2).sum() * rho / 2
    return fx+ptx


def g(zk1):                      # 加一个指数函数，取值为0或正无穷
    temp2 = float("inf")
    c1 = c.copy()     # 完全复制，另外生成
    for i in range(N*M):
        c1[N+M+N*M+i,0] = 1
    if np.all(zk1 >= 0) and np.all(zk1 <= c1):
        return 0
    else:
        return temp2


def ptz(xk1,zk1,uk,r):                   # penalty term二次惩罚项对Z的函数
    return ((A@xk1+zk1-c+uk)**2).sum()*rho/2


def grad_xk(xk,zk,uk):           # 输入x(k)为列向量,求其梯度向量
    gd1 = np.zeros((N*M,1))
    for j in range(N*M):  # 对当前块中的每个元素求梯度,别忘了加负号！
        n = j // M  # 确定j在数组哪一行,即哪个子任务(求商)
        xi = xk[n * M:M + n * M, :].T  # 转置为横向量
        Fxi = Fx[:, n * M:M + n * M]
        gd1[j,0] = np.prod(np.array(1 - Fxi) ** xi) * np.log(1 - Fx[0, j]) / (1 - np.prod(np.array(1 - Fxi) ** xi))  # 原函数梯度
        # 注意anm[0,i]而不是anm[i],这是1行50列数组，而不是一维数组！
    gd2 = rho*np.dot(A.T,(A@xk+zk-c+uk))  # 返回（N*M，1）列向量,,admm文章P15
    gd_x = gd1+gd2
    #print("A.T:",A.T)
    #print("A@xk+zk-c+uk:",A@xk+zk-c+uk)
    #print("x的梯度第一部分为:",gd1.T)
    #print("x的梯度第二部分为",gd2.T)
    #print("x的下降方向:",-gd_x.T)
    return gd_x            # 返回(N*M,1)列向量


def backtracking(x0, delta,z,u,c):
    t = 1
    while True:
        x1 = x0 + t * delta
        if hx(x1,z,u) <= (hx(x0,z,u) + c * t * grad_xk(x0,z,u).T @ delta):  # c为α，取为0.01，x1-x0=t*delta,@为矩阵乘法
            #print("回溯搜索步长为:",t)
            break
        else:
            t = t*0.5                # β取为0.5
    return x1,t                   # 返回更新后x值和步长t


def update_x(xk,zk,uk):              # 输入x,z,u均为列向量
    # print("x%d更新前值为:"%i, xk[i * M:M + i * M, :])
    Over = False
    num1 = 1
    while not Over:
        delta = - grad_xk(xk,zk,uk)  # @为矩阵乘法，linalg.inv为矩阵求逆
        # print("第%d次内部迭代:" % num1)
        xk1, t1 = backtracking(xk, delta,zk,uk,c=0.01)  # 返回列向量
        l2 =  np.linalg.norm(grad_xk(xk,zk,uk))      # 求梯度的二次范数
        #print("停止准则量:", l2)
        if l2 < 1e-3:  # λ的平方     不采用牛顿减量,精度问题,回溯法中步径t很小时t*delta==0,x0+t*delta=x0,陷入死循环
            print("本轮内部迭代结束，找到x(e)")
            Over = True
        else:
            xk = xk1
            #print("更新后x值为:", xk)
        num1 += 1
    return xk
    #print("内部迭代结束，x更新值为:", xk)
    #print("梯度下降法法搜索后x更新值为:", xk)


def update_z(xk1,c,uk,k):   # 求导为0时z值为极值(必须在不等式约束内)
        #print("第%d次迭代z值:"%k,z)
        zk1 =c-uk-A@xk1
        c1 = c.copy()  # 完全复制，另外生成
        for i in range(N * M):
            c1[N + M + N * M + i, 0] = 1
        if np.all(zk1 >= 0) and np.all(zk1 <= c1):
            print("极值点z*在可行域内:")
        else:
            for j in range(N+M+2*N*M):
                if 0<=zk1[j,0]<=c1[j,0]:
                    continue
                elif zk1[j,0] > c1[j,0]:
                    zk1[j,0] = c1[j,0]
                else:
                    zk1[j, 0] = 0
            # print("极值点z*在可行域外,已修改")
            # print(zk1)
            # quit()
        return zk1


def update_u(uk,xk1,zk1,c):        # 更新标准对偶变量u
    #print("A@xk1+zk1-c为",A@xk1+zk1-c)
    #print("u",uk)
    uk1 = uk+A@xk1+zk1-c
    return uk1


# 局部优化交换法(按值的降序选择未选择的anm,并按值的升序选择所选的anm)
# 最终版 将剩余资源服务器和超出资源服务器进行交换
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
    if np.all((Cn - np.dot(A1, xe.T)) >= 0) and np.all((Pm - np.dot(A2, xe.T)) >= 0):  # 当原始整数解满足条件时
        print("满足约束条件")
        for i in range(k1):          # 升序选择所选的anm
            for j in range(N*M-k1):  # 降序选择未选择的anm
                a = xe.copy()
                temp1 = f(a.T)
                if i==0 and j==0:
                    t3=temp1         # 赋初值为原始解
                    t4 = Pr(a.T)
                a[0, arg[k1+j]], a[0, arg[k1-i-1]] = a[0, arg[k1-i-1]], a[0, arg[k1+j]]  # 交换两个anm值
                # print("arg中第%d与第%d个值交换，原函数最优值为:" % (k1 - i - 1, k1 + j), f1(a.T))
                # 判断是否满足约束条件,不满足则直接下一轮循环
                if not np.all((Cn - np.dot(A1, a.T)) >= 0):
                    # print("arg中第%d与第%d个值交换后不满足条件1" % (k1 - i - 1, k1 + j))
                    # print(Cn - np.dot(A1, a.T))
                    continue
                if not np.all((Pm - np.dot(A2, a.T)) >= 0):
                    # print("arg中第%d与第%d个值交换后不满足条件2" % (k1 - i - 1, k1 + j))
                    # print(Pm - np.dot(A2, a.T))
                    continue
                temp2 = f(a.T)       # 满足约束条件交换后值
                if temp2 < temp1:    # 判断是否小于原函数值
                    pass
                    # print("arg中第%d与第%d个值交换后更优，原函数最优值更新为:" % (k1 - i - 1, k1 + j), f(a.T))
                else:
                    # print("arg中第%d与第%d个值交换后没有更优" % (k1 - i - 1, k1 + j))
                    continue         # 交换没有更优则直接下次循环
                if temp2 < t3:       # 判断是否小于目前最小值
                    t1 = k1 - i - 1
                    t2 = k1 + j
                    t3 = f(a.T)
                    t4 = Pr(a.T)
        xe = np.full((1, N * M), 0)
        for m in range(k1):  # 求最优解
            xe[0, arg[m]] = 1  # 每次重置为原始解
        xe[0, arg[t2]], xe[0, arg[t1]] = xe[0, arg[t1]], xe[0, arg[t2]]  # 交换两个anm值
        print("最终x的整数解和为:", k1)
        print("arg中第%d与第%d个值交换后最优，原函数最优值更新为:" % (t1, t2), t3)
        print("公式P(X>τ)最优值更新为", t4)
    else:   # 当原始整数解不满足条件时
        # l1 = Cn - np.dot(A1, xe.T)  # [N,1]   # 不等式1约束右边减左边
        l2 = Pm - np.dot(A2, xe.T)  # [M,1]   # 不等式2约束右边减左边
        if not np.all((Cn - np.dot(A1, xe.T)) >= 0):  # 如果数组中的元素全部满足>=0 则返回False，否则返回True
            print("不满足条件1,Cn-A1@x.T=", (Cn - np.dot(A1, xe.T)).T)
        if not np.all((Pm - np.dot(A2, xe.T)) >= 0):
            print("不满足条件2,Pm-A2@x.T=", (Pm - np.dot(A2, xe.T)).T)
        t1 = [i for i in range(M) if l2[i, 0] < 0]  # 求出为-1的索引号(服务器超出最大资源),t1和t2的长度相同
        t4 = [l2[i, 0] for i in range(M) if l2[i, 0] < 0]  # 求出小于于0的服务器对应超出资源数
        t2 = [i for i in range(M) if l2[i, 0] > 0]  # 求出大于0的索引号(服务器资源没用完)
        t3 = [l2[i, 0] for i in range(M) if l2[i, 0] > 0]  # 求出大于0的服务器对应剩余资源
        print("循环交换次数:",int(-sum(t4)))
        for i in range(int(-sum(t4))):         # 循环交换使所以服务器满足资源预算
            for n in range(k1):  # 升序将满足条件的anm变为0
                n1 = arg[k1-n-1] // M         # 确定当前元素属于哪个子任务(求商)
                m1 = arg[k1-n-1] % M            # 确定当前元素属于哪个服务器(求余)
                if xe[0,arg[k1-n-1]] == 0:      # 已经变为0则直接跳过
                    continue
                if m1 == t1[0] and np.dot(A1, xe.T)[n1,0] > 1:  # 取某个元素，所属服务器资源超出，所属任务副本数大于1(至少为2)
                    xe[0,arg[k1-n-1]] = 0
                    # print("当前赋0结束,第%d个服务器满足约束条件" % t1[0])
                    t4[0] += 1     # 注意这里不是-=
                    if t4[0] == 0:  # 当前还有空闲服务器资源用完
                        t1.pop(0)  # 弹出当前服务器号
                        t4.pop(0)  # 弹出当前服务器
                    break
            l1t = Cn - np.dot(A1, xe.T)  # 每次循环更新一下信息
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
        print("最终x的整数解和为:", k1)
        print("约束条件1,Cn-A1@x.T=", (Cn - np.dot(A1, xe.T)).T)
        print("约束条件2,Pm-A2@x.T=", (Pm - np.dot(A2, xe.T)).T)
        print("公式P(X>τ)最优值更新为", Pr(xe.T))
    variance(xe.T)

# 可视化绘图
def visual(x,y1,y2):
    # 绘制图形(采用指数分布,横坐标为外部迭代次数，纵坐标为）
    # 创建画布
    fig = plt.figure(num=1, figsize=(12, 6))
    # 字体设置为中文黑体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
    # 设置子图
    plt.subplot(121)
    plt.xlabel("外部迭代次数", fontsize=15)
    plt.ylabel("原始残差", fontsize=15)
    plt.title("Ctr-ADMM", fontsize=25, backgroundcolor='#3c7f99',
                  fontweight='bold', color='white', verticalalignment="baseline")
    plt.plot(np.arange(1, x + 1), y1, "r")
    # plt.semilogy(np.arange(1, x + 1), y1, "r")
    plt.subplot(122)
    plt.xlabel("外部迭代次数", fontsize=15)
    plt.ylabel("目标函数值", fontsize=15)
    plt.title("Ctr-ADMM", fontsize=25, backgroundcolor='#3c7f99',
                  fontweight='bold', color='white', verticalalignment="baseline")
    # plt.plot(np.arange(1, x + 1), y2, "r")
    plt.semilogy(np.arange(1, x + 1), y2, "r")
    plt.show()


if __name__ == "__main__":
    # 程序计时器，启动计时器
    start = datetime.datetime.now()
    init()
    print("每个子任务总成本限制：", Cn.T, "其形状", Cn.shape)
    print("每个服务器上处理器数量：", Pm.T, "其形状", Pm.shape)
    #if np.all(np.linalg.eigvalsh(hessian(np.full((N * M, 1), 0.5))) >= 0):
        #print("海森矩阵是半正定的，其特征值为：",np.linalg.eigvalsh(hessian(np.full((N * M, 1), 0.5))))
    #else:
        #print("初始海森不是半正定的啦！其特征值为：", np.linalg.eigvalsh(hessian(np.full((N * M, 1), 0.5))))
    print("x初始值:",x.T)
    print("z初始值:",z.T)
    print("u初始值:",u.T)
    runnum = 0    # 实际运行次数
    residual = 1
    k = 1         # 迭代次数
    while True:
        print("第%d次迭代"%k)
        x = update_x(x,z,u)  # 每次传入的x都是没更新的,x,z,u均为列向量
        s = rho*A.T@(update_z(x,c,u,k)-z)   # 对偶残差
        z = update_z(x,c,u,k)
        #print("z第%d次迭代后值:" % k, z)
        u = update_u(u,x,z,c)
        #print("u第%d次迭代后值:" % k, u)
        obj.append(Pr(x))
        residual = np.linalg.norm(A @ x + z - c, keepdims=True)
        print("原始残差为:", np.linalg.norm(A @ x + z - c, keepdims=True))
        res.append(np.linalg.norm(A @ x + z - c, keepdims=True)[0,0])
        if np.linalg.norm(A @ x + z - c) > 10 * np.linalg.norm(s):  # 使残差大小保持在一个范围
            rho = 2 * rho
        elif np.linalg.norm(s) > 10 * np.linalg.norm(A @ x + z - c):
            rho = rho / 2
        runnum = k  # 实际运行次数
        k += 1
        if residual<=0.0005 and (not np.all((c - np.dot(A, x)) >= 0)):
            print("最优解x:", x.T)
            print("达到停止准则,迭代次数%d,但x不满足约束条件:" %k)
            print("原始残差为:", np.linalg.norm(A @ x + z - c, keepdims=True))
            print("对偶残差为:", np.linalg.norm(s, keepdims=True))
            break
        elif residual<=0.0005 and np.all((c - np.dot(A, x)) >= 0):
            print("最优解x:", x.T)
            print("达到停止准则,迭代次数%d,但满足约束条件:" %k)
            print("原始残差为:", np.linalg.norm(A @ x + z - c, keepdims=True))
            print("对偶残差为:", np.linalg.norm(s, keepdims=True))
            break
    print("未整数化时原函数最优值:", f(x))
    print("小数解之和:", np.sum(x))
    print("未整数化时公式P(X>τ)最优值:", Pr(x))
    end1 = datetime.datetime.now()
    print('未取整时程序运行时间: %s 秒' % (end1 - start).total_seconds())
    # print("z值:",z.T)
    y = u*rho
    # print("y值:",y.T)
    print("传入的x求和值:", int(np.around(np.sum(x))))
    opt(x, int(np.around(np.sum(x))))  # k为x求和后四舍五入取整
    end2 = datetime.datetime.now()
    print('程序运行时间: %s 秒' % (end2 - start).total_seconds())
    np.savetxt('data/1.4/CtrADMM-res.txt', res)
    np.savetxt('data/1.4/CtrADMM-obj.txt', obj, fmt="%.4f")
    visual(runnum, res, obj)