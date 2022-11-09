# 指数分布，JP-admm，通过分布式计算实现，约束1加在每一块中用内点法来求，约束2变为等式
# 取值部分
# 约束为Ax=c   x属于[0,1]
# xk:第k次即当前迭代的x值，xki:对应的第i块值 xk1:xk的下一次迭代 xk1i:xki的下一次迭代
import winsound
import datetime
import numpy as np
import scipy.special
from multiprocessing import Pool
from matplotlib import pyplot as plt
np.set_printoptions(precision = 8)       # 浮点数组输出的精度位数，即小数点后位数(默认为8)。
np.set_printoptions(threshold = 1000000) # 元素门槛值。数组个数沒有超过设置的阈值，NumPy就会将Python將所有元素列印出來。

# 最小化凸函数，即对凹函数取负
# np.seterr(divide='ignore',invalid='ignore')  忽略所有警告
# e是障碍法中的参数1/t，c为回溯直线搜索中α，β取为0.5
# Ynm的累计分布函数服从指数分布
N = 60        # N个汽车，M个服务器，
M = 10

Cnm = np.zeros((N, M))                   # 单个副本的成本，均为1
A1 = np.zeros((N, N*M))                  # 不等式1的系数矩阵，N行（N*M）列
A2 = np.zeros((M, N*M))                  # 等式2的系数矩阵，M行（N*M)列
A = np.zeros((M,N*M))
Cn = np.zeros((N,1))                     # 每个子任务的总成本限制，
c = np.zeros((M,1))

y = np.full((M,1),0.01)              # 拉格朗日乘子
rho = 0.001           # 对偶变量更新使用步长(也是更新步长）
step = 1            # 步长
k = 1               # 迭代次数
u = y / rho         # 标准对偶变量 u=y/rho
x = np.full((N * M, 1), 0.1)
res = []            # 所有的原始残差
obj = []                                # 存储迭代过程小数解
to = []             # 选择迭代次数的运行时间和目标函数值
tao = 1.1*rho
tol = 1e-3
# 13为Ai.T@Ai矩阵的最大特征值 ，要使τI-rho*Ai.T@Ai为半正定矩阵,τI-rho*(N+1)*Ai.T@Ai/(2-gamma)为正定矩阵，所以取13.1则正定
Eta = 1e-9                               # 动态更新tao的判断参数
alpha = 2

tau = 1                                  # 期望阈值为1,去掉了公共延迟分量,这是总的时间，传输时间其实更小
# 随机初始化Fx=Pr(Ynm<=tau)值
# 指数分布
np.random.seed(1)                             # 使用相同的seed()值，保证同一程序每次运行结果一样(设置的seed()值仅一次有效)
B = 1e6                                       # 用户与服务器之间带宽均设置为1Mhz，10的6次方
gamma = np.random.uniform(1,30,size = N*M).reshape((N,M))          # 信噪比取值为[1,30),[1,N*M]
lambert = np.real(scipy.special.lambertw(gamma, k=0, tol=1e-8))    # 信噪比的朗伯w函数,np.real()取实部
r = B*lambert/np.log(2)                       # 传输速率约为几Mb/s
np.random.seed(1)
bn = np.random.randint(4e5,1e6,size = N)      # 任务传输大小0.4Mb-1Mb
bnm = np.zeros((N,M))                         # 方便计算将任务传输大小扩展为N*M维
for i in range(N):
    for j in range(M):
        bnm[i,j] = bn[i]
w = r/bnm * np.exp(-1/gamma*(2**(r/B)-1))     # 服务速率w表达公式
np.random.seed(1)
dn = np.random.uniform(0.1,0.8,size = N)      # 任务所需要计算量，0.1G-0.8G个CPU周期
dnm = np.zeros((N,M))
for i in range(N):
    for j in range(M):
        dnm[i,j] = dn[i]
np.random.seed(1)
Fm = np.random.randint(10,16,size =M)         # 随机初始化服务器计算能力,多核总和(GHz)
fm = 1                                        # 服务器m给单个给单个任务分配的频率，这里假设所有服务器相同,1GHZ
Pm = (Fm/fm).reshape((M,1))                   # 服务器m最大能并行执行的任务副本数
Fx = np.array(1 - np.exp(-1 * w * (tau-dnm/fm))).reshape((1,N*M))  # Ynm的累积分布函数，Pr(Ynm<=dt)的取值
# tau-dnm/fm必须要保证其值大于0，否则FX变为复数
print("信噪比为为:",gamma)
print("朗伯w函数为：", lambert)
print("传输速率速率为：", r)
print("任务传输数据大小(bits):",bnm)
print("服务速率:",w)
print("服务速率w增序取值：", np.sort(w,axis=None))
print("任务执行所需要周期:",dnm)  # 指数分布
print("服务器数量:",Pm,np.sum(Pm))
print("dnm/fm增序取值:",np.sort(dnm/fm,axis=None))
print("任务完成概率Pr(Ynm<=dt):", Fx)
print("任务完成概率Pr(Ynm<=dt)增序取值:", np.sort(Fx,axis=None))


def init():
    for i in range(N):
        for j in range(M):
            Cnm[i][j] = 1  # 单个副本均为1
            A1[i][j + i * M] = Cnm[i][j]  # 使非N所在系数为0
    for i in range(M):
        for j in range(N):
            A2[i][i + j * M] = 1  # 使非M所在系数为0
    global A
    A = A2

    np.random.seed(1)
    for i in range(N):
        Cn[i, 0] = np.random.randint(5,20)  # 随机初始化每个子任务n的成本
    global c
    c = Pm


# 求原函数值(我们将凹函数取反求最小化）
def f(xk):
    anm = xk.T.copy()     # 行向量
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


def pr_task(x,i):        # 计算任务i完成概率Pr(Xn<=dt)
    anm = x.T.copy()     # 行向量
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


def hxi(i, xki, xk, uk, e):                     # 拉格朗日函数对xi求值  fxi+二次惩罚项+对称障碍函数
    Fxi = Fx[:,i*M:M+i*M]
    fxi = -np.log(1-np.prod(np.array(1 - Fxi)**xki.T))  # *是对应元素相乘,dot是矩阵相乘,prod将所有元素相乘
    fai=-np.log(Cn[i,0]-A1[i,i * M:M + i * M]@xki)-np.sum(np.log(xki) + np.log(1 - xki))          # 对称障碍函数
    Ai = A[:, i * M:M + i * M]
    ptxi = rho*(Ai.T@(A@xk-c+uk)).T@xki              # (1,M)@(M,1)
    prx = ((xki - xk[i * M:M + i * M, :]) ** 2).sum() * tao / 2
    return fxi+e*fai+ptxi+prx


def gradient(i,xki,xk,uk,e):           # 输入x(k)i为列向量,求其梯度向量
    xi = xki.T              # 转置为横向量
    Fxi = Fx[:, i * M:M + i * M]
    gd1 = np.zeros((M,1))      # 原函数
    for j in range(M):  # 对当前块中的每个元素求梯度,别忘了加负号！
        gd1[j,0] = np.prod(np.array(1 - Fxi) ** xi) * np.log(1 - Fxi[0, j]) / (1 - np.prod(np.array(1 - Fxi) ** xi))  # 原函数梯度
        # 注意anm[0,i]而不是anm[i],这是1行50列数组，而不是一维数组！
    Ai = A[:, i * M:M + i * M]
    gd2 = rho * (Ai.T @ (A @ xk - c + uk))  # (M,1) 第二项线性函数梯度
    gd3 = tao * (xki - xk[i * M:M + i * M, :])   # (M,1) 第三项函数
    gd4 = np.zeros((M, 1))                       # 0<=x<=1约束加在目标函数上
    for k in range(M):
        gd4[k,0]=e*1/(Cn[i,0]-A1[i,i * M:M + i * M]@xki)-e / xki[k, 0] + e / (1 - xki[k, 0])
    gd_xi = gd1+gd2+gd3+gd4
    return gd_xi            # 返回(M,1)列向量


def hessianx(i,xki,e):
    x1 = xki.T  # 转为行向量（1，M）
    H1 = np.zeros((M, M))
    Fx1 = Fx[:, i * M:M + i * M]  # 取出子任务n1与M个服务器相连所用时间分布
    for k in range(M):  # 此处不能用i，换成j
        for j in range(M):
            temp1 = -np.prod(np.array(1 - Fx1) ** x1) * np.log(1 - Fx1[0, k]) * np.log(1 - Fx1[0, j]) * (
                    1 - np.prod(np.array(1 - Fx1) ** x1))
            temp2 = -np.power(np.prod(np.array(1 - Fx1) ** x1), 2) * np.log(1 - Fx1[0, k]) * np.log(1 - Fx1[0, j])
            temp3 = np.power((1 - np.prod(np.array(1 - Fx1) ** x1)), 2)
            t1 = (temp1 + temp2) / temp3
            if k == j:  # 计算不等式3的二次导
                t2 = 1 / np.power(x1[0, k], 2) + 1 / np.power((1 - x1[0, k]), 2)
            else:
                t2 = 0
            t3 = 1 / np.power((Cn[i,0] - A1[i, i * M:M + i * M] @ xki), 2)
            H1[k, j] = -t1 + e * (t2+t3)
    H2 = tao*np.eye(M)   # [M,M]
    H = H1+H2            # [M,M]
    # print(H,H.shape)
    return H


def backtracking1(xki,xk,delta,uk,e,c):
    t = 1
    while True:
        xk1i = xki + t * delta
        if hxi(i,xk1i,xk,uk,e)<=(hxi(i,xki,xk,uk,e)+ c * t * gradient(i, xki, xk, uk, e).T @ delta):
            # c为α，取为0.01，x1-x0=t*delta,@为矩阵乘法
            #print("回溯搜索步长为:",t)
            break
        else:
            t = t*0.5                # β取为0.2
    return xk1i,t                   # 返回更新后x值和步长t


def newton1(i,xki,xk,uk,e,c,tol=1e-6):
    assert (0 < c < 0.5)  # 判断一个表达式，在表达式条件为 false 的时候触发异常
    num1 = 1  # Newton迭代
    while True:
        delta = - np.linalg.inv(hessianx(i,xki,e)).dot(gradient(i, xki, xk, uk, e))  # @为矩阵乘法，linalg.inv为矩阵求
        x1, t1 = backtracking1(xki,xk,delta,uk,e,c)  # 返回列向量
        l2 = gradient(i, xki, xk, uk, e).T @ np.linalg.inv(hessianx(i, xki, e)) @ gradient(i, xki, xk, uk, e)
        # print("停止准则量:", l2[0, 0] / 2)
        if l2[0, 0] / 2 < tol:  # λ的平方     不采用牛顿减量,精度问题,回溯法中步径t很小时t*delta==0,x0+t*delta=x0,陷入死循环
            # print("本轮内部迭代结束，找到x(e)")
            break
        else:
            xki = x1
        num1 += 1
    return xki


def update_x(i,xk,uk,xt,tol=1e-6):              # 输入x,u均为列向量
    # print("x%d更新前值为:"%i, xk[i * M:M + i * M, :])
    e = 1
    c = 0.1
    print("使用内点法求解第%d块" % i)
    xki = xk[i * M:M + i * M, :]
    while True:
        x2 = newton1(i,xki,xk,uk,e,c)  # 返回列向量
        if 2 * M * e <= tol:  # 相当于t>=1/tol
            break
        else:
            for j in range(M):
                xki[j, 0] = x2[j, 0]
            e /= 3  # 相当于t=μt，μ=3
    for j in range(M):
        xt[i * M + j, 0] = x2[j, 0]
    print("更新后x%d为:" % i, xt[i * M:M + i * M, :].T)


def update_u(uk,xk1,c):        # 更新标准对偶变量u
    uk1 = uk+A@xk1-c
    return uk1


def apt_tao(x,xt,u,ut):   # Adaptive Parameter Tuning
    h1 =((x - xt) ** 2).sum() * tao+((u - ut) ** 2).sum() * 1/rho+2*(u - ut).T@A@(x - xt)
    h2 = Eta*(((x - xt) ** 2).sum()+((u - ut) ** 2).sum())
    if h1 > h2:
        return True
    else:
        return False


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
                    print("arg中第%d与第%d个值交换后更优，原函数最优值更新为:" % (k1 - i - 1, k1 + j), f(a.T))
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
    plt.title("DAA形式", fontsize=25, backgroundcolor='#3c7f99',
              fontweight='bold', color='white', verticalalignment="baseline")
    # plt.semilogy(np.arange(1, x + 1), y1, "r")
    plt.plot(np.arange(1, x + 1), y1, "r")
    # plt.savefig('data/60/DAA0001-res.png')
    plt.subplot(122)
    plt.xlabel("外部迭代次数", fontsize=15)
    plt.ylabel("目标函数值", fontsize=15)
    plt.title("DAA形式", fontsize=25, backgroundcolor='#3c7f99',
              fontweight='bold', color='white', verticalalignment="baseline")
    plt.plot(np.arange(1, x + 1), y2, "r")
    # plt.savefig('data/60/DAA0001-obj.png')
    plt.show()


if __name__ == "__main__":
    start = datetime.datetime.now()
    tt = 0  # 分布式总共时间初始值，取每次迭代耗时最长块相加
    init()
    print("每个子任务总成本限制：", Cn.T, "其形状", Cn.shape)
    print("每个服务器上处理器数量：", Pm.T, "其形状", Pm.shape)
    print("x初始值:",x.T)
    print("u初始值:",u.T)
    for i in range(N):
        # print(np.eye(M))
        # v=(tao*np.eye(M)-rho*A[:,i*M:M+i*M].T@A[:,i*M:M+i*M])
        v=(tao*np.eye(M)-rho*N*A[:,i*M:M+i*M].T@A[:,i*M:M+i*M])
        v2 = A[:, i * M:M + i * M].T @ A[:, i * M:M + i * M]
        print("最大特征值为:",np.linalg.eigvals(v2))
        B=np.linalg.eigvals(v)
        if np.all(B>=0):
            print("是半正定矩阵")
            print(B)
        else:
            print("不是半正定矩阵")
            print(B)
        print(A[:, i * M:M + i * M].T @ A[:, i * M:M + i * M])
    runnum = 0  # 实际运行次数
    while k <= 100:
        print("第%d次外部迭代:"%k)
        xt = x.copy()                # 临时存储每一块更新值
        timek = np.zeros(N+1)
        for i in range(N):         # 一共分为N块
            ti1 = datetime.datetime.now()
            update_x(i,x,u,xt)   # 每次传入的x都是没更新的
            ti2 = datetime.datetime.now()
            timek[i]=(ti2-ti1).total_seconds()
        tk1 = datetime.datetime.now()
        ut = update_u(u, xt, c)  # 暂时存放uk+1
        # print("u第%d次迭代后值:" % k, ut.T)
        """if np.linalg.norm(xt - x) < tol*np.linalg.norm(x):
            print("达到停止准则")
            res.append(np.linalg.norm(A @ xt - c, keepdims=True)[0, 0])
            runnum = k
            break"""
        if apt_tao(x, xt, u, ut):
            print("更新x,u")
            for i in range(N * M):
                x[i, 0] = xt[i, 0]
            u = ut
        else:
            print("不更新x,u")
            tao = alpha * tao     # 不更新x,u,使用修改后的tao重新计算
        obj.append(Pr(x))
        if not np.all((Cn-np.dot(A1,x)) >= 0):   # 如果数组中的元素全部满足>=0 则返回False，否则返回True
            print("不满足条件1",)
            print((Cn - np.dot(A1, x)))
        if not np.all((Pm-np.dot(A2,x)) >= 0):
            print("不满足条件2",)
            print((Pm-np.dot(A2,x)))
        print("原始残差为:",np.linalg.norm(A@x-c,keepdims=True))
        res.append(np.linalg.norm(A@x-c,keepdims=True)[0,0])
        print("任务中断概率为:", Pr(x))
        runnum = k
        k += 1
        if k == 100 and (not np.all((c - np.dot(A, x)) >= 0)):
            print("达到最大迭代次数但x不满足约束条件")
        elif k ==100 and np.all((c - np.dot(A, x)) >= 0):
            print("达到最大迭代次数并且x满足约束条件")
        tk2 = datetime.datetime.now()
        tt += (tk2-tk1).total_seconds()+np.max(timek)
        if k ==100:
            to.append(tt)
            to.append(Pr(x))
    print("最优解x:",x.T)
    print("未整数化时原函数最优值:", f(x))
    print("小数解之和:", np.sum(x))
    print("未整数化时公式7 P(X>τ)最优值:", Pr(x))
    print("100次迭代后目标函数值:", to[1])
    print("不取整分布式后程序运行时间",tt)
    y = u*rho
    # print("y值:",y.T)
    print("传入的x求和值:", int(np.around(np.sum(x))))
    tk3 = datetime.datetime.now()
    opt(x, int(np.around(np.sum(x))))  # k为x求和后四舍五入取整
    end = datetime.datetime.now()
    tt += (end-tk3).total_seconds()
    print('单机程序运行时间: %s 秒' % (end - start))
    print('分布式后程序运行时间: %s 秒' % tt)
    #print("100次迭代所用时间:", to[0]+(end-tk3).total_seconds())
    duration = 2000         # 持续时间以毫秒为单位，这里是2秒
    freq = 440              # Hz
    winsound.Beep(freq, duration)
    # np.savetxt('data/60/DAA0001-res.txt', res)
    # np.savetxt('data/60/DAA0001-obj.txt', obj, fmt="%.4f")
    visual(runnum,res,obj)
