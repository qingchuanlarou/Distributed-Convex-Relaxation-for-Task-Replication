# Distributed Local Optimal Algorithm
# 每个用户轮循竞争选择概率最大的服务器，选择时同时判断是否满足约束条件
import winsound
import datetime
import scipy.special
import numpy as np
from multiprocessing import Pool
from matplotlib import pyplot as plt
np.set_printoptions(precision = 8)       # 浮点数组输出的精度位数，即小数点后位数(默认为8)。
np.set_printoptions(threshold = 1000000) # 元素门槛值。数组个数沒有超过设置的阈值，NumPy就会将Python將所有元素列印出來。

N = 40        # N个汽车，M个服务器，
M = 10
Cn = np.zeros((N,1))                     # 每个子任务的总成本限制，
Pm = np.zeros((M,1))
x = np.full((N * M, 1), 0)
A1 = np.zeros((M, N*M))                  # 不等式2的系数矩阵，M行（N*M）列

wt = 0.5                                                      # 公共延迟分量
tau = 1                                                      # 期望阈值为2至少大于公共延迟分量

# 帕累托分布                                       # 服务器m最大能并行执行的任务副本数
np.random.seed(1)                                            # 使用相同的seed()值，保证同一程序每次运行结果一样
r = np.random.randint(0,101,size=(1, N*M))                   # 在[0,N*M)中随机选取
lam = np.array([0.5+0.03*i for i in r])                     # xm和lam对应帕累托分布的两个参数
Fx = np.array(1 - np.power(wt/tau,lam)).reshape((1,N*M))        # Ynm的累积分布函数，Pr(Ynm<=dt)的取值
print("随机数r为:",r)
print("帕累托分布参数lam取值：", np.array([0.5+0.03*i for i in r]))
print("帕累托分布参数lam增序取值：", np.sort(lam,axis=None))
print("Pr(Ynm<=dt)取值:",Fx)
print("Pr(Ynm<=dt)增序取值:",np.sort(Fx,axis=None))            # 帕累托分布


def init():
    for i in range(M):
        for j in range(N):
            # Cnm[i][j] = 1  # 单个副本均为1
            A1[i][i + j * M] = 1    # 使非N所在系数为0

    np.random.seed(1)
    for i in range(N):
        Cn[i, 0] = np.random.randint(5, 20)  # 随机初始化每个子任务n的成本
    np.random.seed(1)
    for j in range(M):
        Pm[j,0] = np.random.randint(6,12)                # 随机初始化每个服务器的处理器数

# 求原函数值(我们将凹函数取反求最小化）
def f(x):
    a = x.T.copy()     # 行向量
    temp1 = 0
    for i in range(N):
        an = a[:,i*M:M+i*M]
        Fxn = Fx[:,i*M:M+i*M]
        # print("anm1和FX1的形状:",anm1.shape,Fx1.shape)
        t = np.array(1 - Fxn) ** an
        temp1 += np.log(1 - np.prod(t))  # *是对应元素相乘,dot是矩阵相乘,prod将所有元素相乘
    return -temp1


def Pr(x):
    p = 1 - np.exp(-f(x))
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


def DLO(x):   # 取值算法
    # 每个用户轮循选择满足资源的任务副本个数
    arr = np.arange(N)
    np.random.seed(1)
    k = 0
    while True:
        Flag = True
        np.random.shuffle(arr)
        for n in range(N):    # 每个用户轮循选择概率最大的服务器,选择时要判断是否满足资源约束
            am = np.array([Fx[0,int(arr[n])*M+m] for m in range(M)])  # 取am的所有元素
            arg = np.argsort(am)           # 由小到大排序后的索引顺序[M]
            #print(arg.shape)
            for i in range(M):  # 前Cn个anm赋值为1
                if x[arr[n]*M+arg[i],0] == 1:
                    continue
                elif (Pm[arg[i],0] -np.dot(A1[arg[i],:], x)) > 0:
                    x[arr[n]*M+arg[i], 0] = 1
                    k+=1
                    Flag = False
                    break
                else:
                    continue
        if k == min(np.sum(Cn), np.sum(Pm)) or Flag:
            break


# def MC(x):   # 满足成本约束，Meet the constraint
#     for n in range(N): # 将超出成本约束的子任务删除概率小的服务器连接
#         #print(np.dot(A1[n,:], x).shape)
#         if Cn[n, 0]>=M or (np.dot(A1[n,:], x) - Cn[n,0]) <= 0:  # 当前子任务满足成本约束则进入下一轮循环
#             continue
#         else:
#             arg = np.argsort(Fx[:, n * M:M + n * M])  # [1,M],由小到大排列，arg值为索引
#             #print(arg.shape)
#             k = 0
#             # 超出多少预算就执行多少次，每次找到选中的副本中完成概率最低的赋值为1
#             while k < (np.dot(A1[n,:], x) - int(Cn[n, 0])):
#                 for j in range(M):
#                     if x[n*M+arg[0,j],0] ==1:
#                         x[n * M + arg[0, j], 0]=0
#                         k+=1
#                         break
#                     else:continue


if __name__ == "__main__":
    start = datetime.datetime.now()
    init()
    print("Cn为:",Cn.T)
    print("Pm为:",Pm.T)
    DLO(x)
    end = datetime.datetime.now()
    print("最优解x:", x.T)
    print("最优解之和:", np.sum(x))
    print("公式P(X>τ)最优值:", Pr(x))
    print('程序运行时间: %s 秒' % (end - start).total_seconds())
    variance(x)      # 输出方差
    duration = 2000  # 持续时间以毫秒为单位，这里是5秒
    freq = 440  # Hz
    winsound.Beep(freq, duration)
