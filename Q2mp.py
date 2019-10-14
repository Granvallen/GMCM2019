import xlrd
import numpy as np
import math
import sympy
from multiprocessing import Manager, Process
import time
import random

sc = time.time()

dataidx = 2
pnum = 5
maxNse = 30

# excel文件
file = 'dataset{}.xlsx'.format(dataidx)
# 参数
if dataidx == 1:
    # 1
    a1 = 25
    a2 = 15
    b1 = 20
    b2 = 25
    theta = 30
    delta = 0.001
else:
    # 2
    a1 = 20
    a2 = 10
    b1 = 15
    b2 = 20
    theta = 20
    delta = 0.001

# 读取excel
wb = xlrd.open_workbook(filename=file)

# 所有表格的名字
sheetnames = wb.sheet_names()
sheet = wb.sheet_by_name(sheetnames[0])

# 按列获取数据
idx = sheet.col_values(0)[2:]
x = sheet.col_values(1)[2:]
y = sheet.col_values(2)[2:]
z = sheet.col_values(3)[2:]
cat = sheet.col_values(4)[3:-1]
mark = sheet.col_values(5)[3:-1]

# 分离坐标 转为向量
start = np.array([x[0], y[0], z[0]])
end = np.array([x[-1], y[-1], z[-1]])
endi = len(cat)+1 # 终点序号
x, y, z = x[1:-1], y[1:-1], z[1:-1]
vercheck = {}
horcheck = {}
hvmark = {}
for i in range(len(cat)):
    if int(cat[i]) == 1:
        vercheck[i + 1] = np.array([x[i], y[i], z[i]])
    else:
        horcheck[i + 1] = np.array([x[i], y[i], z[i]])

    hvmark[i + 1] = mark[i]


class Navigator:
    def __init__(self):
        # 记录路径
        self.lse = [] # 路线总长度
        self.Nse = [] # 经过校正点个数
        self.Lse = [] # 路径
        self.hvidxse = [] # 记录经过每个校正点时水平与竖直误差 题目要求
        self.drawhelper = [] # 记录下每次转弯时圆心与切点坐标 用于绘制轨迹

        self.terminatepath = set()

    def navigate(self, current, lc, Nc, hvidx, Lc, Eh, Ev, k, vec, dh):
        """
            current: 当前坐标
            lc: 当前为止已行进路径长度
            Nc: 当前为止已经过的校正次数
            Lc: 当前为止的路径
            Eh: 当前为止水平误差
            Ev: 当前为止垂直误差
            k: search factor

            返回在经过当前点时是否成功找到路径
        """
        if Lc == []:
            Lc.append(start)

        dist_end, P0, P2, _ = self.dfun(current, end, vec)
        # 判断是否因为 该路径不可能最优 而中断搜索
        if self.prune(lc+dist_end, Nc):
            print("search terminated")
            return False

        # 这回合到达终点
        error = dist_end*delta
        if Ev + error <= theta and Eh + error <= theta:
            self.lse.append(lc + dist_end)
            self.Nse.append(Nc)
            self.Lse.append(Lc + [end])
            self.hvidxse.append(hvidx + [(endi, Eh + error, Ev + error)])
            self.drawhelper.append(dh + [(current, P2, P0)])
            print("route found {} Nc: {} l: {}".format(len(self.Lse), Nc, self.lse[-1]))
            return True

        # 继续搜索 在问2的假设下不能再采用之前走km的策略了 因为走的是空间弧线 1是不好算 2是不一定是最优的
        # 那么将进入寻找校正点
        flag = False
        if not flag:
            istoend, route, route_hvidx, dist_route, idx, vecnextlist, dhlist, Ehv = self.checkroute(current, Eh, Ev, vec)
            # print(len(route))
            # print(route_hvidx)
            if istoend:
                if self.prune(lc+dist_route, Nc+1):
                    print("search terminated")
                    return False

                self.lse.append(lc + dist_route)
                self.Nse.append(Nc + 1)
                self.Lse.append(Lc + route)
                self.hvidxse.append(hvidx + route_hvidx)
                self.drawhelper.append(dh + dhlist)
                print("route found", len(self.Lse))
                return True
            elif route != []:
                # 随机选择路径 效果确实不好
                # idx = list(range(len(route)))
                # random.shuffle(idx)

                for i in idx[:pnum]:
                    # print(route_hvidx)
                    print([i[0] for i in hvidx+route_hvidx[i]])
                    flag = self.navigate(route[i][1], lc+dist_route[i], Nc+2, hvidx+route_hvidx[i], Lc+list(route[i]), Ehv[i][0], Ehv[i][1], k, vecnextlist[i], dh+dhlist[i])
            else:
                # terminatepath这部分其实在Q2中还是不合理的 因为Q2还有方向问题 换一个到达方向可能这条路就不是死路了
                # Q1中还是能用的
                # if len(hvidx) >= 2:
                #     self.terminatepath |= {(hvidx[-2][0], hvidx[-1][0])}
                return False
        else:
            return True

    # 判断是否需要中断搜索
    def prune(self, lc, Nc):
        if Nc > maxNse:
            return True

        if self.Lse != []:
            return lc >= min(self.lse) or Nc > min(self.Nse) # 这里改成or的话是 寻找校正数与长度都要比已经找到好的路径
        else:
            return False

    # 计算曲线距离
    def dfun(self, From, To, vec1):
        """
            返回路径长度, 圆心坐标, 切点坐标, 到达目的地后的方向向量
        """

        vec2 = To - From

        if type(vec1) != np.ndarray:
            return np.linalg.norm(vec2), None, None, vec2

        crossvec = np.cross(vec1, vec2)

        # 直线 不需要转弯
        if np.linalg.norm(crossvec) == 0:
            # print(np.linalg.norm(crossvec))
            return np.linalg.norm(vec2), None, None, vec1

        # 转弯半径过小 不可达
        if np.linalg.norm(vec2) < 400:
            return None, None, None, None

        # 解出圆心与切点
        x1, y1, z1 = From[0], From[1], From[2]
        x3, y3, z3 = To[0], To[1], To[2]
        xv, yv, zv = vec1[0], vec1[1], vec1[2]
        xd, yd, zd = crossvec[0], crossvec[1], crossvec[2]

        x0, y0, z0 = sympy.symbols("x0 y0 z0")
        xyz0 = sympy.solve([(x1-x0)*xv+(y1-y0)*yv+(z1-z0)*zv, (x1-x0)**2+(y1-y0)**2+(z1-z0)**2-40000,\
             (x0-x1)*xd+(y0-y1)*yd+(z0-z1)*zd], [x0, y0, z0])
        # 圆心有两个解
        for ans in xyz0:
            x0, y0, z0 = float(ans[0]), float(ans[1]), float(ans[2])
            P0 = np.array([x0, y0, z0])
            a = P0-From
            b = To-From
            cosd = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            if cosd >= 0:
                break
        
        # print(P0)

        x, y, z = sympy.symbols("x y z")
        xyz = sympy.solve([(x-x0)*(x-x3)+(y-y0)*(y-y3)+(z-z0)*(z-z3), (x-x0)**2+(y-y0)**2+(z-z0)**2-40000,\
             (x-x1)*xd+(y-y1)*yd+(z-z1)*zd], [x, y, z])

        
        cosdict = {}
        for ans in xyz:
            x, y, z = float(ans[0]), float(ans[1]), float(ans[2])
            P2 = np.array([x, y, z])

            a = P2-From
            b = vec1
            cosd = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            cosdict[cosd] = P2

        P2 = cosdict[max(cosdict.keys())]

        P1P2 = np.linalg.norm(From - P2)
        cosP1OP2 = (200**2 + 200**2 -P1P2**2) / (2*200*200)

        if -1 <= cosP1OP2 <= 1:
            arc = math.acos(cosP1OP2)
        else:
            return None, None, None, None

        arc *= 200

        return arc + np.linalg.norm(P2 - To), P0, P2, To - P2

    # 测试走直线能否到达 以及 是否合理 走直线都到不了 曲线就更不可能了 所谓合理即保证每一步都向终点靠近
    def sfun(self, From, To, Eh, Ev, hve='h'):
        dist = np.linalg.norm(From - To)
        error = dist*delta
        # print(Eh, Ev, error)
        if hve == 'e':
            if Eh + error <= theta and Ev + error <= theta:
                return True
        else:
            if dist != 0 and np.linalg.norm(From - end) >= np.linalg.norm(To - end):
                if (hve == 'h' and Eh + error <= b2 and Ev + error <= b1) or (hve == 'v' and Eh + error <= a2 and Ev + error <= a1):
                    return True

        return False

    # 检测先走水平检测点的路线
    def hfun(self, current, Eh, Ev, vec, route, route_hvidx, dist_route, dist_route_end, vecnextlist, dhlist, istoend, Ehv):
        for hi, h in horcheck.items():

            if not self.sfun(current, h, Eh, Ev, 'h'):
                continue

            dist, P0, P2, vecnext = self.dfun(current, h, vec)
            if dist == None:
                continue

            error = dist*delta
            # 能够到达水平校正点并进行校正
            if Eh + error <= b2 and Ev + error <= b1:

                if self.sfun(h, end, 0, Ev+error, 'e'):

                    # 判断经过水平校正后能否直接到终点
                    dist2, P02, P22, vecnext2 = self.dfun(h, end, vecnext)
                    if dist2 != None:
                        error2 = dist2*delta
                        if Ev + error + error2 <= theta:
                            istoend.append((True, [h, end], [(hi, Eh+error, Ev+error), (endi, error2, Ev+error+error2)], dist+dist2, 0, [vecnext2], [(current, P2, P0), (h, P22, P02)], [0, 0]))
                            return

                # 找接下来可达的垂直校正点
                for vi, v in vercheck.items():
                    # print(vi)
                    if not self.sfun(h, v, 0, Ev+error, 'v'):
                        continue

                    dist3, P03, P23, vecnext3 = self.dfun(h, v, vecnext)
                    if dist3 == None:
                        continue
                    error3 = dist3*delta

                    if 0 + error3 <= a2 and Ev + error + error3 <= a1:
                        # print(1, hi, vi)

                        # 如果已经验证走了 这条路径后 既无法到达终点也无法到达下一个校正点 跳过路径
                        # 但是Q2中由于还要考虑方向 因此不适用
                        # if (hi, vi) in self.terminatepath:
                        #     continue
                        
                        # 这一部分是当有多条路径而终点相同的情况 只保留路径最短的 加这个不会搜索加速 但能让结果更好
                        re = -1
                        for i, hv in enumerate(route_hvidx):
                            if hv[-1][0] == vi:
                                if dist_route[i] > dist + dist3:
                                    re = i
                                else:
                                    re = -2
                                break

                        if re >= 0:
                            del route[re]
                            del route_hvidx[re]
                            del dist_route[re]
                            del dist_route_end[re]
                            del vecnextlist[re]
                            del dhlist[re]
                            del Ehv[re]
                        elif re == -2:
                            continue
                        
                        dist_end, _, _, _ = self.dfun(v, end, vecnext3)
                        route.append([h, v])
                        route_hvidx.append([(hi, Eh+error, Ev+error), (vi, 0+error3, Ev+error+error3)])
                        dist_route.append(dist + dist3)
                        dist_route_end.append(dist_end)
                        vecnextlist.append(vecnext)
                        dhlist.append([(current, P2, P0), (h, P23, P03)])
                        Ehv.append([error3, 0])

        istoend.append(False)
        return
        # return False, route, route_hvidx, dist_route, dist_route_end, vecnextlist, dhlist

    # 检测先走竖直检测点的路线
    def vfun(self, current, Eh, Ev, vec, route, route_hvidx, dist_route, dist_route_end, vecnextlist, dhlist, istoend, Ehv):
        for vi, v in vercheck.items():

            if not self.sfun(current, v, Eh, Ev, 'v'):
                continue

            dist, P0, P2, vecnext = self.dfun(current, v, vec)
            if dist == None:
                continue
            error = dist*delta
            # 能够到达垂直校正点并进行校正
            if Eh + error <= a2 and Ev + error <= a1:

                if self.sfun(v, end, Eh+error, 0, 'e'):

                    # 判断经过垂直校正后能否直接到终点
                    dist2, P02, P22, vecnext2 = self.dfun(v, end, vecnext)
                    if dist2 != None:
                        error2 = dist2*delta
                        if Eh + error + error2 <= theta:
                            istoend.append((True, [v, end], [(vi, Eh+error, Ev+error), (endi, Eh+error+error2, error2)], dist+dist2, 0, [vecnext2], [(current, P2, P0), (v, P22, P02)], [0, 0]))
                            return

                # 找接下来可达的水平校正点
                for hi, h in horcheck.items():

                    if not self.sfun(v, h, Eh+error, 0, 'h'):
                        continue

                    dist3, P03, P23, vecnext3 = self.dfun(v, h, vecnext)
                    if dist3 == None:
                        continue
                    error3 = dist3*delta

                    if 0 + error3 <= b2 and Eh + error + error3 <= b1:

                        # if (vi, hi) in self.terminatepath:
                        #     continue

                        re = -1
                        for i, hv in enumerate(route_hvidx):
                            if hv[-1][0] == hi:
                                if dist_route[i] > dist + dist3:
                                    re = i
                                else:
                                    re = -2
                                break

                        if re >= 0:
                            del route[re]
                            del route_hvidx[re]
                            del dist_route[re]
                            del dist_route_end[re]
                            del vecnextlist[re]
                            del dhlist[re]
                            del Ehv[re]
                        elif re == -2:
                            continue

                        dist_end, _, _, _ = self.dfun(h, end, vecnext3)
                        route.append([v, h])
                        route_hvidx.append([(vi, Eh+error, Ev+error), (hi, Eh+error+error3, 0+error3)])
                        dist_route.append(dist + dist3)
                        dist_route_end.append(dist_end)
                        vecnextlist.append(vecnext)
                        dhlist.append([(current, P2, P0), (v, P23, P03)])
                        Ehv.append([0, error3])
                        

        istoend.append(False)
        return
        # return False, route, route_hvidx, dist_route, dist_route_end, vecnextlist, dhlist


    # 双进程
    def checkroute(self, current, Eh, Ev, vec):
        """
            返回值: 
                istoend 是否到达终点 若到达终点 则其记录了到达终点路径信息 可以直接返回
                route 可行的校正路径列表(直接包含了水平与竖直校正) 
                route_hvidx 路径经过水平与竖直校正点时相应的水平竖直误差列表(题目要求)
                dist_route 路径长度列表
                idx 根据某种规则排序后元素序号列表
                vecnextlist 走完路径后方向向量列表
                dhlist 轨迹作图信息列表 包含两段轨迹出发点 切点 与 圆心坐标
                Ehv 走完路径后 的(Eh, Ev)
        """
        ma = Manager()

        h_route = ma.list()
        h_dist_route = ma.list()
        h_dist_route_end = ma.list()
        h_route_hvidx = ma.list()
        h_vecnextlist = ma.list()
        h_dhlist = ma.list()
        h_istoend = ma.list()
        h_Ehv = ma.list()

        v_route = ma.list()
        v_dist_route = ma.list()
        v_dist_route_end = ma.list()
        v_route_hvidx = ma.list()
        v_vecnextlist = ma.list()
        v_dhlist = ma.list()
        v_istoend = ma.list()
        v_Ehv = ma.list()

        p1 = Process(target=self.hfun, args=(current, Eh, Ev, vec, h_route, h_route_hvidx, h_dist_route, h_dist_route_end, h_vecnextlist, h_dhlist, h_istoend, h_Ehv))
        p2 = Process(target=self.vfun, args=(current, Eh, Ev, vec, v_route, v_route_hvidx, v_dist_route, v_dist_route_end, v_vecnextlist, v_dhlist, v_istoend, v_Ehv))

        p1.start()
        p2.start()

        p1.join()
        p2.join()

        if h_istoend[0]:
            return h_istoend[0]

        if v_istoend[0]:
            return v_istoend[0]

        route = list(h_route) + list(v_route)
        dist_route = list(h_dist_route) + list(v_dist_route)
        dist_route_end = list(h_dist_route_end) + list(v_dist_route_end)
        route_hvidx = list(h_route_hvidx) + list(v_route_hvidx)
        vecnextlist = list(h_vecnextlist) + list(v_vecnextlist)
        dhlist = list(h_dhlist) + list(v_dhlist)
        Ehv = list(h_Ehv) + list(v_Ehv)

        # 到终点距离从小到大排序
        idx = sorted(range(len(dist_route_end)), key=lambda k: dist_route_end[k], reverse=False)
        # 返回8个值
        return False, route, route_hvidx, dist_route, idx, vecnextlist, dhlist, Ehv

    def show(self):
        print("len(Lse):", len(self.Lse))
        print("Nse: ", self.Nse)
        print("lse: ", self.lse)
        print("hvidxse: ", self.hvidxse)
        print("drawhelper: ", self.drawhelper)

na = Navigator()
na.navigate(current=start, lc=0, Nc=0, hvidx=[], Lc=[], Eh=0, Ev=0, k=8000, vec=None, dh=[])

# a = np.array([0,0,200])
# b = np.array([0,400,0])
# vec = np.array([0,0,1])
# dist, P0, P2, vecnext = na.dfun(a, b, vec)
# print(dist, P0, P2, vecnext)
# exit()


print("--- 搜索结束 ---")
print("Time used:", (time.time() - sc))
print("a1: {}\na2: {}\nb1: {}\nb2: {}\ntheta: {}\ndelta: {}".format(a1, a2, b1, b2, theta, delta))
na.show()


# 输出
out = []
for L in na.Lse:
    out.append(list(map(list, L)))

with open('pyout{}.txt'.format(dataidx), 'w') as f:
    f.write(str(out))


with open('drawhelper{}.txt'.format(dataidx), 'w') as f:
    f.write(str(na.drawhelper))
