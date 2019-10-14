import xlrd
import numpy as np
import math

dataidx = 2
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
        self.lse = []
        self.Nse = []
        self.Lse = []
        self.hvidxse = []

    def navigate(self, current, lc, Nc, hvidx, Lc, Eh, Ev, k):
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

        dist_end = np.linalg.norm(current - end)
        # 判断是否因为 该路径不可能最优 而中断搜索
        if self.prune(lc, dist_end, Nc):
            # print("search terminated")
            return False

        # 这回合到达终点
        error = dist_end*delta
        if Ev + error <= theta and Eh + error <= theta:
            self.lse.append(lc + dist_end)
            self.Nse.append(Nc)
            self.Lse.append(Lc + [end])
            self.hvidxse.append(hvidx)
            print("route found {} Nc: {} l: {}".format(len(self.Lse), Nc, self.lse[-1]))
            return True

        # 继续搜索
        # 判断能否朝目标直线飞行km
        flag = False
        if Ev + k*delta < min(a1, b1) and Eh + k*delta < min(a2, b2):
            nextpos = self.getstraightpos(current, end, k)
            flag = self.navigate(nextpos, lc+k, Nc, hvidx, Lc+[nextpos], Eh+k*delta, Ev+k*delta, k)

        # flag为False有两种情况 1.不能直行km 2.直行km后依然找不到路径
        # 那么将进入寻找校正点
        if not flag:
            istoend, route, route_hvidx, dist_route, idx = self.checkroute(current, Eh, Ev)
            if istoend:
                self.lse.append(lc + dist_route)
                self.Nse.append(Nc + 1)
                self.Lse.append(Lc + route)
                self.hvidxse.append(hvidx + route_hvidx)
                print("route found", len(self.Lse))
                return True
            elif route != {}:
                for i in idx[:3]:
                    flag = self.navigate(route[i][1], lc+dist_route[i], Nc+2, hvidx+route_hvidx[i], Lc+list(route[i]), 0, 0, k)
            else:
                return False
        else:
            return True

    # 判断是否需要中断搜索
    def prune(self, lc, dist_end, Nc):
        if self.Lse != []:
            return lc + dist_end >= min(self.lse) and Nc >= min(self.Nse)
        else:
            return False

    # 获取直线行驶d后的坐标 基于相似三角形
    def getstraightpos(self, From, To, d):
        dist = np.linalg.norm(From - To)
        if d >= dist:
            return To

        return From + d/dist * (To - From)

    def checkroute(self, current, Eh, Ev):
        """
            返回值: 是否到达终点 可行的校正路径
        """
        route = []
        dist_route = []
        dist_route_end = []
        route_hvidx = []

        for hi, h in horcheck.items():
            dist = np.linalg.norm(current - h)
            error = dist*delta
            # 能够到达水平校正点并进行校正
            if Eh + error <= b2 and Ev + error <= b1:
                # 判断经过水平校正后能否直接到终点
                dist2 = np.linalg.norm(h - end)
                error2 = dist2*delta
                if Ev + error + error2 <= theta:
                    return True, [h, end], [(hi, Eh+error, Ev+error)], dist + dist2

                # 找接下来可达的垂直校正点
                for vi, v in vercheck.items():
                    dist3 = np.linalg.norm(h - v)
                    error3 = dist3*delta

                    if 0 + error3 <= a2 and Ev + error + error3 <= a1:
                        dist_end = np.linalg.norm(v - end)
                        route.append([h, v])
                        route_hvidx.append([(hi, Eh+error, Ev+error), (vi, 0+error3, Ev+error+error3)])
                        dist_route.append(dist + dist3)
                        dist_route_end.append(dist_end)

        for vi, v in vercheck.items():
            dist = np.linalg.norm(current - v)
            error = dist*delta
            # 能够到达垂直校正点并进行校正
            if Eh + error <= a2 and Ev + error <= a1:
                # 判断经过垂直校正后能否直接到终点
                dist2 = np.linalg.norm(v - end)
                error2 = dist2*delta
                if Eh + error + error2 <= theta:
                    return True, [v, end], [(vi, Eh+error, Ev+error)], dist + dist2

                # 找接下来可达的水平校正点
                for hi, h in horcheck.items():
                    dist3 = np.linalg.norm(v - h)
                    error3 = dist3*delta

                    if 0 + error3 <= b2 and Eh + error + error3 <= b1:
                        dist_end = np.linalg.norm(h - end)
                        route.append([v, h])
                        route_hvidx.append([(vi, Eh+error, Ev+error), (hi, Eh+error+error3, 0+error3)])
                        dist_route.append(dist + dist3)
                        dist_route_end.append(dist_end)

        # 到终点距离从小到大排序
        idx = sorted(range(len(dist_route_end)), key=lambda k: dist_route_end[k], reverse=False)
        return False, route, route_hvidx, dist_route, idx

    def show(self):
        print("len(Lse):", len(self.Lse))
        print("Nse: ", self.Nse)
        print("lse: ", self.lse)
        print("hvidxse: ", self.hvidxse)



na = Navigator()
na.navigate(current=start, lc=0, Nc=0, hvidx=[], Lc=[], Eh=0, Ev=0, k=5000)

print("a1: {}\na2: {}\nb1: {}\nb2: {}\ntheta: {}\ndelta: {}".format(a1, a2, b1, b2, theta, delta))
na.show()

out = []
for L in na.Lse:
    out.append(list(map(list, L)))

with open('pyout.txt{}'.format(dataidx), 'w') as f:
    f.write(str(out))
