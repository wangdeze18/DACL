from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

predef = [471, 458, 493, 483, 492, 486, 486, 460, 487, 485, 490, 493, 500, 472, 494, 496, 463, 485, 468, 487, 491, 483, 483, 467, 492, 485, 495, 483, 495, 492, 430, 481, 490, 482, 493, 490, 482, 492, 461, 463, 498, 498, 489, 455, 491, 494, 490, 497, 487, 496, 483, 486, 497, 470, 494, 487, 485, 470, 491, 493, 487, 484, 494, 492]
xlist = []
ylist = []

with open("std_scores.txt","r") as f:
    scores = f.readlines()
    for i in range(len(scores)):
        score = float(scores[i].strip())
        ylist.append(score)

start_num = 0
pre_num = 0
newy_list = []
for i in range(len(predef)):
    num = predef[i]
    pre_num = start_num
    start_num += num

    sum = 0
    for j in range(pre_num,start_num):
        sum += ylist[j]

    newy_list.append(sum/num)
print(newy_list)
for i in range(1,65):
    xlist.append(i)

plt.plot(xlist, newy_list,c='r')

score_list = []
for i in range(len(predef)):
    num = int(predef[i])
    for j in range(num):
        score_list.append(newy_list[i])
arr = np.array(score_list)
np.save("class_score.npy",arr)

#predef = [1735, 1858, 1978, 1950, 2220, 1981, 1835, 1605, 2074, 2113, 1761, 2280, 1949, 1956, 1992, 1544, 1954, 1898, 2007, 1889, 1944, 2183, 1896, 1813, 1844, 1717, 1713, 2105, 1541, 1976, 1480, 2033, 1833, 1530, 2036, 1899, 2082, 1554, 2226, 1211, 1694, 1999, 1948, 1680, 1874, 2099, 1755, 1956, 1763, 1912, 2148, 1834, 2055, 1751, 2356, 1660, 2056, 2202, 2093, 2109, 1847, 1747, 2002, 1679]
predef = [1825, 1828, 1972, 1931, 1964, 1935, 1940, 1749, 1939, 1905, 1958, 1967, 2000, 1876, 1976, 1982, 1847, 1931, 1870, 1913, 1963, 1930, 1924, 1866, 1966, 1940, 1971, 1932, 1975, 1961, 1685, 1904, 1956, 1892, 1968, 1928, 1924, 1954, 1839, 1781, 1990, 1988, 1936, 1684, 1964, 1968, 1957, 1980, 1934, 1983, 1927, 1909, 1979, 1847, 1965, 1937, 1933, 1878, 1964, 1961, 1935, 1934, 1976, 1962]

ylist_trans0 = []

with open("no_dup_scores.txt","r") as f:
    scores = f.readlines()
    for i in range(len(scores)):
        score = float(scores[i].strip())
        ylist_trans0.append(score)

start_num = 0
pre_num = 0
newy_list_trans0 = []
for i in range(len(predef)):
    num = predef[i]
    pre_num = start_num
    start_num += num

    sum = 0
    for j in range(pre_num,start_num):
        sum += ylist_trans0[j]

    newy_list_trans0.append(sum/num)
print(newy_list_trans0)
plt.plot(xlist, newy_list_trans0,c='b')


plt.title('line chart')
plt.xlabel('x')
plt.ylabel('y')

plt.show()