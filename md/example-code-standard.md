# Example类型程序代码标准

## 准则
- 主要目的是来展示算法的实现
- 面向程序来编写，中间也要模块化，对象化，函数化来实现
- 用doxygen的方式来写程序注释
- 要用参数来控制控制程序
- 编写时注意对不用维度都要适用

## 格式要求
- 逗号后面加空格
- 一行不要太长

## 框架
- 程序说明(doxygen)
```python
'''！    	
@Author: wpx
@File Name: level.py
@Author: wpx
@Mail: wpx15673207315@gmail.com 
@Created Time: 2021年11月19日 星期五 11时42分52秒
@bred: 程序中存在的问题
@ref：参考文献
''' 
```
- 调包(只调取用的到的，简洁)
- 用args对程序进行参数控制
```python
import argparse
parser = argparse.ArgumentParser(description=
        """
        有限元方法求解水平集演化方程,时间离散CN格式
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

degree = args.degree
```
- 用到的函数定义
- 维度控制
```python
if dim == 2:
    domain = [0, 1, 0, 1]
    mesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
    space = LagrangeFiniteElementSpace(mesh, p=degree)
    phi0 = space.interpolation(circle)
    u = space.interpolation(u2, dim=dim)
else:
    domain = [0, 1, 0, 1, 0, 1]
    mesh = MF.boxmesh3d(domain, nx=ns, ny=ns, nz=ns, meshtype='tet')
    space = LagrangeFiniteElementSpace(mesh, p=degree)
    phi0 = space.interpolation(sphere)
    u = space.interpolation(u3, dim=dim)
```
- 主程序代码

