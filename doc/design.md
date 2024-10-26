# Design Doc

## Others

### 去杂质的规则

1. 二值化使用OTSU算法后的校正值
OTSU_PLUS = 20

2. 杂质上限，小于该大小的轮廓直接丢弃
CNT_AREA_MIN = 120

3. 待去除杂质的轮廓的最大面积，再此基础上再根据外形就行区分
CNT_AREA_MAX = 1000000

4. 过滤不显著凹陷（凸缺陷）的阈值
CONCAVE_DEPTH_THRESH = 5000
