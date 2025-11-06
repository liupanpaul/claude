# 备考复习 (Lecture/Tutorial) - Week 1

欢迎来到《金融时间序列导论》的第一周学习。本周的核心任务是搭建整个课程的基石：**理解我们分析的对象是什么（金融数据），以及我们需要用什么理论工具来分析它（概率论）**。我们将从最基础的金融回报率概念入手，学习如何用 Python 计算和呈现它们，并复习描述不确定性所必需的概率论知识。最终，我们会将数据与理论结合，检验一个著名的金融模型。

---

## 1. 课程基石：什么是量化金融 (Quantitative Finance)

### 1.1. 核心思想 (Core Idea)

量化金融 (Quantitative Finance) 试图理解金融世界背后运行的“规律 (laws)”，并利用这些规律来做出更优的财务决策。这不仅仅是提出理论，更关键的是，我们必须做到以下两点：

1.  **数据检验 (Data Validation)**：任何金融理论或模型，都必须用真实观测到的数据进行严格的检验。
2.  **数据驱动 (Data-Driven Decisions)**：决策应基于真实且相关的数据，而不是直觉或猜测。

简单来说，这门课程的框架就是：**（1）建立金融/统计模型 →（2）获取真实数据 →（3）用数据检验模型 →（4）基于检验结果做出决策**。

### 1.2. 举个例子

假设“蜜雪东城”的分析师团队建立了一个模型，预测他们家股票明天的价格。这个模型就是一个理论。为了验证它，他们需要收集过去几年的股票价格数据（真实数据），然后看模型的预测结果与实际情况的吻合度（数据检验）。如果模型表现良好，公司管理层就可以依据它的预测来决定是否回购股票（数据驱动决策）。

---

## 2. 核心数据：金融资产回报率 (Financial Returns)

在金融分析中，我们通常更关心资产价格的 *变化*，而不是价格本身。这个变化就用“回报率”来衡量。

### 2.1. 简单回报率 (Simple Returns)

这是最直观的回报率定义。它衡量的是从一个时期到下一个时期，资产价格变化的百分比。

*   **概念阐释**: 假设 $P_t$ 是资产在时间点 $t$ 的价格，那么在 $t$ 时刻的简单回报率 $R_t$ 就是当期价格相对于上一期价格的增值部分。

*   **公式呈现**:
    $$
    R_t = \frac{P_t - P_{t-1}}{P_{t-1}}
    $$
    这个公式也可以写成：
    $$
    R_t = \frac{P_t}{P_{t-1}} - 1
    $$

*   **举个例子**: 假设“蜜雪东城”的股票昨天（$t-1$）收盘价是 `￥100`，今天（$t$）的收盘价是 `￥102`。

*   **计算过程全景展示**:
    1.  **公式呈现**: $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$
    2.  **案例代入**: $R_t = \frac{￥102 - ￥100}{￥100}$
    3.  **计算步骤拆解**: $R_t = \frac{￥2}{￥100} = 0.02$
    4.  **结果解读**: 今天的简单回报率是 `0.02`，也就是 `2%`。这意味着，如果你昨天花 `￥100` 买入股票，到今天你就赚了 `￥2`。

### 2.2. 多时期回报率 (Multi-period Returns)

当我们需要计算跨越多个时期的总回报率时，例如一个季度的回报率，我们需要将每一期的回报率“串联”起来。

*   **概念阐释**: $k$-时期回报率 ($k$-period return)，记作 $R_t[k]$，衡量的是从 $t-k$ 时期到 $t$ 时期的总价格变化。

*   **公式呈现**:
    直接定义：
    $$
    R_t[k] = \frac{P_t - P_{t-k}}{P_{t-k}}
    $$
    通过单期回报率复合计算：
    $$
    1 + R_t[k] = (1+R_t)(1+R_{t-1})...(1+R_{t-k+1})
    $$

*   **举个例子**: 假设"蜜雪东城"股票第一天的回报率 $R_1$ 是 `10%`，第二天的回报率 $R_2$ 是 `-5%`。我们来计算这两天的总回报率 $R_2[2]$。

*   **计算过程全景展示**:
    1.  **公式呈现**: $1 + R_2[2] = (1+R_2)(1+R_1)$
    2.  **案例代入**: $1 + R_2[2] = (1 - 0.05)(1 + 0.10)$
    3.  **计算步骤拆解**:
        *   $1 + R_2[2] = (0.95) \times (1.10) = 1.045$
        *   $R_2[2] = 1.045 - 1 = 0.045$
    4.  **结果解读**: 这两天的总回报率是 `4.5%`。注意，它不是简单地将 `10%` 和 `-5%` 相加。

### 2.3. 对数回报率 (Log Returns)

对数回报率在金融建模中极为常用，因为它具有非常优秀的数学特性。

*   **概念阐释**: 对数回报率，通常记作 $r_t$，是资产价格取自然对数后的差值。它近似于简单回报率，尤其是在回报率数值很小的时候。

*   **公式呈现**:
    $$
    r_t = \ln(1+R_t) = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})
    $$

*   **核心优势：可加性 (Additivity)**
    简单回报率的复合计算需要用乘法，而对数回报率的复合计算只需要用加法，这极大地方便了数学处理。
    一个 $k$-时期的对数回报率 $r_t[k]$，就是过去 $k$ 个单期对数回报率的和：
    $$
    r_t[k] = \ln(P_t) - \ln(P_{t-k}) = r_t + r_{t-1} + ... + r_{t-k+1}
    $$

*   **举个例子**: 再次使用“蜜雪东城”的例子，昨天价格 `￥100`，今天 `￥102`。

*   **计算过程全景展示**:
    1.  **公式呈现**: $r_t = \ln(P_t) - \ln(P_{t-1})$
    2.  **案例代入**: $r_t = \ln(102) - \ln(100)$
    3.  **计算步骤拆解**: $r_t \approx 4.62497 - 4.60517 = 0.0198$
    4.  **结果解读**: 对数回报率约为 `1.98%`。可以看到，这个结果 `1.98%` 和简单回报率 `2%` 非常接近。当回报率越小，两者就越接近。

### 2.4. 数据频率 (Data Frequency)

我们分析的数据可以是任意时间频率的，比如分钟、小时、日、周、月等。在本课程中，我们将主要关注 **日度回报率 (daily returns)**，即从一个交易日的收盘价到下一个交易日收盘价的回报率。

---

## 7. 模块 1-2 练习题

### 7.1 题目 (Original Example Questions)

1.  (选择题) 某只股票的价格从周一的 `￥50` 上涨到周二的 `￥52`，然后在周三又下跌到 `￥49.4`。请问从周一到周三，这只股票的 **总简单回报率** 是多少？
    A. -1.2%
    B. -1.0%
    C. 1.2%
    D. 4.0%

2.  (选择题) 分析师小王在研究一支股票时，发现其连续五天的对数回报率分别为：0.1%, -0.2%, 0.3%, -0.1%, 0.4%。这五天的 **总对数回报率** 是多少？
    A. 无法计算，需要知道每日价格
    B. 0.5%
    C. 1.1%
    D. 0.0005%

3.  (判断题) 当资产价格的回报率非常小（例如，绝对值小于1%）时，其简单回报率和对数回报率的数值几乎相等。

4.  (判断题) 如果一项资产的简单回报率 $R_t = -1.5$，这意味着该资产价格变成了负数。

5.  (简答题) 为什么在很多金融学术研究和量化模型中，研究者更偏爱使用对数回报率而不是简单回报率？请至少说出一个核心原因。

### 7.2 解析 (Solution Walkthrough)

1.  **答案: A**
    *   **思路**: 总简单回报率可以直接用期初和期末的价格计算，无需关心中间价格。
    *   **计算**: $R = \frac{P_{周三} - P_{周一}}{P_{周一}} = \frac{￥49.4 - ￥50}{￥50} = \frac{-￥0.6}{￥50} = -0.012 = -1.2\%$。

2.  **答案: B**
    *   **思路**: 对数回报率的核心优势是其可加性。多时期的总对数回报率等于各单期对数回报率之和。
    *   **计算**: 总对数回报率 = $0.1\% - 0.2\% + 0.3\% - 0.1\% + 0.4\% = (0.1 - 0.2 + 0.3 - 0.1 + 0.4)\% = 0.5\%$。

3.  **答案: 正确**
    *   **思路**: 这是对数回报率的一个重要特性。数学上，当 $x$ 趋近于0时，$\ln(1+x) \approx x$。因为简单回报率是 $R_t$，对数回报率是 $\ln(1+R_t)$，所以在 $R_t$ 很小时，两者非常接近。

4.  **答案: 错误**
    *   **思路**: 回报率的定义是 $R_t = \frac{P_t}{P_{t-1}} - 1$。因此，回报率的最小值是 `-1` (或 `-100%`)，这发生在资产价格 $P_t$ 跌至 `0` 的时候。回报率不可能低于 `-1`，因为资产价格不会是负数。

5.  **答案**:
    *   **核心原因**: **可加性 (Additivity)**。如第2题所示，计算多时期的总回报时，对数回报率可以直接相加，而简单回报率需要进行复合（相乘）计算。加法运算在数学和统计建模中远比乘法运算方便处理，尤其是在分析长期收益分布时。

---

## 3. Python实践：处理与可视化金融数据 (Data Handling in Python)

理论需要通过实践来巩固。在量化金融中，Python 是进行数据分析和模型建立的核心工具。我们将以澳大利亚联邦银行 (Commonwealth Bank of Australia, CBA) 的股票数据为例，展示一个完整的处理流程。

### 3.1. 读取和准备数据 (Reading and Preparing Data)

第一步总是获取数据。通常，金融数据会以 `.csv` 格式存储，其中包含日期、开盘价、最高价、最低价、收盘价等信息。

*   **概念阐释**: 我们使用 `pandas` 库，它是 Python 中数据处理的“瑞士军刀”。`pd.read_csv()` 函数可以轻松读取数据文件。读取后，数据被存储在一个叫做 `DataFrame` 的强大结构中，可以把它想象成一个智能的电子表格。

*   **关键代码解析**:
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt

    # 1. 读取数据
    # 假设数据文件名为 'cba_yf_2000_072025.csv'
    cba = pd.read_csv('cba_yf_2000_072025.csv')

    # 2. 转换日期格式
    # 原始数据中的'Date'列是文本，需要转换为Python可识别的日期时间格式
    cba['Date'] = pd.to_datetime(cba['Date'])
    ```
    *   **`import pandas as pd`**: 引入 `pandas` 库，并用 `pd` 作为它的简称，这是社区的通用惯例。
    *   **`pd.read_csv(...)`**: 从 CSV 文件中读取数据，并将其加载到名为 `cba` 的 DataFrame 中。
    *   **`pd.to_datetime(...)`**: 这是一个非常重要的步骤。它将文本格式的日期（如 "2023-10-27"）转换成标准的日期时间对象，这样我们才能基于时间进行排序、筛选和绘图。

### 3.2. 绘制股价走势图 (Plotting Share Price)

可视化是理解数据的第一步。通过绘制价格随时间变化的图表，我们可以直观地看到股票的长期趋势、波动性和重大事件的影响。

*   **概念阐释**: 我们使用 `matplotlib.pyplot` 库，通常简写为 `plt`，它是 Python 中最基础也最强大的绘图库。

*   **关键代码解析**:
    ```python
    # 使用 'Date' 作为 x 轴, 'Close' (收盘价) 作为 y 轴
    plt.plot(cba['Date'], cba['Close'])

    # 添加图表标签，让图表更具可读性
    plt.xlabel('Date', fontsize='15')
    plt.ylabel('CBA Price (AUD)', fontsize='15')

    # 显示图表
    plt.show()
    ```
    *   **`plt.plot(x, y)`**: 绘制线图，x 轴是日期，y 轴是收盘价。
    *   **`plt.xlabel()` / `plt.ylabel()`**: 分别设置 x 轴和 y 轴的标签。
    *   **`plt.show()`**: 将绘制好的图表显示出来。从图中（如幻灯片第9页所示），我们可以清晰地看到 CBA 股价在 2008 年金融危机和 2020 年疫情期间的大幅下跌。

### 3.3. 计算并绘制回报率 (Computing and Plotting Returns)

现在我们来计算之前讨论的两种回报率。

*   **计算简单回报率 (Simple Returns)**:
    *   **概念阐释**: 要计算 $R_t = (P_t - P_{t-1}) / P_{t-1}$，我们需要用到当天的价格 $P_t$ 和昨天的价格 $P_{t-1}$。在 `pandas` 中，可以用 `.shift(1)` 方法非常方便地获取上一期的数据。
    *   **关键代码解析**:
        ```python
        # .shift(1) 会将整个 'Close' 列的数据向下移动一行，从而得到 P_{t-1}
        cba['Returns'] = (cba['Close'] - cba['Close'].shift(1)) / cba['Close'].shift(1)
        
        # 绘制回报率图
        plt.plot(cba['Date'], cba['Returns'])
        plt.ylabel('CBA Returns')
        plt.show()
        ```
        *   从回报率图中（如幻灯片第11页所示），我们看到回报率序列不像价格序列那样有明显的长期趋势。它看起来更像是在一个固定水平（零附近）上下波动的随机序列。这种现象被称为 **“波动率聚集” (Volatility Clustering)**，即大的波动（无论正负）倾向于聚集在一起出现。

*   **计算对数回报率 (Log Returns)**:
    *   **概念阐释**: 计算 $r_t = \ln(P_t) - \ln(P_{t-1})$。我们需要使用 `numpy` 库（通常简写为 `np`）来进行对数运算。
    *   **关键代码解析**:
        ```python
        import numpy as np

        # 使用 np.log() 计算自然对数
        cba['logReturns'] = np.log(cba['Close']) - np.log(cba['Close'].shift(1))
        ```
    *   **对比简单回报率与对数回报率**: 为了验证 $r_t \approx R_t$，我们可以绘制一个散点图，横轴是对数回报率，纵轴是简单回报率。
        ```python
        plt.scatter(cba['logReturns'], cba['Returns'])
        plt.xlabel('CBA log Returns')
        plt.ylabel('CBA Returns')
        # 添加一条 y=x 的参考线
        plt.plot([-0.1, 0.1], [-0.1, 0.1])
        plt.show()
        ```
        *   从图中（如幻灯片第16页所示），可以看到所有的点几乎都落在 $y=x$ 这条线上，完美地印证了在日度回报率这种小数值尺度下，两种回报率高度一致。

---

## 8. 模块 3 练习题

### 8.1 题目

1.  (选择题) 在使用 `pandas` 处理金融数据时，`df['Price'].shift(-1)` 这行代码的含义是什么？
    A. 获取昨天的价格
    B. 获取明天的价格
    C. 将价格数据整体向上平移一行
    D. B 和 C 都是正确的

2.  (选择题) 你拿到一份“蜜雪东城”的股票数据 `mx_stock`，其中包含 `'Date'` 和 `'Close'` 两列。你想计算其每日简单回报率并存入新列 `'Simple_Return'`，以下哪段代码是正确的？
    A. `mx_stock['Simple_Return'] = (mx_stock['Close'] - mx_stock['Close'].shift(1)) / mx_stock['Close'].shift(1)`
    B. `mx_stock['Simple_Return'] = (mx_stock['Close'].shift(1) - mx_stock['Close']) / mx_stock['Close']`
    C. `mx_stock['Simple_Return'] = np.log(mx_stock['Close']) - np.log(mx_stock['Close'].shift(1))`
    D. `mx_stock['Simple_Return'] = mx_stock['Close'] / mx_stock['Close'].shift(1)`

3.  (判断题) 在绘制股票价格图时，如果发现日期轴 (`x` 轴) 的标签混乱且顺序不正确，最可能的原因是没有使用 `pd.to_datetime()` 函数将日期列转换为标准的日期时间格式。

4.  (填空题) 在 Python 中，进行科学计算（如对数、指数运算）通常需要导入 `___________` 库，而进行数据操作和分析则主要依赖 `___________` 库。

5.  (简答题) 观察典型的股票 **价格** 时间序列图和 **回报率** 时间序列图，它们在视觉上最显著的一个区别是什么？这个区别在金融分析中意味着什么？

### 8.2 解析

1.  **答案: D**
    *   **思路**: `.shift()` 函数的参数决定了移动的方向和幅度。`shift(1)` 是向下移动（获取过去的数据），而 `shift(-1)` 则是向上移动（获取未来的数据）。因此，`shift(-1)` 既是获取明天的价格，也是将数据向上平移一行的操作。

2.  **答案: A**
    *   **思路**: 简单回报率的公式是 $(P_t - P_{t-1}) / P_{t-1}$。在代码中，$P_t$ 对应 `mx_stock['Close']`，$P_{t-1}$ 对应 `mx_stock['Close'].shift(1)`。选项 A 完美匹配该公式。选项 B 分子分母都错了；选项 C 计算的是对数回报率；选项 D 计算的是 $1+R_t$。

3.  **答案: 正确**
    *   **思路**: `matplotlib` 在绘图时，如果 x 轴是标准的日期时间对象，它会自动进行合理的排序和格式化。如果 x 轴只是普通的文本字符串，它会按照字符串的默认顺序来绘制，这往往会导致混乱。因此，`pd.to_datetime()` 是保证时间序列图正确性的关键一步。

4.  **答案: numpy, pandas**
    *   **思路**: 这是对 Python 数据科学生态中最核心的两个库的基本认知。`numpy` (Numerical Python) 负责底层的数值计算和数组操作。`pandas` 建立在 `numpy` 之上，提供了 DataFrame 这种高级数据结构，专门用于数据清洗、转换和分析。

5.  **答案**:
    *   **最显著的区别**: **价格** 序列通常表现出明显的 **趋势性 (Trend)** 或 **非平稳性 (Non-stationarity)**，即序列的均值和方差会随时间变化，看起来像是在“游走”。而 **回报率** 序列通常表现出 **平稳性 (Stationarity)**，即序列看起来像是在一个固定的均值（通常是0）附近随机波动，没有明显的长期趋势。
    *   **金融意义**: 这个区别至关重要。大多数标准的时间序列模型（如 ARMA, GARCH 等）都要求输入数据是平稳的。因此，我们不能直接对价格序列建模，而是要先将其转换为平稳的回报率序列，再对回报率进行建模和预测。这是金融时间序列分析的一个基本原则。

---

## 4. 概率论基础：量化不确定性的语言 (Review of Probability)

### 4.1. 为什么需要概率论？ (Why Probability?)

在金融领域，我们永远无法 100% 确定地预测未来。明天的股票回报率会是多少？没有人能给出确切答案。我们是 **预测者 (forecasters)**，而非 **算命师 (fortune tellers)**。

因此，任何预测行为都必然包含 **不确定性 (uncertainty)**。概率论正是我们用来描述、衡量和处理这种不确定性的数学语言。进行任何涉及风险的决策，都离不开 **概率思维 (probabilistic thinking)**。

### 4.2. 随机变量：我们的分析对象 (Random Variable)

*   **概念阐释**: 随机变量 (Random Variable, rv)，可以看作是一个其数值取决于随机事件结果的变量。我们通常用大写字母（如 $Y$）表示随机变量本身，用小写字母（如 $y$）表示它可能取到的一个具体数值。
    *   **举个例子**: $Y$ = “蜜雪东城”明天的股票对数回报率（这是一个随机的、未知的量）。$y = 0.01$ (即 1%) 是这个回报率可能取到的一个具体值。

*   **随机变量的分类**:
    1.  **离散型随机变量 (Discrete RV)**: 其可能取到的值是有限或可数的。
        *   **例子**: “蜜雪东城”在下一分钟内的交易笔数。可能是0笔, 1笔, 2笔... 但不可能是1.5笔。
    2.  **连续型随机变量 (Continuous RV)**: 可以在一个或多个区间内取任何值，其可能的结果是不可数的。
        *   **例子**: “蜜雪东城”的股票对数回报率。它可以是 0.01, 0.011, 0.0112... 等区间内的任意数值。

### 4.3. 累积分布函数 (CDF)：描述随机变量的通用工具

我们如何从数学上描述一个随机变量的概率特性呢？最通用、最基础的工具就是累积分布函数 (Cumulative Distribution Function, CDF)。

*   **概念阐释**: CDF，记作 $F(y)$，给出了随机变量 $Y$ 的取值小于或等于某个特定值 $y$ 的概率。它回答的问题是：“$Y$ 的值落在这个点左边的概率有多大？”

*   **公式呈现**:
    $$
    F(y) = \text{Pr}(Y \le y)
    $$

*   **核心特性**:
    1.  **取值范围**: CDF 的值域永远在 `[0, 1]` 之间。
    2.  **单调非减**: 随着 $y$ 的增大，$F(y)$ 只会增加或保持不变，绝不会减小。这很直观，因为“小于等于2的概率”不可能比“小于等于1的概率”更小。

### 4.4. 概率密度/质量函数 (PDF/PMF)：更具体的概率描述

除了通用的 CDF，我们还有更具体的函数来描述不同类型的随机变量。

*   **1. 概率质量函数 (Probability Mass Function, PMF)** - **离散专用**
    *   **概念阐释**: PMF，记作 $f(y)$，直接给出了离散随机变量 $Y$ 取某个精确值 $y$ 的概率。
    *   **公式呈现**: $f(y) = \text{Pr}(Y = y)$
    *   **举个例子**: $f(3)$ 就是下一分钟交易笔数正好是3笔的概率。

*   **2. 概率密度函数 (Probability Density Function, PDF)** - **连续专用**
    *   **概念阐释**: PDF，记作 $f(y)$，描述了连续随机变量在某个点附近的概率“密度”。
    *   **⚠️ 重要警告**: PDF 的值 **不是概率**！对于一个连续随机变量，它取任何一个精确值的概率都为零（即 $\text{Pr}(Y=y) = 0$）。
    *   **如何使用 PDF**: 概率是通过计算 PDF 曲线下的 **面积** 得到的。变量落在区间 $[a, b]$ 内的概率等于 PDF 曲线在 $[a, b]$ 上的积分。
        $$
        \text{Pr}(a < Y < b) = \int_a^b f(y)dy
        $$
    *   **PDF 与 CDF 的关系**: PDF 是 CDF 的导数。

### 4.5. 分位函数 (Quantile Function)：CDF 的逆运算

CDF 回答的是“给定一个值，概率是多少？”。分位函数则回答相反的问题。

*   **概念阐释**: 分位函数 (Quantile Function)，是 CDF 的逆函数。它回答的是：“给定一个概率 $p$，对应的那个值是多少？” 换句话说，我们要找一个 $y$，使得 $\text{Pr}(Y \le y) = p$。

*   **金融应用：风险价值 (Value at Risk, VaR)**
    分位函数在金融中有一个极其重要的应用，即计算 VaR。例如，一个投资组合回报率的 5% 分位数 (p=0.05)，就是它的 VaR。如果计算出的值为 -2.5%，则意味着“我们有 95% 的信心，明天的损失不会超过 2.5%”。

*   **特殊分位数：中位数 (Median)**
    当 $p=0.5$ 时，对应的分位数就是 **中位数**。它是一个分界点，随机变量的取值落在此点之上和之下的概率各为 50%。

### 4.6. 期望值 (Expected Value)：均值的理论形态

期望值，或称均值 (mean)，是衡量随机变量中心趋势的最常用指标。

*   **概念阐释**: 期望值 $E[Y]$，可以被理解为一个随机变量在无数次重复试验后，所有可能结果的“平均值”。它是一个理论上的“总体均值 (population mean)”。

*   **公式呈现**:
    *   离散型: $E[Y] = \sum_y y \cdot f(y)$ (所有可能值乘以其概率的总和)
    *   连续型: $E[Y] = \int_y y \cdot f(y)dy$

*   **与样本均值的关系**: 我们在课程中计算的数据均值（例如 CBA 股票回报率的平均值）是 **样本均值 (sample mean)**。根据大数定律，当样本量足够大时，样本均值会非常接近理论上的期望值。

### 4.7. 两种重要分布的实例分析

通过课程中提到的两个分布，可以将上述概念串联起来。

*   **1. 指数分布 (Exponential Distribution)** - **连续型**
    *   **应用场景**: 常用于为两次事件发生之间的时间间隔建模，例如“距离下一次股票交易发生还需要多长时间”。
    *   **参数**: 由一个参数 $\lambda$ (lambda) 控制。
    *   **PDF**: $f(y) = \lambda e^{-\lambda y}$
    *   **CDF**: $F(y) = 1 - e^{-\lambda y}$
    *   **期望值**: $E[Y] = 1/\lambda$
    *   **中位数**: $\ln(2)/\lambda \approx 0.693/\lambda$

    *   **Python 计算 (以 $\lambda=0.5$ 为例)**:
        ```python
        from scipy import stats
        import numpy as np

        # ⚠️ 注意：scipy.stats 中，指数分布的参数是 scale，且 scale = 1/λ
        lam = 0.5
        scale_param = 1/lam # 即 2.0

        # P(Y <= 1)，即 CDF 在 y=1 处的值
        prob = stats.expon.cdf(x=1, scale=scale_param) # 结果约为 0.393

        # PDF 在 y=1 处的值 (不是概率!)
        density = stats.expon.pdf(x=1, scale=scale_param) # 结果约为 0.303

        # 40% 分位数 (p=0.4)
        quantile_40 = stats.expon.ppf(q=0.4, scale=scale_param) # 结果约为 1.021

        # 理论均值
        mean = stats.expon.mean(scale=scale_param) # 结果为 2.0
        ```
    *   **关键洞察**: 对于 $\lambda=0.5$ 的指数分布，其均值 (2.0) 大于中位数 ($\ln(2)/0.5 \approx 1.386$)。这表明该分布是 **右偏 (right-skewed)** 的，即有少量的大值将均值“向右拉”，我们将在下一部分详细讨论。

---

## 9. 模块 4 练习题

### 9.1 题目

1.  (选择题) 对于一个服从参数 $\lambda=0.2$ 的指数分布的随机变量 $Y$，其 PDF 在 $y=0$ 处的值 $f(0)$ 是多少？
    A. 0
    B. 0.2
    C. 1
    D. 5

2.  (选择题) 某分析师使用 `scipy.stats.expon.cdf(x=2, scale=4)` 计算了一个概率。这对应于一个指数分布随机变量 $Y$，其参数 $\lambda$ 是多少？
    A. 4
    B. 2
    C. 0.5
    D. 0.25

3.  (判断题) 某连续型随机变量的 PDF 在某点 $y_0$ 的值为 1.5。这个结果是有效的，因为 PDF 的值可以大于 1。

4.  (填空题) 随机变量的中位数是其分位函数在概率 $p=$ `___________` 处的值。

5.  (简答题) 请用通俗的语言解释“期望值”和“样本均值”之间的区别与联系。

### 9.2 解析

1.  **答案: B**
    *   **思路**: 指数分布的 PDF 公式为 $f(y) = \lambda e^{-\lambda y}$。将 $\lambda=0.2$ 和 $y=0$ 代入即可。
    *   **计算**: $f(0) = 0.2 \times e^{-0.2 \times 0} = 0.2 \times e^0 = 0.2 \times 1 = 0.2$。

2.  **答案: D**
    *   **思路**: 这道题考察 `scipy.stats` 中指数分布的参数化方式。`scipy` 使用的 `scale` 参数与我们理论学习中的 $\lambda$ 是倒数关系，即 `scale` $= 1/\lambda$。
    *   **计算**: 已知 `scale` = 4，所以 $\lambda = 1 / \text{scale} = 1 / 4 = 0.25$。

3.  **答案: 正确**
    *   **思路**: 这是一个非常重要的概念点。概率的取值必须在 [0, 1] 之间，但 **概率密度 (PDF) 不是概率**，它仅仅衡量在该点附近的概率集中程度。PDF 的值可以大于1，只要保证其在整个定义域上的积分（总面积）等于1即可。

4.  **答案: 0.5**
    *   **思路**: 中位数是定义上的 50% 分位数，它将概率分布精确地分为两半。

5.  **答案**:
    *   **区别**:
        *   **期望值 (Expected Value)** 是一个 **理论** 概念，属于概率分布本身的一个性质。它是在上帝视角下，一个随机变量所有可能取值的加权平均，是“总体”的均值。
        *   **样本均值 (Sample Mean)** 是一个 **实践** 概念，是根据我们实际观测到的一组 **有限** 数据计算出来的平均值。它是对总体的部分观测的总结。
    *   **联系**:
        *   **大数定律 (Law of Large Numbers)** 将两者联系起来。该定律指出，随着我们收集的样本数据越来越多（即样本量趋于无穷），样本均值会收敛于（越来越接近）理论上的期望值。因此，在实际工作中，我们使用样本均值来 **估计** 未知的理论期望值。

---

## 5. 分布的深层特征：高阶矩 (Higher Order Moments)

均值（期望值）描述了分布的“中心”在哪里，但这远远不够。两个分布可以有相同的均值，但形态却截然不同。为了更全面地理解一个随机变量，我们需要引入“矩”的概念。

*   **概念阐释**: **矩 (Moments)** 是衡量一个概率分布形状的特定量化指标。
    *   **$k$-阶原点矩 ($k$-th moment)**: $ \mu_k = E[Y^k] $
    *   **$k$-阶中心矩 ($k$-th central moment)**: $ \mu'_k = E[(Y - \mu_1)^k] $，其中 $ \mu_1 $ 就是均值 $E[Y]$。中心矩衡量的是变量相对于其均值的离散情况。

### 5.1. 第二中心矩：方差 (Variance)

*   **概念阐释**: **方差 (Variance)**，记作 $V[Y]$ 或 $\sigma^2$，是第二中心矩。它衡量的是一个随机变量的取值与其均值的偏离程度的平方的期望值。通俗地说，方差描述了数据的 **离散程度 (spread)** 或 **波动性 (volatility)**。方差越大，数据点越分散；方差越小，数据点越集中。

*   **公式呈现**:
    $$
    V[Y] = \sigma^2 = E[(Y - E[Y])^2]
    $$
    一个更常用的计算公式是：
    $$
    V[Y] = E[Y^2] - (E[Y])^2
    $$

*   **标准差 (Standard Deviation)**: 方差的平方根，记作 $\sigma$，量纲与随机变量本身相同，更易于解释。

### 5.2. 第三中心矩：偏度 (Skewness)

*   **概念阐释**: **偏度 (Skewness)** 是衡量概率分布 **不对称性 (asymmetry)** 的指标。
    *   **零偏度 (Zero Skewness)**: 分布是完全对称的，例如正态分布（高斯分布）。
    *   **正偏度 (Positive Skewness / Right-skewed)**: 分布的“尾巴”在右边更长。这意味着存在一些较大的极端正值，把均值“向右拉”，使得 **均值 > 中位数**。
    *   **负偏度 (Negative Skewness / Left-skewed)**: 分布的“尾巴”在左边更长。这意味着存在一些较小的极端负值，把均值“向左拉”，使得 **均值 < 中位数**。

*   **公式呈现** (标准化后):
    $$
    \text{Skewness} = \frac{\mu'_3}{\sigma^3} = \frac{E[(Y - \mu)^3]}{(\sigma^2)^{3/2}}
    $$

### 5.3. 第四中心矩：峰度 (Kurtosis)

*   **概念阐释**: **峰度 (Kurtosis)** 衡量的是分布尾部的 **厚度 (thickness)**，或者说，极端值出现的频率。
    *   **正态分布的峰度**: 标准正态分布的峰度值为 `3`。这通常被用作一个基准。
    *   **高尖峰 (Leptokurtic)**: 峰度 > 3。分布比正态分布更“尖峭”，同时尾部更“厚”。这意味着出现极端值（非常大或非常小）的概率要高于正态分布。金融资产回报率通常都具有这种 **“尖峰厚尾” (fat tails)** 的特征。
    *   **低阔峰 (Platykurtic)**: 峰度 < 3。分布比正态分布更“平坦”，尾部更“薄”，极端值出现的概率较低。

*   **公式呈现** (标准化后):
    $$
    \text{Kurtosis} = \frac{\mu'_4}{\sigma^4} = \frac{E[(Y - \mu)^4]}{(\sigma^2)^2}
    $$
*   **超额峰度 (Excess Kurtosis)**: 在实际应用中，尤其是在 `scipy` 等软件库里，通常报告的是超额峰度，其定义为 `Kurtosis - 3`。这样，正态分布的超额峰度就是 `0`，方便我们直接判断尾部厚度。

### 5.4. Python 中的矩计算

我们可以使用 `scipy.stats` 方便地计算一个分布的理论四阶矩。

```python
from scipy import stats

# 1. 指数分布 (λ=0.5, scale=2)
# moments='mvsk' 分别代表 Mean, Variance, Skewness, Kurtosis
m, v, s, k = stats.expon.stats(scale=2, moments='mvsk')
# 结果: (mean=2.0, variance=4.0, skewness=2.0, kurtosis=6.0)
# 解读: 正偏度(2.0>0)，厚尾(6.0>3)。

# 2. 标准正态分布
m, v, s, k = stats.norm.stats(moments='mvsk')
# 结果: (mean=0.0, variance=1.0, skewness=0.0, kurtosis=0.0)
# 注意: scipy.stats.norm 返回的是超额峰度 (Excess Kurtosis)，所以为0。
```

---

## 6. 融会贯通：检验 Black-Scholes 模型的假设 (Bringing Data and Probability Together)

现在，我们将本周的所有知识点应用到一个真实的金融问题上。

*   **背景理论**: 著名的 **Black-Scholes 期权定价模型** 是现代金融的基石之一。它的一个核心假设是：资产价格服从 **几何布朗运动 (Geometric Brownian motion)**。这个假设经过数学推导，其直接结论就是：**该资产的对数回报率 (log returns) 必须服从正态分布 (Normal/Gaussian distribution)**。

*   **我们的任务**: 利用我们手头的 CBA 股票数据，来检验这个核心假设是否成立。换句话说，**CBA 股票的对数回报率真的是正态分布的吗？**

### 6.1. 步骤1：初步可视化 - 直方图

我们可以绘制对数回报率的直方图，直观地看一下它的形状是否像正态分布的“钟形曲线”。

```python
# 假设 cba['logReturns'] 已经计算好了
plt.hist(cba['logReturns'].dropna(), bins=30) # dropna() 用于移除缺失值
plt.show()
```
从直方图（幻灯片第59页）来看，数据大致呈钟形，中间高两边低，似乎与正态分布相似。但这只是初步印象，我们需要更严格的量化检验。

### 6.2. 步骤2：检验样本矩

如果数据真的来自一个正态分布，那么它的样本偏度和样本峰度应该接近正态分布的理论值。
*   **正态分布理论值**: 偏度 = 0，（超额）峰度 = 0。

```python
# .dropna() 在这里也很重要，因为回报率的第一个值是NaN
log_returns = cba['logReturns'].dropna()

# 计算样本偏度
sample_skew = stats.skew(log_returns)
# 结果: 约为 -0.066 (非常接近0)

# 计算样本超额峰度
sample_kurtosis = stats.kurtosis(log_returns) 
# 结果: 约为 7.80 (远大于0!)
```

*   **结果解读**:
    *   **偏度**: 样本偏度 `-0.066` 非常接近 `0`，这与正态分布的对称性假设是吻合的。
    *   **峰度**: 样本超额峰度 `7.80` 远远大于 `0`！这表明 CBA 的回报率分布具有非常显著的 **“尖峰厚尾”** 特征。相比于正态分布，它更容易产生极端的回报（无论是大涨还是大跌）。这是对正态分布假设的一个有力反驳。

### 6.3. 步骤3：进行统计检验 - Jarque-Bera 测试

为了得到一个统计上更严谨的结论，我们使用 **Jarque-Bera (JB) 检验**。

*   **概念阐释**: JB 检验是一种专门用于判断样本数据是否符合正态分布的统计测试。它构建了一个同时利用样本偏度和样本峰度的统计量。
    *   **原假设 (Null Hypothesis, $H_0$)**: 数据来自于一个正态分布。
    *   **检验逻辑**: 如果原假设成立，JB 统计量应该很小。如果 JB 统计量很大（意味着样本偏离正态分布很远），我们就可以拒绝原假设。
    *   **p-value**: 这是做出决策的关键。p-value 表示“如果原假设为真，我们观测到当前这样（或更极端）的样本数据的概率”。通常，我们设定一个显著性水平（如 0.05）。
        *   如果 **p-value < 0.05**，我们认为这是一个小概率事件，因此我们有足够的信心 **拒绝原假设**。
        *   如果 **p-value > 0.05**，我们 **不能拒绝原假设**。

*   **Python 实现**:
    ```python
    from scipy.stats import jarque_bera

    # 进行 JB 检验
    jb_statistic, p_value = jarque_bera(log_returns)

    # 打印 p-value
    print(p_value) 
    # 结果: 0.0
    ```

### 6.4. 最终结论 (Conclusions)

*   **我们的 p-value 结果是 `0.0`，这远小于 `0.05`。**
*   因此，我们以极高的置信度 **拒绝原假设**。
*   **结论**: CBA 股票的对数回报率 **不服从** 正态分布。
*   **引申**: Black-Scholes 模型的这个核心假设在现实世界中并不成立。这为后续课程中学习更复杂的模型（例如能够捕捉“尖峰厚尾”特征的 GARCH 模型等）埋下了伏笔。这也完美地诠释了本课程的核心思想：**用真实数据去检验金融理论**。

---

## 10. 模块 5-6 练习题

### 10.1 题目

1.  (选择题) 分析师发现一支股票的日回报率分布具有 `2.5` 的 **超额峰度**。与正态分布相比，这意味着什么？
    A. 该股票回报率的波动性更小。
    B. 该股票回报率出现极端下跌的概率更低。
    C. 该股票回报率的分布是左偏的。
    D. 该股票回报率出现极端波动的概率更高（“厚尾”）。

2.  (选择题) 在进行 Jarque-Bera 检验时，你得到的 p-value 是 `0.12`。在 5% 的显著性水平下，你应该得出什么结论？
    A. 拒绝原假设，数据不是正态分布。
    B. 拒绝原假设，数据是正态分布。
    C. 不能拒绝原假设，我们没有足够证据表明数据不是正态分布。
    D. 接受原假设，数据肯定是正态分布。

3.  (判断题) 两个不同的概率分布不可能拥有完全相同的均值和方差。

4.  (填空题) 在 Black-Scholes 模型的基本假设下，股票的 `___________` (填“简单回报率”或“对数回报率”) 服从正态分布。

5.  (简答题) "尖峰厚尾" (Leptokurtosis and Fat Tails) 是金融资产回报率的一个普遍特征。请问这个特征对于风险管理者来说意味着什么？为什么它很重要？

### 10.2 解析

1.  **答案: D**
    *   **思路**: 超额峰度 = 峰度 - 3。超额峰度 `2.5` 意味着真实峰度是 `5.5`，这远大于正态分布的峰度 `3`。高峰度意味着“厚尾”，即极端值（无论正负）出现的概率比正态分布预测的要高。

2.  **答案: C**
    *   **思路**: 检验的决策规则是：如果 p-value < 显著性水平（这里是0.05），则拒绝原假设。我们的 p-value 是 `0.12`，大于 `0.05`。因此，我们没有足够的统计证据来拒绝“数据是正态分布”这一原假设。注意，统计检验中我们从不说“接受原假设”，因为没能证明它是错的，不代表它就一定是正确的。

3.  **答案: 错误**
    *   **思路**: 均值和方差只是描述分布的前两个矩。很多不同的分布都可以拥有相同的均值和方差，但它们的偏度和峰度可能完全不同，从而导致分布形状的巨大差异。例如，可以构建一个学生t分布，使其方差与某个正态分布相同，但t分布的峰度会更高（尾部更厚）。

4.  **答案: 对数回报率**
    *   **思路**: 这是 Black-Scholes 模型的一个核心数学推论。模型假设价格过程，而其直接后果是**对数回报率**服从正态分布，而不是简单回报率。

5.  **答案**:
    *   **风险管理意义**: “尖峰厚尾”意味着 **传统的、基于正态分布假设的风险模型（如某些VaR模型）会严重低估极端风险**。
    *   **重要性**: 一个假设正态分布的风险模型可能会告诉你，“百年一遇”的金融危机（例如，回报率为-10%）可能真的要100年才发生一次。但如果真实分布是“厚尾”的，这种级别的危机可能每10-20年就会发生一次。对于风险管理者来说，这种差异是致命的。如果他们基于错误的模型来设定资本储备或风险限额，当真正的危机来临时，机构可能会因为准备不足而面临破产。因此，正确识别并为“厚尾”风险建模，是现代风险管理的核心挑战之一。

***

## 11. 更多练习题 (More Practice Questions)

### 11.1 原始练习题 (Source Material Questions)

1.  The number of customers arriving at a bubble tea shop per minute is modeled by a geometric distribution with a success probability `p = 0.25`. What is the probability that there are 4 or fewer customer arrivals in the next minute?

2.  The time until a critical server fails, measured in hours, follows an exponential distribution with a rate parameter `λ = 0.04`. What is the probability that the server fails in less than 10 hours?

3.  For a continuous random variable Y following an exponential distribution with parameter `λ = 0.8`, what is the value of its Probability Density Function (PDF) at `y = 0.5`?

4.  For any continuous random variable (e.g., time, return rates), what is the theoretical probability that it takes on a single, specific value, such as `Pr(Y = 1.2345)`?

5.  If a statistical analysis of daily stock returns reveals that the mean return is `0.05%` while the median return is `0.01%`, what does this difference imply about the shape of the return distribution?

### 11.2 原创练习题 (Original Practice Questions)

**Conceptual Questions**

1.  Why do financial analysts often prefer to model asset returns rather than asset prices directly?
2.  Which Python library is primarily used for numerical operations like logarithms and exponentiation, and which one is used for handling structured data like time series?
3.  Explain the concept of "volatility clustering" that is often observed in financial return series.
4.  A random variable has a Probability Density Function (PDF) value of `f(5) = 1.8`. Is this possible? Explain why or why not.
5.  What is the primary difference between a Probability Mass Function (PMF) and a Probability Density Function (PDF)?
6.  The 5% quantile of a portfolio's daily return distribution is -3.2%. How would you interpret this value in the context of Value at Risk (VaR)?
7.  If the Jarque-Bera test on a series of log returns yields a p-value of 0.001, what conclusion should be drawn regarding the normality of the returns?

**Calculation Questions**

8.  An asset is priced at ￥120 on Day 1, ￥126 on Day 2, and ￥122.85 on Day 3. Calculate the two-day simple return from Day 1 to Day 3.
9.  Using the prices from the previous question, calculate the two-day log return from Day 1 to Day 3.
10. A stock has a log return of 1.5% on Monday and -0.8% on Tuesday. What is the total log return over the two-day period?
11. A continuous random variable Y has a Cumulative Distribution Function (CDF) defined as $ F(y) = 1 - 1/y^2 $ for $ y \ge 1 $. What is the median of this distribution?
12. The expected value of a random variable Y is `E[Y] = 10` and the expected value of its square is `E[Y^2] = 116`. What is the variance of Y?
13. Let X be a random variable representing the return of a stock with `E[X] = 0.02` and `V(X) = 0.05`. You create a new portfolio `P = 50 + 10X`. What are the expected value and variance of P?
14. The theoretical skewness of a distribution is -1.8. Does this distribution have a longer tail on the right or the left? Are extreme negative values more or less likely than extreme positive values?
15. In a `pandas` DataFrame named `stock_data`, which line of code correctly calculates the daily log returns from the 'Adj Close' price column?
    A) `np.log(stock_data['Adj Close'] / stock_data['Adj Close'].shift(-1))`
    B) `stock_data['Adj Close'] - stock_data['Adj Close'].shift(1)`
    C) `np.log(stock_data['Adj Close']) - np.log(stock_data['Adj Close'].shift(1))`
    D) `np.log(stock_data['Adj Close'].diff())`

***

## 12. 练习题答案 (Practice Question Answers)

### 12.1 原始练习题答案

1.  **题号与核心概述**: 1 (几何分布CDF计算)
    **答案**: 0.6836
    **解析**: 设几何分布的取值集合为 $\{1,2,\ldots\}$，表示“在第 $y$ 次到达时第一次成功”。累积分布函数为 $F(y)=1-(1-p)^y$。
    *   参数: `p = 0.25`, `y = 4`。
    *   计算: $F(4) = 1 - 0.75^4 = 1 - 0.31640625 \approx 0.6836$。
    *   也可以直接对 PMF 求和：$P(Y=1)+P(Y=2)+P(Y=3)+P(Y=4)=0.25+0.1875+0.140625+0.105469=0.6836$。
    *   **备注**: 若使用“失败次数”定义（支持 $\{0,1,\ldots\}$），只需将指数替换为 $k+1$ 即可，思路相同。

2.  **题号与核心概述**: 2 (指数分布CDF计算)
    **答案**: 0.3297
    **解析**: 本题求解指数分布的累积概率 `Pr(Y < 10)`。指数分布的CDF公式为 $ F(y) = 1 - e^{-\lambda y} $。
    *   参数: `λ = 0.04`, `y = 10`。
    *   计算: $ F(10) = 1 - e^{-0.04 \times 10} = 1 - e^{-0.4} \approx 1 - 0.6703 = 0.3297 $。

3.  **题号与核心概述**: 3 (指数分布PDF值计算)
    **答案**: 0.5363
    **解析**: 本题求解指数分布在某一点的概率密度值。PDF公式为 $ f(y) = \lambda e^{-\lambda y} $。
    *   参数: `λ = 0.8`, `y = 0.5`。
    *   计算: $ f(0.5) = 0.8 \times e^{-0.8 \times 0.5} = 0.8 \times e^{-0.4} \approx 0.8 \times 0.6703 = 0.5363 $。

4.  **题号与核心概述**: 4 (连续变量单点概率)
    **答案**: 0
    **解析**: 对于任何连续型随机变量，其在任意一个精确数值点上的概率都为零。概率是通过对概率密度函数（PDF）在一个区间上进行积分（求面积）来定义的。一个点的宽度为零，因此其下的面积也为零。

5.  **题号与核心概述**: 5 (均值与中位数关系)
    **答案**: 这意味着该回报率的分布是 **正偏（右偏）** 的 (Positively / Right-skewed)。
    **解析**: 当分布不对称时，极端值会影响均值。均值 (`0.05%`) 大于中位数 (`0.01%`)，表明存在一些较大的正回报（极端收益），这些值将均值向右侧（正方向）拉动，而中位数不受这些极端值的影响。

### 12.2 原创练习题答案

6.  **题号与核心概述**: 1 (回报率 vs 价格)
    **答案**: 因为价格序列通常是 **非平稳的 (non-stationary)**，表现出趋势性，其均值和方差随时间变化。而回报率序列通常是 **平稳的 (stationary)**，在零附近波动。大多数标准时间序列模型都要求输入数据是平稳的。

7.  **题号与核心概述**: 2 (Python库识别)
    **答案**: **Numpy** 用于数值运算，**Pandas** 用于处理结构化数据。

8.  **题号与核心概述**: 3 (波动率聚集)
    **答案**: 这是指金融回报率序列中，大的价格波动（无论正负）倾向于集中出现，形成“波动集群”；同样，小的价格波动也倾向于聚集在一起。图形上表现为一段时间内波动剧烈，随后一段时间内又相对平稳。

9.  **题号与核心概述**: 4 (PDF值大于1)
    **答案**: **是的，这是可能的**。
    **解析**: 概率密度函数（PDF）的值本身不是概率，它衡量的是在该点附近的概率密度。概率是通过对PDF曲线下的面积积分得到的。唯一的要求是PDF在整个定义域上的总积分（总面积）必须等于1。只要满足这个条件，PDF在某些点上的值完全可以大于1。

10. **题号与核心概述**: 5 (PMF vs PDF)
    **答案**: PMF（概率质量函数）用于 **离散** 随机变量，它直接给出变量取某个精确值的 **概率**。PDF（概率密度函数）用于 **连续** 随机变量，它给出变量在某点附近的 **概率密度**，其值本身不是概率。

11. **题号与核心概述**: 6 (VaR解读)
    **答案**: 这个值意味着，在正常的市场条件下，我们有95%的信心，该投资组合在一天内的损失 **不会超过3.2%**。或者说，该投资组合一天内损失超过3.2%的概率是5%。

12. **题号与核心概述**: 7 (JB检验结果解读)
    **答案**: p-value为0.001，远小于常用的显著性水平（如0.05）。因此，我们应该 **拒绝原假设**，结论是该对数回报率序列 **不服从正态分布**。

13. **题号与核心概述**: 8 (多期简单回报率)
    **答案**: 2.375%
    **解析**: 多期简单回报率可以直接用期初和期末价格计算。
    *   公式: $ R_t[k] = (P_t / P_{t-k}) - 1 $
    *   计算: $ R = (￥122.85 / ￥120) - 1 = 1.02375 - 1 = 0.02375 $，即 2.375%。

14. **题号与核心概述**: 9 (多期对数回报率)
    **答案**: 2.347%
    **解析**: 多期对数回报率同样可以直接用期初和期末价格的对数计算。
    *   公式: $ r_t[k] = \ln(P_t) - \ln(P_{t-k}) $
    *   计算: $ r = \ln(122.85) - \ln(120) \approx 4.81109 - 4.78749 = 0.02360 $，即 2.347%。

15. **题号与核心概述**: 10 (对数回报率可加性)
    **答案**: 0.7%
    **解析**: 对数回报率的核心优势是可加性。多期的总对数回报率等于各单期对数回报率之和。
    *   计算: 总回报率 = $ 1.5\% + (-0.8\%) = 0.7\% $。

16. **题号与核心概述**: 11 (从CDF求中位数)
    **答案**: $\sqrt{2} \approx 1.414$
    **解析**: 中位数是使CDF等于0.5的那个y值。
    *   方程: $ F(y) = 1 - 1/y^2 = 0.5 $
    *   求解: $ 1/y^2 = 0.5 \implies y^2 = 2 \implies y = \sqrt{2} $。

17. **题号与核心概述**: 12 (利用期望计算方差)
    **答案**: 16
    **解析**: 方差的计算公式为 $ V[Y] = E[Y^2] - (E[Y])^2 $。
    *   代入: $ V[Y] = 116 - (10)^2 = 116 - 100 = 16 $。

18. **题号与核心概述**: 13 (期望与方差的性质)
    **答案**: $ E[P] = 50.2 $, $ V(P) = 5 $
    **解析**:
    *   **期望**: $ E[a + bX] = a + bE[X] $。$ E[P] = 50 + 10 \times E[X] = 50 + 10 \times 0.02 = 50.2 $。
    *   **方差**: $ V[a + bX] = b^2V(X) $。$ V(P) = 10^2 \times V(X) = 100 \times 0.05 = 5 $。

19. **题号与核心概述**: 14 (偏度解读)
    **答案**: 分布有一个更长的 **左尾** (left tail)。这意味着 **极端负值** 比极端正值 **更可能** 出现。

20. **题号与核心概述**: 15 (Pandas代码)
    **答案**: C
    **解析**: 对数回报率的公式是 $ r_t = \ln(P_t) - \ln(P_{t-1}) $。
    *   `stock_data['Adj Close']` 对应 $P_t$。
    *   `stock_data['Adj Close'].shift(1)` 对应 $P_{t-1}$。
    *   `np.log()` 用于计算自然对数。
    *   因此，选项 C 正确实现了这个公式。

---

