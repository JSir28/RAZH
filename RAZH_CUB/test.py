import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def hamming_distance(h1, h2):
    """
    计算两个哈希码的汉明距离。假设 h1 和 h2 是二进制字符串或者 NumPy 数组。
    """
    # 将二进制哈希码转换为整数
    h1_int = int(''.join(h1.astype(str)), 2)  # h1 是二进制数组，转换为字符串再转为整数
    h2_int = int(''.join(h2.astype(str)), 2)  # h2 是二进制数组，转换为字符串再转为整数

    # 计算汉明距离
    return bin(h1_int ^ h2_int).count('1')


def compute_confusion_matrix(queries, query_labels, retrievals, retrieval_labels, num_classes):
    """
    计算混淆矩阵。
    queries: 查询哈希码数组，形状为 (1000, N)，N是哈希码的长度
    query_labels: 查询标签，形状为 (1000,)
    retrievals: 检索集哈希码数组，形状为 (32322, N)
    retrieval_labels: 检索集标签，形状为 (32322,)
    num_classes: 类别数
    """
    confusion_matrix = np.zeros((num_classes, num_classes))

    # 遍历每个查询类别和检索集类别对
    for c_q in range(num_classes):
        print(c_q)
        for c_r in range(num_classes):
            # 获取属于查询类别 c_q 和检索类别 c_r 的样本索引
            query_indices = np.where(query_labels == c_q)[0]
            retrieval_indices = np.where(retrieval_labels == c_r)[0]

            # 累加所有查询和检索样本的哈希距离
            total_distance = 0
            count = 0
            for q_idx in query_indices:
                for r_idx in retrieval_indices:
                    total_distance += hamming_distance(queries[q_idx], retrievals[r_idx])
                    count += 1

            # 计算平均哈希距离
            if count > 0:
                avg_distance = total_distance / count
            else:
                avg_distance = 0

            # 将计算结果放入混淆矩阵
            confusion_matrix[c_q, c_r] = avg_distance

    return confusion_matrix


# 假设你已经有了查询样本、查询标签、检索集样本和检索集标签
queries = np.random.randint(0, 2, size=(1000, 64))  # 1000个查询样本，每个64位哈希码
query_labels = np.random.randint(0, 50, size=1000)  # 1000个查询样本的标签（假设有10类）
retrievals = np.random.randint(0, 2, size=(32322, 64))  # 32322个检索集样本，每个64位哈希码
retrieval_labels = np.random.randint(0, 50, size=32322)  # 32322个检索集样本的标签（假设有10类）

# 计算混淆矩阵
conf_matrix = compute_confusion_matrix(queries, query_labels, retrievals, retrieval_labels, num_classes=50)

# 设置绘图的大小
plt.figure(figsize=(10, 8))

# 绘制热力图
sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True,
            xticklabels=[f"Class {i}" for i in range(10)],
            yticklabels=[f"Class {i}" for i in range(10)])

# 添加标题
plt.title("Confusion Matrix - Hash Clustering Average Distance")

# 显示图形
plt.tight_layout()

# 保存图形
plt.savefig("confusion_matrix.png", dpi=300)

# # 展示图形
# plt.show()
