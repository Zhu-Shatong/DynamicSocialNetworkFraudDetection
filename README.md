# 🚀 DynamicSocialNetworkFraudDetection 🚀

🛡️**DynamicSocialNetworkFraudDetection** (动态社交网络反欺诈检测) 专注于利用图神经网络在动态社交网络中进行金融欺诈检测的前沿技术。

🔍 **项目概述**

- **数据集应用**：项目通过分析企业在不同时间段的业务数据集，构建了一个全连通的有向动态图。这个图深刻地描绘了用户之间的复杂关系，其中节点代表注册用户，有向边表示紧急联系人关系。
- **技术创新**：我们利用包括GAT、GraphSAGE及新型的GEARSage在内的图神经网络模型，来分析用户间的复杂互联关系，并预测潜在的欺诈活动。
- **目标与成就**：我们的目标是通过深入理解社交网络的演化特性，并应用最先进的机器学习技术，来增强金融欺诈检测的能力。我们的工作不仅提升了欺诈检测的准确性，也为理解复杂社交网络结构提供了新的视角。

🛡️**DynamicSocialNetworkFraudDetection** is an innovative project that zeroes in on detecting financial fraud within dynamic social networks through cutting-edge graph neural network technology.

🔍 **Project Overview**

- **Dataset Utilization**: By leveraging a comprehensive dataset gathered from various business operations over time, we have developed a fully connected directed dynamic graph that intricately represents the relationships between users. In this graph, nodes symbolize registered users, while directed edges signify emergency contact relationships.
- **Technological Innovation**: We employ advanced graph neural network models, including but not limited to GAT, GraphSAGE, and the novel GEARSage, to dissect the complex interconnections between users and predict fraudulent activities.
- **Goals & Achievements**: Our aim is to enhance fraud detection capabilities by deeply understanding the evolving nature of social networks and applying state-of-the-art machine learning techniques. Our efforts have not only improved the accuracy of fraud detection but also offered new insights into understanding the complex structures of social networks.



## 📈 模型比较

| 算法名称                    | 描述                                                         | 特点                                                         | 提交分数 |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| GAT                         | 图注意力网络（Graph Attention Network），通过学习节点之间的注意力权重，更好地捕捉社交网络中的复杂关系。 | 侧重于节点特征的注意力机制，未直接处理边属性或时间信息，可能在捕捉边相关的复杂动态特性方面不足。 | 72.95    |
| GAT (NeighborSampler)       | 图注意力网络的变种，采用邻居采样策略以降低计算复杂度，适用于大规模图数据。 | 通过邻居采样提高计算效率，保持对网络结构的敏感性，有助于识别和预测潜在的欺诈行为。 | 73.29    |
| GraphSAGE                   | 图采样与聚合网络（Graph Sample and Aggregate），通过从节点的邻居中学习并聚合信息，更好地理解和表示网络中的节点。 | 使用基于邻居的聚合策略，捕捉节点间的局部关系，适合处理动态社交网络数据。 | 77.27    |
| GraphSAGE (NeighborSampler) | GraphSAGE的变种，同样采用邻居采样策略，针对大规模图数据进行优化。 | 在由特定子图上操作，通过聚合局部邻域信息以降低计算复杂度，保留模型对网络结构特征的捕捉能力。 | 78.10    |
| GearSage                    | 新型图神经网络模型，是对GraphSAGE的扩展，更有效地处理带有边属性和时间特征的图数据。 | 集成边属性和时间特征，通过时间编码器捕捉动态性，提供更丰富的信息以精准捕捉和理解网络中的复杂交互模式。 | 80.75    |

[![Stargazers repo roster for @Zhu-Shatong/DynamicSocialNetworkFraudDetection](https://reporoster.com/stars/dark/Zhu-Shatong/DynamicSocialNetworkFraudDetection)](https://github.com/Zhu-Shatong/DynamicSocialNetworkFraudDetection/stargazers)

## 🌟 支持我们！

如果你对 **DynamicSocialNetworkFraudDetection** 项目感兴趣，请给我们一个星标！🌟

你的支持是我们不断进步和创新的最大动力！

