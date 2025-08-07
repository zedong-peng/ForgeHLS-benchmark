import json
import numpy as np
import argparse
import os
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import Counter
import re
from datetime import datetime
from scipy import sparse


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self):
        # C++关键字
        self.cpp_keywords = {
            'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
            'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
            'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static',
            'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while',
            'class', 'private', 'protected', 'public', 'friend', 'inline', 'template',
            'virtual', 'bool', 'true', 'false', 'namespace', 'using', 'try', 'catch',
            'throw', 'new', 'delete', 'this', 'operator', 'const_cast', 'dynamic_cast',
            'reinterpret_cast', 'static_cast', 'typeid', 'typename', 'explicit', 'mutable'
        }
        
        # HLS关键字
        self.hls_keywords = {
            'pragma', 'hls', 'pipeline', 'unroll', 'dataflow', 'interface', 'resource',
            'allocation', 'array_partition', 'array_map', 'dependence', 'inline',
            'loop_tripcount', 'loop_merge', 'loop_flatten', 'occurrence', 'protocol',
            'stable', 'stream', 'axis', 'ap_int', 'ap_uint', 'ap_fixed', 'ap_ufixed',
            'ap_shift_reg', 'ap_fifo', 'ap_memory', 'ap_bus', 'ap_none', 'ap_vld',
            'ap_ack', 'ap_hs', 'ap_ovld', 'bram', 'uram', 'lutram', 'dsp', 'ff'
        }
        
        # 算法相关关键字
        self.algorithm_keywords = {
            'sort', 'search', 'find', 'merge', 'split', 'partition', 'filter',
            'map', 'reduce', 'scan', 'prefix', 'suffix', 'matrix', 'vector',
            'array', 'list', 'tree', 'graph', 'hash', 'heap', 'stack', 'queue',
            'dfs', 'bfs', 'dijkstra', 'floyd', 'kmp', 'fft', 'convolution'
        }
    
    @staticmethod
    def extract_structural_features(design: Dict) -> List[float]:
        """提取结构特征"""
        source_code = design.get('source_code', [])
        
        # 基础统计
        total_lines = 0
        total_chars = 0
        file_count = len(source_code)
        
        # 语法特征计数
        function_count = 0
        loop_count = 0
        include_count = 0
        
        # 文件类型统计
        file_types = Counter()
        
        for file_info in source_code:
            content = file_info.get('file_content', '')
            file_name = file_info.get('file_name', '')
            
            lines = content.split('\n')
            total_lines += len(lines)
            total_chars += len(content)
            
            # 文件扩展名
            if '.' in file_name:
                ext = file_name.split('.')[-1].lower()
                file_types[ext] += 1
            
            # 语法特征
            function_count += len(re.findall(r'\w+\s*\([^)]*\)\s*\{', content))
            loop_count += len(re.findall(r'\b(for|while)\s*\(', content))
            include_count += len(re.findall(r'#include', content))
        
        # 计算派生特征
        avg_lines_per_file = total_lines / file_count if file_count > 0 else 0
        code_density = total_chars / total_lines if total_lines > 0 else 0
        
        return [
            file_count,
            total_lines,
            avg_lines_per_file,
            code_density,
            function_count,
            loop_count,
            include_count,
            len(file_types),  # 文件类型多样性
            file_types.get('cpp', 0),
            file_types.get('h', 0),
            file_types.get('c', 0),
        ]
    
    def extract_code_text(self, design: Dict) -> str:
        """提取代码文本用于语义分析 - 改进版本"""
        texts = []
        
        for file_info in design.get('source_code', []):
            content = file_info.get('file_content', '')
            
            # 1. 移除注释 - 改进正则表达式
            content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL | re.MULTILINE)
            
            # 2. 移除字符串字面量，但保留结构
            content = re.sub(r'"[^"]*"', '"STRING"', content)
            content = re.sub(r"'[^']*'", "'CHAR'", content)
            
            # 3. 标准化数字
            content = re.sub(r'\b\d+\b', 'NUM', content)
            content = re.sub(r'\b\d+\.\d+\b', 'FLOAT', content)
            
            # 4. 提取和标记化
            tokens = re.findall(r'\w+|[^\w\s]', content)
            
            # 5. 规范化tokens
            normalized_tokens = []
            for token in tokens:
                token = token.lower().strip()
                if not token or token.isspace():
                    continue
                
                # 跳过单字符标点符号（除了重要的操作符）
                if len(token) == 1 and token in '.,;(){}[]':
                    continue
                
                # 保留重要操作符
                if token in ['++', '--', '&&', '||', '==', '!=', '<=', '>=', '->', '<<', '>>', '+=', '-=', '*=', '/=']:
                    normalized_tokens.append(f'OP_{token}')
                elif token == 'num':
                    normalized_tokens.append('NUM')
                elif token == 'float':
                    normalized_tokens.append('FLOAT')
                elif token == 'string':
                    normalized_tokens.append('STRING')
                elif token == 'char':
                    normalized_tokens.append('CHAR')
                elif token in self.hls_keywords:
                    normalized_tokens.append(f'HLS_{token.upper()}')
                elif token in self.cpp_keywords:
                    normalized_tokens.append(f'CPP_{token.upper()}')
                elif token in self.algorithm_keywords:
                    normalized_tokens.append(f'ALG_{token.upper()}')
                else:
                    # 保留标识符，但进行一些规范化
                    if token.isalpha() and len(token) > 1:
                        normalized_tokens.append(token)
            
            # 6. 重建文本，保持一定的结构信息
            processed_text = ' '.join(normalized_tokens)
            texts.append(processed_text)
        
        return ' '.join(texts)
    
    def extract_enhanced_structural_features(self, design: Dict) -> List[float]:
        """提取增强的结构特征"""
        source_code = design.get('source_code', [])
        
        # 基础统计
        total_lines = 0
        total_chars = 0
        file_count = len(source_code)
        
        # 语法特征计数
        function_count = 0
        loop_count = 0
        include_count = 0
        pragma_count = 0
        class_count = 0
        template_count = 0
        
        # 复杂度指标
        nested_loop_count = 0
        conditional_count = 0
        pointer_usage = 0
        array_usage = 0
        
        # HLS特征
        hls_directive_count = 0
        ap_type_count = 0
        
        # 文件类型统计
        file_types = Counter()
        
        for file_info in source_code:
            content = file_info.get('file_content', '')
            file_name = file_info.get('file_name', '')
            
            lines = content.split('\n')
            total_lines += len(lines)
            total_chars += len(content)
            
            # 文件扩展名
            if '.' in file_name:
                ext = file_name.split('.')[-1].lower()
                file_types[ext] += 1
            
            # 基础语法特征
            function_count += len(re.findall(r'\w+\s*\([^)]*\)\s*\{', content))
            loop_count += len(re.findall(r'\b(for|while)\s*\(', content))
            include_count += len(re.findall(r'#include', content))
            pragma_count += len(re.findall(r'#pragma', content))
            class_count += len(re.findall(r'\bclass\s+\w+', content))
            template_count += len(re.findall(r'\btemplate\s*<', content))
            
            # 复杂度特征
            nested_loop_count += len(re.findall(r'for\s*\([^}]*for\s*\(', content, re.DOTALL))
            conditional_count += len(re.findall(r'\b(if|switch)\s*\(', content))
            pointer_usage += len(re.findall(r'\*\w+|\w+\*|\->', content))
            array_usage += len(re.findall(r'\w+\[\w*\]', content))
            
            # HLS特征
            hls_directive_count += len(re.findall(r'#pragma\s+HLS', content, re.IGNORECASE))
            ap_type_count += len(re.findall(r'\bap_(int|uint|fixed|ufixed)', content))
        
        # 计算派生特征
        avg_lines_per_file = total_lines / file_count if file_count > 0 else 0
        code_density = total_chars / total_lines if total_lines > 0 else 0
        complexity_ratio = (nested_loop_count + conditional_count) / total_lines if total_lines > 0 else 0
        hls_ratio = hls_directive_count / total_lines if total_lines > 0 else 0
        
        # 对长尾分布特征应用log1p变换
        return [
            file_count,
            np.log1p(total_lines),  # log变换
            avg_lines_per_file,
            code_density,
            np.log1p(function_count),  # log变换
            np.log1p(loop_count),  # log变换
            include_count,
            pragma_count,
            class_count,
            template_count,
            nested_loop_count,
            conditional_count,
            pointer_usage,
            array_usage,
            hls_directive_count,
            ap_type_count,
            complexity_ratio,
            hls_ratio,
            len(file_types),  # 文件类型多样性
            file_types.get('cpp', 0),
            file_types.get('h', 0),
            file_types.get('c', 0),
            file_types.get('hpp', 0),
        ]


class ClusteringEvaluator:
    """聚类评估器"""
    
    @staticmethod
    def find_optimal_k(features, min_k: int = 3, max_k: int = 50) -> int:
        """寻找最优的K值 - 支持稀疏矩阵"""
        print(f"Evaluating K-Means with k from {min_k} to {max_k}...")
        
        best_score = -1
        best_k = min_k
        scores = []
        
        for k in range(min_k, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                
                if len(set(labels)) < 2:
                    scores.append(0)
                    continue
                
                # 使用轮廓系数评估
                score = silhouette_score(features, labels)
                scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    
            except Exception as e:
                print(f"Error with k={k}: {e}")
                scores.append(0)
        
        print(f"Best k: {best_k} with silhouette score: {best_score:.4f}")
        return best_k


class RepresentativeSelector:
    """代表选择器"""
    
    def __init__(self, target_range: Tuple[int, int] = (80, 120)):
        self.target_min, self.target_max = target_range
    
    def select_representatives(self, features, labels: np.ndarray) -> List[int]:
        """选择代表性样本 - 支持稀疏矩阵"""
        print(f"Selecting representatives (target: {self.target_min}-{self.target_max})")
        
        unique_labels = set(labels)
        cluster_sizes = Counter(labels)
        
        # 计算每个聚类应选择的代表数
        representatives_per_cluster = self._calculate_representatives_per_cluster(
            unique_labels, cluster_sizes
        )
        
        selected_indices = []
        
        # 从每个聚类选择代表
        for label, n_representatives in representatives_per_cluster.items():
            if n_representatives == 0:
                continue
                
            cluster_indices = np.where(labels == label)[0]
            cluster_features = features[labels == label]
            
            cluster_selected = self._select_from_cluster(
                cluster_indices, cluster_features, n_representatives
            )
            selected_indices.extend(cluster_selected)
        
        print(f"Selected {len(selected_indices)} representatives")
        print(f"Per cluster: {representatives_per_cluster}")
        
        return selected_indices
    
    def _calculate_representatives_per_cluster(self, unique_labels: set, 
                                            cluster_sizes: Counter) -> Dict[int, int]:
        """计算每个聚类的代表数 - 改进版本"""
        target_total = (self.target_min + self.target_max) // 2
        total_samples = sum(cluster_sizes.values())
        
        representatives_per_cluster = {}
        
        # 按比例分配，但每个聚类至少1个，最多不超过合理上限
        for label in unique_labels:
            proportion = cluster_sizes[label] / total_samples
            base_count = max(1, int(proportion * target_total))
            
            # 改进：限制最大值，避免大簇过度代表
            max_count = min(cluster_sizes[label], max(1, min(5, cluster_sizes[label] // 2)))
            representatives_per_cluster[label] = min(base_count, max_count)
        
        # 调整到目标范围
        total_assigned = sum(representatives_per_cluster.values())
        
        # 如果超出上限，减少最大的聚类
        while total_assigned > self.target_max:
            largest_cluster = max(representatives_per_cluster.keys(), 
                                key=lambda x: representatives_per_cluster[x])
            if representatives_per_cluster[largest_cluster] > 1:
                representatives_per_cluster[largest_cluster] -= 1
                total_assigned -= 1
            else:
                break
        
        # 如果低于下限，增加较大的聚类
        while total_assigned < self.target_min:
            # 选择还有增长空间的最大聚类
            best_cluster = None
            for label in unique_labels:
                current_count = representatives_per_cluster[label]
                max_possible = min(5, cluster_sizes[label])  # 限制单簇最大代表数
                if current_count < max_possible:
                    if best_cluster is None or current_count > representatives_per_cluster[best_cluster]:
                        best_cluster = label
            
            if best_cluster is not None:
                representatives_per_cluster[best_cluster] += 1
                total_assigned += 1
            else:
                break
        
        return representatives_per_cluster
    
    def _select_from_cluster(self, cluster_indices: np.ndarray, 
                           cluster_features, n_representatives: int) -> List[int]:
        """从单个聚类中选择代表 - 支持稀疏矩阵"""
        n_samples = cluster_features.shape[0]
        
        if n_representatives >= n_samples:
            return cluster_indices.tolist()
        
        # 转换为dense进行计算（只对单个聚类，数据量小）
        if sparse.issparse(cluster_features):
            cluster_features_dense = cluster_features.toarray()
        else:
            cluster_features_dense = cluster_features
        
        if n_representatives == 1:
            # 选择最接近聚类中心的点
            center = np.mean(cluster_features_dense, axis=0)
            distances = np.linalg.norm(cluster_features_dense - center, axis=1)
            best_idx = np.argmin(distances)
            return [cluster_indices[best_idx]]
        
        # 多个代表：先选中心点，再用最远点采样
        selected_indices = []  # 在cluster_indices中的索引
        selected_local_indices = []  # 在cluster_features_dense中的索引
        
        # 1. 选择聚类中心
        center = np.mean(cluster_features_dense, axis=0)
        distances_to_center = np.linalg.norm(cluster_features_dense - center, axis=1)
        center_local_idx = np.argmin(distances_to_center)
        selected_indices.append(cluster_indices[center_local_idx])
        selected_local_indices.append(center_local_idx)
        
        # 2. 最远点采样选择其余点
        for _ in range(n_representatives - 1):
            max_min_distance = -1
            best_local_idx = -1
            best_global_idx = -1
            
            for local_idx in range(n_samples):
                global_idx = cluster_indices[local_idx]
                if global_idx in selected_indices:
                    continue
                
                # 计算到已选点的最小距离
                min_distance = min([
                    np.linalg.norm(cluster_features_dense[local_idx] - cluster_features_dense[selected_local_idx])
                    for selected_local_idx in selected_local_indices
                ])
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_local_idx = local_idx
                    best_global_idx = global_idx
            
            if best_global_idx != -1:
                selected_indices.append(best_global_idx)
                selected_local_indices.append(best_local_idx)
        
        return selected_indices


class AdaptiveClusteringSelector:
    """自适应聚类选择器"""
    
    def __init__(self, target_range: Tuple[int, int] = (80, 120)):
        self.target_range = target_range
        self.designs = []
        self.features = None
        self.cluster_labels = None
        self.optimal_k = None
        self.silhouette_score = None
        
        # 组件
        self.feature_extractor = FeatureExtractor()
        self.clustering_evaluator = ClusteringEvaluator()
        self.representative_selector = RepresentativeSelector(target_range)
    
    def load_data(self, input_file: str) -> bool:
        """加载数据"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                self.designs = json.load(f)
            print(f"Loaded {len(self.designs)} designs from {input_file}")
            return len(self.designs) > 0
        except Exception as e:
            print(f"Error loading data from {input_file}: {e}")
            return False
    
    def extract_features(self, feature_mode: str = "combined"):
        """提取特征 - 改进版本，支持稀疏矩阵"""
        print(f"Extracting features (mode: {feature_mode})...")
        
        # 提取增强的结构特征
        structural_features = []
        code_texts = []
        
        for design in self.designs:
            struct_feat = self.feature_extractor.extract_enhanced_structural_features(design)
            structural_features.append(struct_feat)
            
            code_text = self.feature_extractor.extract_code_text(design)
            code_texts.append(code_text)
        
        # 处理结构特征
        structural_features = np.array(structural_features)
        
        # 使用RobustScaler处理结构特征，抗极端值
        scaler_struct = RobustScaler()
        structural_features_scaled = scaler_struct.fit_transform(structural_features)
        
        if feature_mode == "structural_only":
            self.features = structural_features_scaled
            print(f"Feature shape: {self.features.shape} (structural only)")
            return
        
        # 提取语义特征 - 保持稀疏
        tfidf_vectorizer = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            token_pattern=r'\b\w+\b',
            stop_words=None,
            sublinear_tf=True
        )
        
        try:
            semantic_features_sparse = tfidf_vectorizer.fit_transform(code_texts)
            print(f"TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
            print(f"TF-IDF sparsity: {1 - semantic_features_sparse.nnz / (semantic_features_sparse.shape[0] * semantic_features_sparse.shape[1]):.3f}")
        except Exception as e:
            print(f"TF-IDF failed: {e}, using structural features only")
            self.features = structural_features_scaled
            return
        
        if feature_mode == "semantic_only":
            self.features = semantic_features_sparse
            print(f"Feature shape: {self.features.shape} (semantic only, sparse)")
            return
        
        # 组合特征 - 保持稀疏
        # 1. 转换结构特征为稀疏矩阵
        structural_sparse = sparse.csr_matrix(structural_features_scaled)
        
        # 2. 应用权重
        structural_weight = 1
        structural_sparse = structural_sparse * structural_weight
        
        # 3. 拼接特征
        self.features = sparse.hstack([structural_sparse, semantic_features_sparse]).tocsr()
        
        # 4. 可选：整体缩放到[-1,1]，保持稀疏性
        max_abs_scaler = MaxAbsScaler()
        self.features = max_abs_scaler.fit_transform(self.features)
        
        print(f"Feature shape: {self.features.shape} (combined, sparse)")
        print(f"Final sparsity: {1 - self.features.nnz / (self.features.shape[0] * self.features.shape[1]):.3f}")
    
    def perform_clustering(self):
        """执行聚类"""
        print("Performing K-Means clustering...")
        
        # 根据数据大小确定K的范围
        data_size = len(self.designs)
        min_k = max(3, data_size // 100)
        max_k = min(50, data_size // 10)
        
        # 寻找最优K
        self.optimal_k = self.clustering_evaluator.find_optimal_k(self.features, min_k, max_k)
        
        # 执行聚类
        kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(self.features)
        
        # 计算轮廓系数
        self.silhouette_score = silhouette_score(self.features, self.cluster_labels)
        
        # 统计结果
        cluster_sizes = Counter(self.cluster_labels)
        print(f"Clustering completed: {self.optimal_k} clusters")
        print(f"Silhouette score: {self.silhouette_score:.4f}")
        print(f"Cluster sizes: {dict(cluster_sizes)}")
    
    def select_representatives(self) -> List[int]:
        """选择代表性样本"""
        return self.representative_selector.select_representatives(
            self.features, self.cluster_labels
        )
    
    def visualize_results(self, selected_indices: List[int], output_dir: str):
        """可视化结果 - 支持稀疏矩阵"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # PCA降维可视化 - 处理稀疏矩阵
        pca = PCA(n_components=2)
        if sparse.issparse(self.features):
            features_dense = self.features.toarray()
        else:
            features_dense = self.features
        features_2d = pca.fit_transform(features_dense)
        
        # 聚类可视化 - 使用更柔和的颜色
        scatter = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=self.cluster_labels, cmap='Set3', alpha=0.6, s=10)
        
        # 高亮选中的点 - 使用更美观的标记
        selected_2d = features_2d[selected_indices]
        axes[0].scatter(selected_2d[:, 0], selected_2d[:, 1], 
                       c='darkred', marker='o', s=10, alpha=0.9, 
                       edgecolors='white', linewidth=1.5, label='Selected Representatives')
        
        axes[0].set_title('Clustering Results with Selected Representatives', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=10)
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=10)
        axes[0].legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        axes[0].grid(True, alpha=0.3)
        
        # 聚类大小分布 - 使用更美观的柱状图
        cluster_sizes = Counter(self.cluster_labels)
        labels, sizes = zip(*sorted(cluster_sizes.items()))
        
        # 为每个聚类分配颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        bars = axes[1].bar(range(len(labels)), sizes, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # 添加数值标签
        for i, (bar, size) in enumerate(zip(bars, sizes)):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{size}', ha='center', va='bottom', fontsize=9)
        
        axes[1].set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Cluster ID', fontsize=10)
        axes[1].set_ylabel('Number of Designs', fontsize=10)
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 调整布局和样式
        plt.tight_layout(pad=3.0)
        
        # 保存图片到输出目录
        plot_path = os.path.join(output_dir, 'clustering_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Visualization saved to: {plot_path}")
        plt.close()
    
    def analyze_results(self, selected_indices: List[int]) -> Dict:
        """分析结果"""
        selected_designs = [self.designs[i] for i in selected_indices]
        selected_clusters = self.cluster_labels[selected_indices]
        
        # 聚类覆盖率
        total_clusters = len(set(self.cluster_labels))
        covered_clusters = len(set(selected_clusters))
        
        # 文件类型统计
        file_types = Counter()
        for design in selected_designs:
            for file_info in design.get('source_code', []):
                file_name = file_info.get('file_name', '')
                if '.' in file_name:
                    ext = file_name.split('.')[-1].lower()
                    file_types[ext] += 1
        
        return {
            'selected_count': len(selected_indices),
            'target_range': f"{self.target_range[0]}-{self.target_range[1]}",
            'total_clusters': total_clusters,
            'covered_clusters': covered_clusters,
            'coverage_ratio': covered_clusters / total_clusters,
            'cluster_distribution': dict(Counter(selected_clusters)),
            'file_types': dict(file_types.most_common(5)),
            'optimal_k': self.optimal_k,
            'silhouette_score': self.silhouette_score
        }
    
    def run_comparison_experiment(self):
        """运行对比实验"""
        print("\n" + "="*60)
        print("RUNNING FEATURE COMPARISON EXPERIMENT")
        print("="*60)
        
        modes = ["structural_only", "semantic_only", "combined"]
        results = {}
        
        for mode in modes:
            print(f"\n--- Testing {mode} ---")
            self.extract_features(feature_mode=mode)
            self.perform_clustering()
            
            results[mode] = {
                'optimal_k': self.optimal_k,
                'silhouette_score': self.silhouette_score,
                'feature_shape': self.features.shape,
                'is_sparse': sparse.issparse(self.features)
            }
            
            print(f"Results for {mode}:")
            print(f"  K: {self.optimal_k}")
            print(f"  Silhouette: {self.silhouette_score:.4f}")
            print(f"  Shape: {self.features.shape}")
            print(f"  Sparse: {sparse.issparse(self.features)}")
        
        # 选择最佳模式
        best_mode = max(results.keys(), key=lambda x: results[x]['silhouette_score'])
        print(f"\n--- Best mode: {best_mode} ---")
        
        # 重新运行最佳模式
        self.extract_features(feature_mode=best_mode)
        self.perform_clustering()
        
        return best_mode, results


def create_readme(args, analysis: Dict, input_file: str, output_dir: str, execution_time: float):
    """创建README文件"""
    readme_content = f"""# Clustering-based Deduplication Results

## Experiment Information

**Execution Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Duration**: {execution_time:.2f} seconds

## Input Parameters

- **Input File**: `{input_file}`
- **Output Directory**: `{output_dir}`
- **Target Range**: {analysis['target_range']} designs
- **Method**: K-Means Clustering with Adaptive Representative Selection

## Algorithm Configuration

- **Feature Extraction**: 
  - Structural features (11 dimensions)
  - TF-IDF semantic features (up to 200 dimensions)
  - Feature standardization applied
- **Clustering Method**: K-Means with silhouette score optimization
- **Representative Selection**: Adaptive selection with cluster center + farthest point sampling

## Results Summary

### Dataset Statistics
- **Input Designs**: {len(analysis.get('input_designs', []))} designs
- **Selected Designs**: {analysis['selected_count']} designs
- **Selection Ratio**: {(analysis['selected_count'] / len(analysis.get('input_designs', [1]))) * 100:.2f}%

### Clustering Performance
- **Optimal K**: {analysis['optimal_k']} clusters
- **Silhouette Score**: {analysis['silhouette_score']:.4f}
- **Total Clusters**: {analysis['total_clusters']}
- **Covered Clusters**: {analysis['covered_clusters']}
- **Cluster Coverage**: {analysis['coverage_ratio']:.2%}

### Cluster Distribution
```
{chr(10).join([f"Cluster {k}: {v} designs" for k, v in sorted(analysis['cluster_distribution'].items())])}
```
### File Type Distribution
```
{chr(10).join([f"{ext}: {count} files" for ext, count in analysis['file_types'].items()])}
```

## Output Files

- `after_cluster.json`: Selected representative designs
- `clustering_results.png`: Visualization of clustering results
- `README.md`: This summary report

## Quality Metrics

- **Diversity**: {analysis['coverage_ratio']:.2%} of clusters represented
- **Efficiency**: {(1 - analysis['selected_count'] / len(analysis.get('input_designs', [1]))) * 100:.1f}% reduction in dataset size
- **Clustering Quality**: Silhouette score of {analysis['silhouette_score']:.4f}

## Notes

The clustering-based approach groups similar designs and selects representative samples from each cluster. This ensures both diversity (covering different types of designs) and efficiency (reducing redundancy within similar designs).

The selection process prioritizes:
1. Cluster centers (most representative of each group)
2. Diverse samples within clusters (using farthest point sampling)
3. Proportional representation based on cluster sizes
"""

    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README saved to: {readme_path}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Clustering-based deduplication for C++ benchmark designs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deduplicator_clustering.py -i data/designs.json -o results/Clustering
  python deduplicator_clustering.py -i data/designs.json -o results/Clustering --target-min 50 --target-max 100
  python deduplicator_clustering.py -i data/designs.json -o results/Clustering --compare-features
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input JSON file containing designs')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for results')
    parser.add_argument('--target-min', type=int, default=80,
                       help='Minimum number of designs to select (default: 80)')
    parser.add_argument('--target-max', type=int, default=120,
                       help='Maximum number of designs to select (default: 120)')
    parser.add_argument('--compare-features', action='store_true',
                       help='Run feature comparison experiment')
    
    return parser.parse_args()


def main():
    """主函数"""
    import time
    start_time = time.time()
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    print(f"{'='*60}")
    print("CLUSTERING-BASED DEDUPLICATION")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Target range: {args.target_min}-{args.target_max}")
    print(f"{'='*60}")
    
    # 创建选择器
    selector = AdaptiveClusteringSelector(target_range=(args.target_min, args.target_max))
    
    # 执行流程
    if not selector.load_data(args.input):
        print("Failed to load data. Exiting.")
        return
    
    # 是否运行对比实验
    if args.compare_features:
        best_mode, comparison_results = selector.run_comparison_experiment()
        print(f"\nUsing best mode: {best_mode}")
    else:
        selector.extract_features(feature_mode="combined")
        selector.perform_clustering()
    
    selected_indices = selector.select_representatives()
    
    # 保存结果
    selected_designs = [selector.designs[i] for i in selected_indices]
    
    output_file = os.path.join(args.output, 'after_cluster.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(selected_designs, f, indent=2, ensure_ascii=False)
    
    # 分析和可视化
    analysis = selector.analyze_results(selected_indices)
    analysis['input_designs'] = selector.designs  # 添加原始数据用于统计
    
    selector.visualize_results(selected_indices, args.output)
    
    # 计算执行时间
    execution_time = time.time() - start_time
    
    # 创建README
    create_readme(args, analysis, args.input, args.output, execution_time)
    
    # 输出结果
    print(f"\n{'='*60}")
    print("CLUSTERING RESULTS")
    print(f"{'='*60}")
    print(f"Input designs: {len(selector.designs)}")
    print(f"Selected designs: {analysis['selected_count']}")
    print(f"Target range: {analysis['target_range']}")
    print(f"Selection ratio: {(analysis['selected_count'] / len(selector.designs)) * 100:.2f}%")
    print(f"Optimal K: {analysis['optimal_k']}")
    print(f"Silhouette score: {analysis['silhouette_score']:.4f}")
    print(f"Cluster coverage: {analysis['coverage_ratio']:.2%}")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"\nResults saved to: {args.output}/")
    print(f"- after_cluster.json")
    print(f"- clustering_results.png") 
    print(f"- README.md")


if __name__ == "__main__":
    main()