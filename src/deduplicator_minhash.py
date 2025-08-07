import re
import json
import hashlib
import argparse
import os
from typing import List, Set, Dict, Tuple, Any, Optional
from collections import defaultdict
import numpy as np
from datasketch import MinHashLSH, MinHash

class CppCodeDeduplicator:
    def __init__(self, 
                 k_shingle: int = 5,
                 num_perm: int = 128, 
                 similarity_threshold: float = 0.85,
                 enable_hls_awareness: bool = True,
                 enable_lsh: bool = False,
                 lsh_threshold: float = 0.8,
                 verbose: bool = True):
        """
        简化的C++代码去重器
        """
        self.k_shingle = k_shingle
        self.num_perm = num_perm
        self.similarity_threshold = similarity_threshold
        self.enable_hls_awareness = enable_hls_awareness
        self.enable_lsh = enable_lsh
        self.lsh_threshold = lsh_threshold
        self.verbose = verbose
        
        # 初始化LSH（如果启用）
        if self.enable_lsh:
            self.lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
        else:
            self.lsh = None
        
        # 关键词集合
        self.cpp_keywords = {
            'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
            'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
            'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static',
            'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while',
            'class', 'namespace', 'template', 'typename', 'public', 'private', 'protected',
            'virtual', 'override', 'final', 'explicit', 'inline', 'friend', 'operator',
            'new', 'delete', 'this', 'true', 'false', 'nullptr', 'throw', 'try', 'catch'
        }
        
        self.hls_keywords = {
            'ap_int', 'ap_uint', 'ap_fixed', 'ap_ufixed', 'hls::stream', 'stream',
            'pragma', 'HLS', 'PIPELINE', 'UNROLL', 'DATAFLOW', 'INTERFACE',
            'ARRAY_PARTITION', 'ARRAY_RESHAPE', 'RESOURCE', 'LATENCY'
        }

    def deduplicate(self, designs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        主要的去重方法
        
        Returns:
            (去重后的设计列表, 统计信息)
        """
        if self.verbose:
            print(f"开始去重，原始设计数量: {len(designs)}")
            print(f"模式: {'LSH加速' if self.enable_lsh else '精确比较'}")
        
        if self.enable_lsh:
            return self._deduplicate_with_lsh(designs)
        else:
            return self._deduplicate_exact(designs)

    def _deduplicate_with_lsh(self, designs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """LSH加速去重"""
        deduplicated_designs = []
        duplicate_info = []
        processed_count = 0
        
        for i, design in enumerate(designs):
            try:
                features = self._extract_features(design)
                if not features:
                    continue
                
                minhash = self._compute_minhash(features)
                design_id = f"design_{i}"
                
                # 查找相似设计
                similar_designs = self.lsh.query(minhash)
                
                if similar_designs:
                    # 找到重复，记录信息但不添加到结果中
                    duplicate_info.append({
                        'duplicate_index': i,
                        'similar_to': similar_designs[0],
                        'estimated_similarity': 'LSH_match'
                    })
                else:
                    # 新设计，添加到LSH和结果中
                    self.lsh.insert(design_id, minhash)
                    deduplicated_designs.append(design)
                
                processed_count += 1
                
                if self.verbose and processed_count % 100 == 0:
                    print(f"已处理: {processed_count}/{len(designs)}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"处理设计 {i} 时出错: {e}")
                continue
        
        stats = {
            'original_count': len(designs),
            'deduplicated_count': len(deduplicated_designs),
            'duplicate_count': len(duplicate_info),
            'processing_mode': 'LSH',
            'duplicate_info': duplicate_info
        }
        
        return deduplicated_designs, stats

    def _deduplicate_exact(self, designs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """精确比较去重"""
        # 第一步：提取所有有效设计的特征
        design_features = []
        for i, design in enumerate(designs):
            if self.verbose and i < 5:  # 只打印前5个设计的详细信息
                print(f"\n处理设计 {i}:")
                print(f"设计键: {list(design.keys())}")
                
            features = self._extract_features(design)
            if features:
                minhash = self._compute_minhash(features)
                design_features.append((i, minhash, design))
                if self.verbose and i < 5:
                    print(f"提取到 {len(features)} 个特征")
            else:
                if self.verbose and i < 5:
                    print("未提取到特征")
        
        if self.verbose:
            print(f"有效设计数量: {len(design_features)}")
        
        # 第二步：两两比较找重复
        is_duplicate = [False] * len(design_features)
        duplicate_info = []
        
        for i in range(len(design_features)):
            if is_duplicate[i]:
                continue
                
            idx_i, minhash_i, design_i = design_features[i]
            
            for j in range(i + 1, len(design_features)):
                if is_duplicate[j]:
                    continue
                    
                idx_j, minhash_j, design_j = design_features[j]
                similarity = minhash_i.jaccard(minhash_j)
                
                if similarity >= self.similarity_threshold:
                    is_duplicate[j] = True
                    duplicate_info.append({
                        'duplicate_index': idx_j,
                        'original_index': idx_i,
                        'similarity': similarity
                    })
        
        # 第三步：收集非重复设计
        deduplicated_designs = []
        for i, (idx, minhash, design) in enumerate(design_features):
            if not is_duplicate[i]:
                deduplicated_designs.append(design)
        
        stats = {
            'original_count': len(designs),
            'valid_count': len(design_features),
            'deduplicated_count': len(deduplicated_designs),
            'duplicate_count': len(duplicate_info),
            'processing_mode': 'Exact',
            'duplicate_info': duplicate_info
        }
        
        return deduplicated_designs, stats

    def _extract_features(self, design: Dict[str, Any]) -> Set[str]:
        """从设计中提取特征"""
        source_code = design.get('source_code', [])
        if not source_code:
            if self.verbose:
                print("警告: source_code为空")
            return set()
            
        if not isinstance(source_code, list):
            if self.verbose:
                print(f"警告: source_code不是列表，类型为: {type(source_code)}")
            return set()
        
        all_tokens = []
        for i, file_content in enumerate(source_code):
            if isinstance(file_content, dict):
                content = file_content.get('file_content', '')
                if self.verbose and i == 0:  # 只打印第一个文件的信息
                    print(f"文件内容长度: {len(content)}")
            else:
                content = str(file_content)
                if self.verbose and i == 0:
                    print(f"文件内容(字符串)长度: {len(content)}")
            
            tokens = self._tokenize_cpp_code(content)
            all_tokens.extend(tokens)
        
        if self.verbose and len(all_tokens) == 0:
            print("警告: 没有提取到任何token")
        
        # 生成k-shingles
        shingles = set()
        if len(all_tokens) >= self.k_shingle:
            for i in range(len(all_tokens) - self.k_shingle + 1):
                shingle = ' '.join(all_tokens[i:i + self.k_shingle])
                shingles.add(shingle)
        else:
            if self.verbose:
                print(f"警告: token数量({len(all_tokens)})少于k-shingle大小({self.k_shingle})")
        
        return shingles

    def _tokenize_cpp_code(self, code: str) -> List[str]:
        """简化的C++代码标记化"""
        # 移除注释
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # 移除字符串字面量
        # for example: "a" -> ""
        code = re.sub(r'"[^"]*"', '""', code)
        code = re.sub(r"'[^']*'", "''", code)
        
        # 标记化
        tokens = re.findall(r'\w+|[^\w\s]', code)
        
        # 规范化
        normalized_tokens = []
        for token in tokens:
            token = token.lower().strip()
            if not token or token.isspace():
                continue
            
            if token.isdigit():
                normalized_tokens.append('<NUM>')
            elif self.enable_hls_awareness and token in self.hls_keywords:
                normalized_tokens.append(f'HLS_{token.upper()}')
            elif token in self.cpp_keywords:
                normalized_tokens.append(f'CPP_{token.upper()}')
            else:
                normalized_tokens.append(token)
        
        return normalized_tokens

    def _compute_minhash(self, shingles: Set[str]) -> MinHash:
        """计算MinHash签名"""
        minhash = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            minhash.update(shingle.encode('utf-8'))
        return minhash


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='C++ Code Deduplicator')
    parser.add_argument('-i', '--input', required=True, help='输入JSON文件')
    parser.add_argument('-o', '--output', default='./output', help='输出目录路径')
    parser.add_argument('--similarity-threshold', type=float, default=0.85, help='相似度阈值')
    parser.add_argument('--enable-lsh', action='store_true', help='启用LSH加速')
    parser.add_argument('--lsh-threshold', type=float, default=0.8, help='LSH阈值')
    parser.add_argument('--quiet', action='store_true', help='安静模式')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    with open(args.input, 'r', encoding='utf-8') as f:
        designs = json.load(f)
    
    # 记录输入参数
    input_params = {
        'input_file': args.input,
        'output_directory': output_dir,
        'similarity_threshold': args.similarity_threshold,
        'enable_lsh': args.enable_lsh,
        'lsh_threshold': args.lsh_threshold,
        'quiet_mode': args.quiet,
        'original_design_count': len(designs)
    }
    
    # 初始化去重器
    deduplicator = CppCodeDeduplicator(
        similarity_threshold=args.similarity_threshold,
        enable_lsh=args.enable_lsh,
        lsh_threshold=args.lsh_threshold,
        verbose=not args.quiet
    )
    
    # 执行去重
    deduplicated_designs, stats = deduplicator.deduplicate(designs)
    
    # 1. 保存纯粹的去重后设计列表（与输入格式一致）
    deduplicated_json_path = os.path.join(output_dir, 'after_minhash.json')
    with open(deduplicated_json_path, 'w', encoding='utf-8') as f:
        json.dump(deduplicated_designs, f, indent=2, ensure_ascii=False)
    
    # 2. 保存完整的实验结果（包含参数和统计信息）
    output_json_path = os.path.join(output_dir, 'report.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'input_parameters': input_params,
            'statistics': stats,
            'deduplicated_designs': deduplicated_designs
        }, f, indent=2, ensure_ascii=False)
    
    # 生成结果报告到 readme.md
    readme_md_path = os.path.join(output_dir, 'readme.md')
    with open(readme_md_path, 'w', encoding='utf-8') as f:
        f.write("# C++ 代码去重实验结果\n\n")
        
        f.write("## 实验参数\n")
        f.write(f"- **输入文件**: {input_params['input_file']}\n")
        f.write(f"- **输出目录**: {input_params['output_directory']}\n")
        f.write(f"- **相似度阈值**: {input_params['similarity_threshold']}\n")
        f.write(f"- **LSH加速**: {'启用' if input_params['enable_lsh'] else '禁用'}\n")
        if input_params['enable_lsh']:
            f.write(f"- **LSH阈值**: {input_params['lsh_threshold']}\n")
        f.write(f"- **处理模式**: {stats['processing_mode']}\n\n")
        
        f.write("## 去重结果\n")
        f.write(f"- **原始设计数量**: {stats['original_count']}\n")
        if 'valid_count' in stats:
            f.write(f"- **有效设计数量**: {stats['valid_count']}\n")
        f.write(f"- **去重后数量**: {stats['deduplicated_count']}\n")
        f.write(f"- **重复设计数量**: {stats['duplicate_count']}\n")
        
        dedup_rate = (1 - stats['deduplicated_count']/stats['original_count'])*100
        f.write(f"- **去重率**: {dedup_rate:.1f}%\n\n")
        
        f.write("## 重复设计详情\n")
        if stats['duplicate_count'] > 0:
            f.write("| 重复设计索引 | 原始设计索引 | 相似度 |\n")
            f.write("|-------------|-------------|--------|\n")
            for dup_info in stats['duplicate_info'][:10]:  # 只显示前10个
                if 'similarity' in dup_info:
                    f.write(f"| {dup_info['duplicate_index']} | {dup_info['original_index']} | {dup_info['similarity']:.3f} |\n")
                else:
                    f.write(f"| {dup_info['duplicate_index']} | {dup_info.get('similar_to', 'N/A')} | LSH匹配 |\n")
            
            if stats['duplicate_count'] > 10:
                f.write(f"\n*注：仅显示前10个重复设计，总共发现 {stats['duplicate_count']} 个重复设计*\n")
        else:
            f.write("未发现重复设计。\n")
        
        f.write(f"\n## 文件输出\n")
        f.write(f"- **去重后设计**: `{deduplicated_json_path}`\n")
        f.write(f"- **完整实验数据**: `{output_json_path}`\n")
        f.write(f"- **结果报告**: `{readme_md_path}`\n")
    
    # 输出结果到控制台
    print(f"\n=== 去重完成 ===")
    print(f"原始设计: {stats['original_count']}")
    print(f"去重后: {stats['deduplicated_count']}")
    print(f"重复数量: {stats['duplicate_count']}")
    print(f"去重率: {(1 - stats['deduplicated_count']/stats['original_count'])*100:.1f}%")
    print(f"输出目录: {output_dir}")
    print(f"去重后设计: {deduplicated_json_path}")
    print(f"完整实验数据: {output_json_path}")
    print(f"结果报告: {readme_md_path}")


if __name__ == "__main__":
    main() 