# **战略性选择多样化C++基准：一种多阶段结构与语义去重方法**

## **1\. 引言：多样化C++基准的必要性**

本报告旨在解决用户面临的核心挑战：从900个C++文件中去重并挑选出100个具有代表性和独特性的文件作为基准。每个文件都应实现不同功能，且结构不应过于重复，例如，如果存在多个功能相似但仅有细微结构差异（如都是两重循环且计算路径相似）的文件，则只应保留一个。由于数据集规模庞大，无法进行逐一手动挑选，因此需要提出一种自动化且可扩展的策略。  
传统的去重方法，例如用户提及的基于MinHash和Jaccard相似度的文本去重方法，在处理大规模文本集合中的近乎重复文档时表现出高效性 1。然而，将这些方法直接应用于C++源代码以实现结构多样性存在固有限制。源代码中的表层修改，例如空格、注释的增删、变量名的重命名或独立语句的重新排序，都可能显著改变词法相似度分数，但程序的底层逻辑或结构却保持不变 11。用户对“结构不应过于重复”的明确要求，以及对MinHash之外“更好方法”的探寻，都强调了纯词法方法在C++代码去重方面存在的不足。MinHash在设计上侧重于“shingles”（词元或字符）的集合，本身无法捕捉代码的语义含义或控制流 2。  
用户希望选取的基准文件既要“比较典型”，又要包含“比较特别的”文件，这在基准选择中构成了一个精妙的挑战。“典型”通常意味着选择数据集中最常见或最核心的模式，这可以通过选择聚类中心来实现。然而，“特别”则暗示着需要包含独特、异常或不常见但功能和结构上具有显著差异的文件。如果选择策略仅仅关注“典型性”（例如，只选择聚类中心），则可能会无意中排除那些远离任何聚类中心但代表重要且独特功能或结构的“特别”案例。因此，一个成功的选择策略不能仅仅依赖于识别最常见的模式。它必须整合机制，确保最终的100个基准文件能够充分覆盖多样性的全谱，包括普遍存在的和稀有但重要的结构或功能特征。这表明需要采用多样性最大化技术（例如最远点采样），这些技术明确选择在特征空间中相互距离最大的点，并可能与聚类方法结合使用。  
为了有效应对C++代码中结构和语义去重以及多样性最大化的复杂性，本报告提出了一种多阶段策略。该方法将首先利用高效的词法去重技术快速过滤掉明显的文本克隆。随后，它将采用先进的结构和语义分析技术，将剩余代码转换为丰富的数值表示。这些表示将与复杂的聚类和多样性最大化算法结合使用，以选择最终的100个基准文件，确保它们在各种结构和功能模式上真正具有多样性和代表性。

## **2\. 代码相似性分析基础**

本节建立了比较源代码的理论基础，从表层文本比较逐步深入到更深层的结构和语义分析。

### **2.1. 词法相似性：MinHash与Jaccard指数**

Jaccard相似度系数，正式定义为 J(A,B)=∣A∪B∣∣A∩B∣​，量化了两个集合A和B之间的重叠程度。其值范围从0（无共同元素）到1（集合完全相同）2。MinHash是一种强大的概率技术，旨在高效估计Jaccard相似度，尤其适用于非常大的集合，而无需显式计算它们的交集或并集 1。它通过为每个集合生成一个紧凑的“签名”来实现这一点，该签名是根据集合元素进行多次随机排列后得到的最小哈希值的集合 1。两个集合的MinHash值相同的概率近似等于它们的Jaccard相似度 4。这种估计的准确性随着排列数（k）的增加而提高，其误差界为 O(1/k​) 1。  
为了将MinHash应用于源代码，代码必须首先转换为集合表示。这通常通过“分片”（shingling）完成，即将文档分解为k个“词元”或“字符”的重叠序列（k-gram）2。这些k-gram（shingles）随后被哈希为整数ID，文档则表示为这些唯一哈希值的集合 2。  
对于C++代码，分片策略可以包括：

* **字符k-gram：** 尽管简单，但这种方法对细微的文本变动（如空格、注释或轻微的格式更改）高度敏感。  
* **单词/词元k-gram：** 这是一种更稳健的代码处理方法，因为它考虑的是有意义的词法单元（例如，关键字、标识符、运算符、字面量）序列，而不是原始字符 2。这需要一个C++词法分析器首先将源代码转换为词元流。  
* **规范化词元k-gram：** 为了进一步增强对保留代码逻辑的表层差异（例如，变量重命名、重新格式化）的鲁棒性，词元可以被规范化。这涉及将特定标识符（例如，myVar、temp）替换为通用占位符（例如，IDENTIFIER），并将字面量（例如，123、"hello"）替换为LITERAL 12。这个过程超越了纯粹的词法匹配，即使在基于词元框架内，也趋向于更抽象的结构比较。

尽管MinHash对大型数据集高效，但其在原始或规范化词元上操作时，本质上将代码视为“shingles的集合” 6。它无法固有地捕捉定义程序逻辑的*结构*层次、*控制流*或*数据依赖* 2。例如，两个C++文件实现相同的底层算法，但使用不同的变量名或重新排序的独立语句，其词法相似度可能较低，即使它们的结构和语义相似度很高。研究也明确承认了这一局限性，指出MinHash本身无法提取语义含义 2。

### **2.2. 结构相似性：抽象语法树（ASTs）**

抽象语法树（AST）是一种树状数据结构，用于表示源代码的抽象语法结构 11。与具体的解析树不同，ASTs抽象掉了非必要的细节，如标点符号（例如，大括号、分号、括号），并通过其层次结构隐式表示分组 20。AST中的每个节点都表示源代码中的一个构造（例如，函数声明、循环语句、表达式）20。ASTs在编译器中作为关键的中间表示，是许多代码分析任务的基础 20。  
通过关注编程构造的层次结构，ASTs固有地忽略了空格、注释甚至标识符名称的变化（如果在比较过程中标识符被规范化或忽略）11。这一特性使得ASTs在检测“Type-II克隆”（除了标识符重命名外完全相同的代码片段）和许多“Type-III克隆”（具有轻微语法变体，如添加/删除语句或重新排序独立语句的代码片段）方面非常有效 11。即使存在这些表层变化，树结构也大体保持相似。  
ASTs的相似性度量方法包括：

* **树编辑距离：** 该指标量化了将一个AST转换为另一个AST所需的最小编辑操作（节点插入、删除或替换）数量 25。编辑距离越小，结构相似度越高。这种方法能有效捕捉复杂的代码结构 25。  
* **AST节点类型频率向量：** 每个AST可以表示为一个数值向量，其中每个维度对应一种特定的AST节点类型（例如，FunctionDecl（函数声明）、ForStmt（for循环）、IfStmt（if语句）、BinaryOperator（二元操作）、CallExpr（函数调用）、DeclRefExpr（声明引用））13。每个维度的值可以是该节点类型的频率或其存在与否的二元指示。然后，可以通过对这些频率向量应用标准向量距离度量（例如，余弦相似度、欧几里得距离）来计算两个AST之间的相似度 13。这种方法提供了代码的“语法指纹”。例如，一个ForStmt和BinaryOperator节点频率较高的文件可能表示一个计算密集型、循环驱动的函数。  
* **哈希AST子树：** 类似于文本的MinHash，AST的子树可以被哈希以识别共同的结构模式 11。可以采用技术使这些哈希函数对微小变体具有鲁棒性，例如忽略标识符名称（树中的叶子节点）或处理可交换运算符（例如，+、\*）11。这允许将结构相似的子树分组到相同的哈希桶中，以便进行高效比较。  
* **AST上的图核：** 更先进的技术将AST视为通用图，并应用图核方法（例如，Weisfeiler-Lehman核）来计算结构相似度 30。这些方法可以捕捉树结构中复杂的关联和模式。

### **2.3. 控制流相似性：控制流图（CFGs）**

控制流图（CFG）是一种有向图，直观地表示程序或函数中所有可能的执行路径 21。CFG中的节点代表“基本块”（单入口、单出口的指令序列），有向边代表这些块之间可能的控制转移 33。CFGs是静态分析、编译器优化和程序理解的关键工具 33。  
CFGs提供了程序指令如何执行的清晰“地图”，说明了基于条件语句、循环和函数调用的不同路径 33。不同类型的循环（例如，for、while、do-while、集合控制循环）和条件构造（例如，if-else、switch）会转换为CFG中独特且可识别的结构模式 35。这使得CFGs在根据代码的控制逻辑识别和区分代码方面特别有价值，而简单的词法或基本AST节点计数可能会忽略这一点。  
CFG的相似性度量方法包括：

* **图同构/子图匹配：** 直接比较CFGs以检测结构等价性或识别共同子图可用于克隆检测 34。这种方法虽然精确，但对于大型图来说计算量可能很大。  
* **CFG特征向量：** 可以从CFG中提取数值特征以形成用于相似性比较的向量。常见指标包括：  
  * **圈复杂度：** 一种广泛使用的指标，量化程序源代码中线性独立路径的数量，直接从其CFG计算得出 36。它提供了控制流复杂性的定量评估。例如，每个if、for、case、&&或||语句通常会使复杂度分数增加1 36。值越高表示分支和循环越多，表明控制结构越复杂。  
  * **CFG节点/边/路径计数：** 反映控制流图大小和连接性的基本计数 46。  
  * **循环和条件分支计数：** 明确计数各种循环构造（for、while、do-while）和条件语句（if-else、switch）的出现次数 35。这些是代码控制流模式的直接指标。  
  * **基本块深度/广度：** 源自CFG拓扑的指标，例如平均和最大基本块深度和广度，可以描述控制流的“形状”和嵌套 46。  
* **CFG路径特征：** 分析CFG中的特定执行路径可以捕捉更细微的行为模式 46。这涉及识别基本块的共同序列或与循环迭代和条件分支相关的特定模式 35。  
* **CFG上的图核：** 类似于ASTs，图核方法可以应用于CFGs以计算结构相似性，捕捉控制流节点之间复杂的关联 30。

CFGs擅长可视化程序流、识别不可达代码以及辅助代码优化和测试 33。它们通过表示所有可能的执行路径来有效编码程序行为 34。然而，CFGs对于非常大的程序可能变得极其复杂，可能无法捕捉不可预测的运行时行为，并且主要关注控制流，对数据操作的洞察有限 33。

### **2.4. 语义相似性：程序依赖图（PDGs）**

程序依赖图（PDG）是一种复杂的图形表示，用于捕获过程中源代码的数据依赖和控制依赖 21。在PDG中，语句表示为顶点，它们之间的依赖关系表示为边 47。直观地，PDGs编码了基本的程序逻辑，并反映了开发人员编写代码时的底层思维过程 47。  
PDGs对各种代码转换和抄袭伪装具有高度鲁棒性，这些转换和伪装会使词法甚至纯语法（基于AST）方法失效 24。这些转换包括格式更改（例如，空格、注释）、标识符重命名、独立语句的重新排序、控制结构替换（例如，for循环替换为等效的while循环）以及插入无关代码 47。一个关键优势是，只要程序的正确性和底层功能逻辑得以保留，PDG结构就基本保持不变 47。这使得PDGs在检测“Type-IV克隆”方面异常有效，即功能或语义相似但通过完全不同语法变体实现的代码片段 12。  
PDG的相似性度量方法包括：

* **子图同构：** 使用PDGs进行抄袭检测的常见方法是寻找原始程序PDG与可疑程序PDG之间的子图同构 42。通常采用宽松的同构形式（例如，γ-同构）来解释轻微的、非抄袭的差异 47。虽然子图同构通常是一个NP完全问题，但在该特定应用领域中通常是可处理的，因为单个过程的PDG通常不会任意大，并且其特定的图属性（例如，顶点类型的多样性）可以使基于回溯的同构算法高效 47。  
* **PDG特征向量：** 可以从PDG中提取数值特征以形成用于相似性比较的向量。这些特征可以包括数据和控制依赖边的数量和类型，以及图密度 42。这些特征提供了代码逻辑结构的定量摘要。  
* **PDG上的图核：** 图核方法，例如Weisfeiler-Lehman（WL）核，可以应用于PDGs以计算近似图相似性 30。这些方法在准确性和计算成本之间取得了平衡，通过捕获各种类型的图信息（例如，有向边、标签）同时保留核函数计算成本低的优势，使其适用于大型数据集 32。

PDG方法在对抗复杂的代码转换方面非常有效，并且可以检测非连续克隆，提供比单独的AST或CFG更深层的语义理解 32。

### **2.5. 统一表示：代码属性图（CPGs）**

代码属性图（CPG）代表了代码表示的突破性进展，通过将抽象语法树（ASTs）、控制流图（CFGs）和程序依赖图（PDGs）的信息集成到单个互联的图数据库中，提供了源代码的整体统一视图 49。这种多层方法在单个、全面的数据结构中捕获了语法结构、控制流以及数据/控制依赖关系 21。  
CPGs作为一种语言无关的中间表示，专门为高级代码分析任务而设计，包括漏洞发现、安全研究和静态程序分析 50。通过整合所有三种基本图类型，CPGs可以有效建模各种复杂的代码模式和漏洞，这些模式和漏洞在单独分析单个表示时可能会被遗漏 49。生成CPGs的工具（例如Joern）的一个显著优势是其“模糊解析”能力，这使得它们即使在代码不完整、缺少依赖项或无法完美编译的情况下也能导入和分析代码 50。这种鲁棒性对于分析真实世界中可能混乱的代码库至关重要。CPGs能够有效地跟踪应用程序中的数据流和控制流，有助于识别更简单的表示无法捕捉的问题语义模式 52。  
研究表明，代码相似性分析存在一个清晰的演进过程，从表层的词法（基于原始文本/词元的MinHash）到越来越稳健和语义丰富的表示（基于ASTs和CFGs的结构，以及基于PDGs的语义）。每种表示级别都提供了独特的视角，并对不同类型的代码转换具有不同的鲁棒性。MinHash高效但浅层，无法捕捉结构或语义含义。ASTs抽象了语法结构，忽略了格式和变量名。CFGs建模了执行路径，区分了控制流。PDGs捕捉了数据和控制依赖，使其对保留语义的更改（Type-IV克隆）具有高度弹性。CPGs将所有这些统一起来，提供了最全面的视图。用户对“结构多样性”以及避免“过多重复”（例如，相似的循环模式）的明确要求，直接表明纯词法方法不足，更深层的结构和语义分析至关重要。因此，一个真正稳健和有效的C++基准选择解决方案必须超越简单的词法相似性。它需要一种多层特征提取方法，整合来自ASTs、CFGs以及理想情况下PDGs（或最有效地，整合所有三者的CPGs）的洞察。这种混合方法确保所捕获的“多样性”在代码逻辑和结构方面具有实际意义。  
代码表示越详细、语义越丰富（例如，完整的ASTs，PDGs上的直接图同构），其在检测细微相似性和功能等价性方面的准确性就越高。然而，这种增加的粒度通常伴随着显著的计算成本。例如，直接树编辑距离可能计算成本很高（例如，N个AST节点为O(N3)）11，PDGs上的子图同构通常是NP完全问题 32。对于900个C++文件的数据集，使用这种计算密集型方法进行两两比较（这将需要O(N2)次比较，每次都可能很复杂）将是计算上不可行的，并且资源密集。为了在900个文件上实现可扩展性，同时保留深层的结构和语义理解，该策略必须平衡所需的细节级别与计算可行性。这强烈表明，不应直接进行图或树比较，而应从这些丰富的表示中提取*数值特征向量*（例如，AST节点类型频率、CFG结构指标、PDG依赖计数），然后对这些向量应用高效的聚类和采样算法。代码嵌入技术，即将复杂的代码结构转换为固定长度的数值向量，正是为了解决这种权衡而设计的，它能够实现高效的下游机器学习任务，如聚类和相似性搜索 21。  
**表2.1：代码相似性分析的代码表示方法比较**

| 表示类型 | 示例 | 捕获的关键信息 | 对转换的鲁棒性 | 主要用例 | 计算复杂度（一般） |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 词法 | 原始文本，词元 | 字符/单词序列 | 低（对格式、变量名敏感） | 文本去重、抄袭检测（表层） | 低 |
| 结构 | 抽象语法树（AST） | 语法层次结构 | 中（忽略格式、变量名，对语句重排敏感） | 代码理解、重构、克隆检测（Type-II, Type-III） | 中 |
| 控制流 | 控制流图（CFG） | 执行路径、控制逻辑 | 中（区分循环/条件结构，对数据流有限） | 编译器优化、漏洞分析、路径测试 | 中 |
| 语义 | 程序依赖图（PDG） | 数据/控制依赖 | 高（对语义保留的修改鲁棒） | 抄袭检测（Type-IV）、程序理解、漏洞分析 | 高 |
| 统一 | 代码属性图（CPG） | 语法、控制流、数据/控制依赖的整体视图 | 极高（模糊解析，对多种转换鲁棒） | 漏洞发现、安全研究、高级程序分析 | 高 |

## **3\. C++代码的特征工程**

本节详细介绍了将C++源代码转换为可量化的数值特征的过程，这些特征适用于相似性比较和聚类。这些特征的质量对于准确捕捉“结构多样性”至关重要。

### **3.1. 从词法表示中提取：MinHash签名**

MinHash签名可以作为文档（包括源代码）的初始紧凑特征向量 3。每个签名是一个固定长度的哈希值向量 9。num\_permutations（或numPerm）参数决定了此签名的长度，并显著影响Jaccard相似度估计的准确性，其误差界为O(1/k​)，其中k是使用的哈希函数或排列的数量 1。虽然更高的num\_permutations会带来更准确的近似，但也会增加计算成本 1。在实际应用中，num\_permutations的常见值范围为128到240 17。  
对于C++代码，基于词元的分片通常优于字符k-gram，因为它能捕捉更有意义的结构片段 2。此过程包括：

1. **词元化：** 将C++源代码分解为其基本的词法单元（例如，if、for、class等关键字；myVariable、calculateSum等标识符；+、==等运算符；10、"hello"等字面量）。  
2. **规范化（可选但推荐）：** 为了增强对表层差异的鲁棒性，词元可以被规范化。例如，所有用户定义的标识符都可以替换为通用占位符（例如，IDENTIFIER），所有数值或字符串字面量替换为LITERAL。这种规范化过程有助于确保MinHash更关注词元*类型*的序列，而不是其具体值，从而降低对变量重命名或常量更改等琐碎变化的敏感性 12。  
3. **分片形成：** 从规范化的词元流中生成重叠的k个规范化词元序列（例如，k=3到k=5个词元）2。MinHash实现中的window\_size参数通常对应于此k值 59。

### **3.2. 从结构/语义表示中提取（ASTs、CFGs、PDGs、CPGs）**

**结构特征的提取：** 这些特征捕捉代码的内在组织和逻辑，超越了简单的文本内容。

* **AST节点类型频率：** 将每个C++文件解析为其抽象语法树（AST），然后统计不同AST节点类型的出现次数。C++ AST节点类型的示例包括FunctionDecl（函数声明）、ForStmt（for循环）、IfStmt（if语句）、BinaryOperator（二元操作）、CallExpr（函数调用）和DeclRefExpr（声明引用）13。这会创建一个向量，其中每个维度代表一个特定的节点类型，其值是该节点在AST中的频率。此向量可作为代码的“语法指纹”。例如，一个ForStmt和BinaryOperator节点频率较高的文件可能表示一个计算密集型、循环驱动的函数。  
* **CFG结构模式：** 从控制流图（CFG）中提取定量属性，以表征代码的控制逻辑：  
  * **圈复杂度：** 直接衡量程序源代码中线性独立路径数量的指标，从其CFG计算得出 36。它提供了控制流复杂性的定量评估。例如，每个if、for、case、&&或||语句通常会使复杂度分数增加1 36。值越高表示分支和循环越多，表明控制结构越复杂。  
  * **CFG元素计数：** CFG节点（基本块）和边的总数等简单计数 46。  
  * **循环和条件分支计数：** 明确计数各种循环构造（for、while、do-while）和条件语句（if-else、switch）的出现次数 35。这些是代码控制流模式的直接指标。  
  * **基本块深度/广度：** 源自CFG拓扑的指标，例如平均和最大基本块深度和广度，可以描述控制流的“形状”和嵌套 46。  
* **PDG依赖计数：** 量化程序依赖图（PDG）中存在的数据和控制依赖的数量和类型 42。这可以包括特定依赖类型（例如，写后读数据依赖、源自条件语句的控制依赖）的计数。这些特征捕捉了语句之间复杂的逻辑关系。  
* **基于度量的特征：** 除了特定于图的度量之外，通用代码度量也很有价值：  
  * **代码行数（LOC）：** 虽然是一个简单的度量，但与圈复杂度结合使用时，LOC可以提供更全面的潜在复杂性和错误倾向图景 43。  
  * **程序元素计数：** 函数、类、方法和全局变量的数量。  
  * **Halstead度量：** 根据唯一运算符和操作数的数量及其总出现次数得出的度量，提供有关程序大小、难度和工作量的洞察 45。

**代码嵌入技术：** 对于更复杂的语义理解，尤其是在处理代码属性图（CPGs）时，嵌入技术至关重要。它们将复杂的代码结构转换为固定长度的数值向量（嵌入），适用于机器学习算法 21。

* **图嵌入：** DeepWalk、Walklets或LINE等技术可以应用于CPGs（或单独的ASTs、CFGs或PDGs，将其视为图）以学习节点或整个图的低维向量表示 30。这些嵌入旨在捕获图的结构和关系属性，从而在向量空间中实现高效的相似性比较。例如，Joern可以输出CPGs，节点信息（代码元素类型、语句）可以使用独热编码（one-hot encoding）进行类型编码，并使用Word2Vec对语句进行编码，然后将它们连接起来形成一个全面的节点特征向量 49。  
* **基于神经网络的嵌入：** 使用图神经网络（GNNs）如图卷积网络（GCNs）或其他深度学习模型，可以处理复杂的代码表示（ASTs、CFGs、PDGs或CPGs）以生成捕捉语法和语义属性的嵌入 21。这些模型在学习复杂的代码依赖和关系方面特别有效，为相似性分析提供了稳健的向量表示 21。

**降维（可选但推荐）：** 如果提取的特征向量组合导致维度非常高，建议应用降维技术，例如主成分分析（PCA）或均匀流形近似与投影（UMAP）62。此过程将数据转换为较低维子空间，同时保留尽可能多的相关信息，这可以显著提高后续聚类算法的效率和可解释性。  
提取丰富而全面的C++代码特征集（包括AST节点类型频率、CFG结构模式、PDG依赖计数和各种代码度量）将不可避免地导致每个C++文件的高维特征向量。虽然更多的特征可以捕捉代码结构和逻辑的更细微之处，但高维性可能会引入“维度灾难”。在高维空间中，数据点往往显得等距，使得传统距离度量失去意义，并导致聚类算法性能不佳 28。这种现象可能会掩盖真实的相似性和差异，从而阻碍识别“结构多样化”文件的目标。因此，有效的特征工程不仅在于*提取什么*特征，还在于*如何管理*其维度。降维技术（例如，PCA、UMAP）不仅仅是优化，它们可能是确保提取的特征向量对于聚类和相似性分析有意义且有效的关键步骤。这一步将原始的高维特征空间转换为更紧凑和更具区分性的表示，从而实现更好的聚类和更准确的多样性评估。  
用户对“典型和特殊”C++文件以及“结构不应过于重复”（例如，如果都是两重循环，只选择一个）的期望，暗示着更深层次的需求：捕捉*算法*或*功能等价性*。虽然PDGs对许多保留语义的语法更改（Type-IV克隆）具有高度鲁棒性 24，但它们主要捕捉数据和控制依赖。这意味着两段代码即使实现*完全相同的功能*，但采用完全不同的算法或逻辑结构，在基于ASTs、CFGs或PDGs的分析中仍可能显示出结构上的显著差异。例如，迭代阶乘函数与递归阶乘函数将具有非常不同的CFG和PDG，即使它们的函数输出相同。在不执行代码或不应用形式化验证方法的情况下检测真正的“功能等价性”是一个极其具有挑战性的问题，通常超出静态分析技术的范围。虽然CPGs及其派生嵌入代表了捕捉深层结构和语义相似性的最先进静态分析方法，但管理对“功能等价性”的期望至关重要。所提出的策略将有效地识别并分组那些在内部结构和数据/控制流方面*看起来*和*行为*相似的文件，这是通过静态分析可实现的功能相似性的最接近代理。本报告应明确区分这一点，强调所选方法擅长识别在内部结构和数据/控制流方面*看起来*和*行为*相似的代码，而不是保证相同输入下的相同输出（这需要动态分析或形式化方法）。  
**表3.1：C++代码相似性推荐特征类型**

| 特征类别 | 具体特征示例 | 对多样性的贡献 | 提取工具/方法 |
| :---- | :---- | :---- | :---- |
| 词法 | MinHash签名 | 捕捉文本重叠，快速识别近乎重复 | 自定义词元化器/分片器，datasketch库 3 |
| 结构（AST） | AST节点类型频率 | 反映语法结构，区分不同语法模式 | Clang LibTooling 22, cppast 68 |
| 控制流（CFG） | 圈复杂度，循环/条件计数，CFG节点/边/路径数，基本块深度/广度 | 量化控制流复杂性，识别控制逻辑模式 | Clang LibTooling 71, LLVM Passes 73, Joern 50 |
| 数据流（PDG） | PDG依赖计数（数据/控制） | 识别数据/控制逻辑关系，对语义保留的修改鲁棒 | Joern 50, LLVM Passes 73 |
| 统一（CPG） | 图嵌入（节点/边特征向量） | 提供全面的语义/结构表示，捕捉复杂依赖 | Joern 49, GNNs (GCNs, GGNN) 21 |
| 通用度量 | 代码行数（LOC），Halstead度量 | 代码大小和复杂度的通用指示 | CppDepend 45, 其他静态分析工具 44 |

## **4\. 提出的多阶段基准选择策略**

本节概述了一个全面的多阶段方法，旨在高效地对900个C++文件进行去重，并选择100个结构和语义多样化的基准，以满足用户对典型性和独特性的具体要求。

### **4.1. 阶段1：快速初步去重（词法）**

此阶段旨在根据词法内容快速过滤掉近乎精确的重复文件和高度相似的文件，从而减小数据集规模，以便进行计算密集度更高的后续分析。

* **方法：**  
  * **预处理和词元化：** 对于每个C++文件，第一步是将其源代码词元化。这涉及将代码解析为一系列有意义的词法单元（例如，关键字、标识符、运算符、字面量）。为了增强对表层文本差异的鲁棒性，强烈建议对这些词元进行规范化。这可以包括：  
    * 将所有用户定义的标识符（变量名、函数名）转换为通用占位符（例如，IDENTIFIER）。  
    * 将所有数值和字符串字面量替换为通用占位符（例如，LITERAL）。  
    * 删除注释和过多的空格。  
    * 这种规范化过程有助于确保MinHash更关注词元*类型*的序列，而不是其具体值，从而降低对变量重命名或常量更改等琐碎变化的敏感性 12。  
  * **分片生成：** 从规范化的词元流中生成k-shingles（k-gram）。对于代码，k值在3到5个词元之间通常是有效的 2。这些shingles构成了MinHash操作的集合。  
  * **MinHash签名计算：** 计算每个文件shingles集合的MinHash签名。num\_permutations（或numPerm）参数决定了签名向量的长度，并显著影响Jaccard相似度估计的准确性，在平衡准确性和计算成本之间进行选择。128或240个排列数是常见且能提供良好平衡的值 17。  
  * **局部敏感哈希（LSH）实现高效比较：** 为了避免900个文件的O(N2)两两比较，采用LSH。LSH将相似的MinHash签名分组到“带”（bands）和“桶”（buckets）中，显著减少了需要进行完整相似度计算的候选对数量 8。这使得亚线性时间相似度估计成为可能 4。  
  * **去重阈值：** 应用高Jaccard相似度阈值（例如，0.8到0.9，如用户查询和研究材料中建议 17）。如果一个文件的MinHash签名与任何先前文件的签名（在同一LSH桶内）之间的Jaccard相似度超过此阈值，则该文件被视为重复并被过滤掉。此过程通常迭代文件，保留重复簇中遇到的第一个实例 59。  
* **目的：** 此阶段作为高效的粗粒度过滤器。它快速删除明显的文本克隆和近重复项，显著减小需要进行计算密集度更高的结构和语义分析（阶段2）的数据集规模。

### **4.2. 阶段2：结构/语义去重与多样性最大化**

此阶段针对经过词法去重后的文件，进行更深层次的分析，以捕捉其固有的结构和语义属性。

#### **4.2.1. 高级代码表示与特征提取**

* **方法：**  
  * **代码属性图（CPG）提取：** 利用工具（例如Joern 49）将每个C++文件转换为其代码属性图（CPG）。CPG通过整合AST、CFG和PDG的信息，提供代码的统一、多层表示 21。Joern的“模糊解析”能力使其即使在代码不完整或无法编译的情况下也能进行分析，这对于处理真实世界代码库至关重要 50。  
  * **特征向量生成：** 从CPG中提取丰富的数值特征向量，以全面表征每个C++文件。这些特征可以包括：  
    * **AST节点类型频率：** 统计CPG中AST节点的类型及其频率。例如，FunctionDecl、ForStmt、IfStmt、BinaryOperator、CallExpr等节点的出现次数 13。这些频率形成一个向量，反映代码的语法结构。  
    * **CFG结构模式：** 提取CFG的定量属性，如圈复杂度 36、CFG节点和边的数量 46、循环和条件分支的明确计数 35。这些特征捕获代码的控制流逻辑。  
    * **PDG依赖计数：** 量化PDG中数据和控制依赖的数量和类型 42。这些特征反映了语句之间的逻辑关系。  
    * **代码嵌入：** 利用图神经网络（GNNs）或Word2Vec等技术，将CPG的节点信息（代码元素类型、语句内容）转换为固定长度的数值嵌入 21。这些嵌入旨在捕捉代码的深层语义和结构属性，从而在向量空间中进行高效的相似性比较。  
    * **其他通用度量：** 纳入额外的代码度量，如代码行数（LOC）和Halstead度量 43。  
  * **降维（如果需要）：** 如果生成的特征向量维度过高，可以应用主成分分析（PCA）或均匀流形近似与投影（UMAP）等技术进行降维 62。这有助于缓解“维度灾难”问题，提高聚类算法的效率和结果的清晰度。在高维空间中，数据点之间的距离差异往往变得不明显，这会削弱距离度量的有效性，并可能导致聚类算法性能下降 28。降维通过将数据投影到更低维、更具区分性的空间，确保了后续分析能够更准确地反映代码的真实结构和语义差异。

#### **4.2.2. 聚类以分组相似代码**

* **方法：**  
  * **选择距离度量：** 基于特征向量的性质，选择合适的距离度量。  
    * **欧几里得距离：** 适用于特征具有相似尺度且聚类形状偏球形的情况 28。  
    * **余弦相似度（或余弦距离）：** 更适合于高维稀疏数据，或当关注向量方向而非幅度时，例如在文本或代码嵌入中 27。它衡量两个向量之间的角度，角度越小表示相似度越高。  
  * **选择聚类算法：**  
    * **K-Means聚类：** 一种流行且高效的算法，用于将数据点分组到预定义数量（k）的簇中，以最小化簇内方差 27。K-Means++初始化策略有助于选择更好的初始聚类中心，从而提高收敛速度和聚类结果 81。K-Means通过迭代分配点到最近的质心并重新计算质心来工作 80。  
    * **K-Medoids聚类：** 类似于K-Means，但使用实际数据点作为簇的“中心”（medoids），而不是计算平均值。K-Medoids对异常值不敏感，并且可以处理任意距离函数，使其在某些情况下优于K-Means 82。  
    * **DBSCAN：** 一种基于密度的聚类算法，能够发现任意形状的簇，并有效处理噪声（异常值）65。它不需要预先指定簇的数量。  
  * **确定最佳簇数（k）：** 对于K-Means或K-Medoids，需要确定最佳的簇数量。这可以通过肘部法则、轮廓系数或领域知识来指导。由于用户希望最终选择100个基准，因此初始聚类数量可以设置为一个大于100的值，以便在后续多样性最大化步骤中进行精细选择。  
  * **聚类执行：** 将提取的特征向量输入所选聚类算法，将相似的C++文件分组到不同的簇中。每个簇代表一类结构或语义相似的代码。

#### **4.2.3. 多样性最大化以选择基准**

* **方法：**  
  * **从每个簇中选择代表：** 从每个聚类中，选择一个或多个代表文件。这可以通过选择：  
    * **簇的中心（Centroid/Medoid）:** 选择最接近簇中心的文件，代表该簇的“典型”模式 27。  
    * **Farthest Point Sampling (FPS)：** 为了确保最终选出的100个基准文件具有最大的多样性，可以应用FPS 82。FPS从数据集中选择一个初始点，然后迭代选择距离已选点集合最远的点，直到达到所需的数量（100个文件）83。这种方法能够有效捕捉数据集的“特殊”或“边缘”情况，确保选出的基准文件在特征空间中尽可能分散，从而最大化整体多样性。  
  * **结合聚类和FPS：** 一种有效的策略是首先通过聚类将文件分组，然后从每个簇中选择一个或几个代表（例如，簇中心），再将这些代表与通过FPS选出的“最特别”的文件结合起来。或者，可以直接在所有去重后的文件上运行FPS，以确保选出的100个文件在结构和语义上尽可能不同。  
  * **迭代细化：** 如果第一次选择的100个文件未能满足多样性或代表性要求，可以调整聚类参数（如k值、距离度量）或FPS的起始点/迭代次数，并重复选择过程。

#### **4.2.4. 人工审查与精炼（最终100个基准）**

* **重要性：** 尽管自动化方法能够高效处理大规模数据，但C++代码的复杂性和用户对“典型”与“特殊”的微妙定义，使得人工审查成为最终基准选择不可或缺的一步。自动化算法可能无法完全捕捉人类对代码功能、可读性或特定应用场景中“典型性”的理解。  
* **审查过程：** 对通过自动化方法选出的100个候选文件进行人工审查。此阶段可用于：  
  * **验证代表性：** 确认选定的文件确实代表了其所属簇的结构和语义特征。  
  * **识别遗漏的“特殊”案例：** 检查是否存在未被自动化方法充分捕捉的独特或边缘案例，并根据需要手动添加。  
  * **排除误报：** 移除那些虽然在特征空间中看似独特，但实际功能或结构上并不适合作为基准的文件。  
  * **确保质量：** 检查代码的可读性、注释质量和实际功能，以确保基准文件的整体质量。  
* **反馈循环：** 人工审查的结果可以作为反馈，用于调整阶段1和阶段2的参数（例如，MinHash阈值、聚类参数、特征权重），从而在未来的迭代中改进自动化选择的质量。

## **5\. 结论与建议**

本报告提出了一种从900个C++文件中去重并选择100个多样化基准的多阶段策略。该策略超越了传统的词法相似性分析，通过深度特征工程和先进的聚类与多样性最大化算法，旨在捕捉C++代码的结构和语义多样性。  
首先，通过基于MinHash和LSH的词法去重，可以高效地过滤掉文本上高度相似的重复文件，大幅缩小数据集规模，为后续更深层次的分析奠定基础。这一步骤的效率对于处理大规模代码库至关重要。  
其次，通过从代码属性图（CPG）中提取丰富的结构和语义特征，包括AST节点类型频率、CFG结构模式、PDG依赖计数以及代码嵌入，可以为每个C++文件构建全面的数值表示。这些特征能够有效地区分不同代码的功能逻辑和内部结构，从而实现比纯词法方法更深入的相似性评估。降维技术的使用有助于管理高维特征空间带来的挑战，确保度量和聚类结果的有效性。  
最后，结合聚类和最远点采样（FPS）的策略，能够系统地从去重后的数据集中选择出既“典型”又“特殊”的100个基准文件。聚类有助于识别代码的常见模式并选择代表，而FPS则确保了选定基准文件在特征空间中的最大分散性，从而覆盖更广泛的代码多样性。尽管自动化方法提供了高效性，但最终的人工审查环节是不可或缺的，它确保了最终基准集在功能、可读性和实际应用场景中的高质量和代表性。  
**建议：**

1. **分步实施与迭代优化：** 建议分阶段实施本策略。首先完成词法去重，然后逐步引入结构和语义特征提取。在每个阶段，对结果进行评估，并根据需要调整参数，例如MinHash阈值、特征权重、聚类算法选择和簇数量。  
2. **利用现有工具链：** 对于C++代码分析，可以利用成熟的开源工具，例如Clang LibTooling或Joern，来生成AST、CFG和PDG，并从中提取特征。这些工具为构建强大的代码分析管道提供了基础。  
3. **明确“典型”和“特殊”的定义：** 在人工审查阶段，建议团队内部对“典型”和“特殊”的C++代码功能和结构达成共识，这将有助于指导最终的基准选择，并为未来的自动化优化提供明确的指导。  
4. **持续评估与维护：** 一旦选定基准集，应定期对其进行评估和维护。随着新代码的开发或现有代码的演变，基准集的代表性可能会发生变化，可能需要重新运行部分或全部选择流程以保持其有效性。  
5. **认识到功能等价性的局限性：** 重要的是要认识到，静态分析方法在检测纯粹的“功能等价性”（即不同算法实现相同输入-输出行为）方面存在固有限制。本报告提出的方法主要侧重于捕捉代码的结构和依赖逻辑的相似性，这通常是功能相似性的强有力代理。对于需要严格功能等价性验证的场景，可能需要结合动态分析或形式化验证方法。

#### **Works cited**

1. MinHash \- Wikipedia, accessed May 29, 2025, [https://en.wikipedia.org/wiki/MinHash](https://en.wikipedia.org/wiki/MinHash)  
2. MinHash Tutorial with Python Code \- Chris McCormick, accessed May 29, 2025, [https://mccormickml.com/2015/06/12/minhash-tutorial-with-python-code/](https://mccormickml.com/2015/06/12/minhash-tutorial-with-python-code/)  
3. MinHash — How To Find Similarity At Scale \[Python Tutorial\] \- Spot Intelligence, accessed May 29, 2025, [https://spotintelligence.com/2023/01/02/minhash/](https://spotintelligence.com/2023/01/02/minhash/)  
4. MinHash Similarity | duvni.io, accessed May 29, 2025, [https://duvni.io/2017/10/minhash-similarity/](https://duvni.io/2017/10/minhash-similarity/)  
5. MinHash \- Fast Jaccard Similarity at Scale \- Arpit Bhayani, accessed May 29, 2025, [https://arpitbhayani.me/blogs/jaccard-minhash/](https://arpitbhayani.me/blogs/jaccard-minhash/)  
6. Shingling and MinHash \- INF/PUC-Rio, accessed May 29, 2025, [https://www-di.inf.puc-rio.br/\~casanova/Disciplinas/INF1331/Slides/03.2-03.3-Shingling-MinHash.pdf](https://www-di.inf.puc-rio.br/~casanova/Disciplinas/INF1331/Slides/03.2-03.3-Shingling-MinHash.pdf)  
7. Maintaining k-MinHash Signatures over Fully-Dynamic Data Streams with Recovery \- arXiv, accessed May 29, 2025, [https://arxiv.org/pdf/2407.21614](https://arxiv.org/pdf/2407.21614)  
8. Lecture 12 1 Review: Jaccard Similarity 2 Revew: MinHash 3 Parity of MinHash \- Rice University, accessed May 29, 2025, [https://www.cs.rice.edu/\~as143/COMP480\_580\_Spring19/scribe/S12.pdf](https://www.cs.rice.edu/~as143/COMP480_580_Spring19/scribe/S12.pdf)  
9. Locality Sensitive Hashing (LSH): The Illustrated Guide \- Pinecone, accessed May 29, 2025, [https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/)  
10. CS494 Lecture Notes \- MinHash, accessed May 29, 2025, [https://web.eecs.utk.edu/\~jplank/plank/classes/cs494/494/notes/Min-Hash/index.html](https://web.eecs.utk.edu/~jplank/plank/classes/cs494/494/notes/Min-Hash/index.html)  
11. leodemoura.github.io, accessed May 29, 2025, [https://leodemoura.github.io/files/ICSM98.pdf](https://leodemoura.github.io/files/ICSM98.pdf)  
12. Beyond Surveys: A Deep Analysis of Persistent Challenges in Vulnerable Code Clone Detection, accessed May 29, 2025, [https://eudoxuspress.com/index.php/pub/article/download/2324/1561/4463](https://eudoxuspress.com/index.php/pub/article/download/2324/1561/4463)  
13. A Support Vector Machine based approach for plagiarism detection in Python code submissions in undergraduate settings \- Frontiers, accessed May 29, 2025, [https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2024.1393723/full](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2024.1393723/full)  
14. Efficient Textual Similarity using Semantic MinHashing \- BCU Open Access Repository \- Birmingham City University, accessed May 29, 2025, [https://www.open-access.bcu.ac.uk/16059/2/Efficient\_Textual\_Similarity\_using\_Semantic\_MinHashing.pdf](https://www.open-access.bcu.ac.uk/16059/2/Efficient_Textual_Similarity_using_Semantic_MinHashing.pdf)  
15. Locality-sensitive hashing for the edit distance \- PMC \- PubMed Central, accessed May 29, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6612865/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6612865/)  
16. Jaccard Similarity | GeeksforGeeks, accessed May 29, 2025, [https://www.geeksforgeeks.org/jaccard-similarity/](https://www.geeksforgeeks.org/jaccard-similarity/)  
17. Efficient implementation of MinHash, part 1 \- Andrei Gudkov, accessed May 29, 2025, [https://gudok.xyz/minhash1/](https://gudok.xyz/minhash1/)  
18. Minhash and locality-sensitive hashing • textreuse \- Docs, accessed May 29, 2025, [https://docs.ropensci.org/textreuse/articles/textreuse-minhash.html](https://docs.ropensci.org/textreuse/articles/textreuse-minhash.html)  
19. Minhash LSH Implementation Walkthrough: Deduplication \- DZone, accessed May 29, 2025, [https://dzone.com/articles/minhash-lsh-implementation-walkthrough](https://dzone.com/articles/minhash-lsh-implementation-walkthrough)  
20. Abstract syntax tree \- Wikipedia, accessed May 29, 2025, [https://en.wikipedia.org/wiki/Abstract\_syntax\_tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree)  
21. Advancing Vulnerability Detection: An Innovative Approach to Generate Embeddings of Code Snippets \- International Journal of Intelligent Systems and Applications in Engineering, accessed May 29, 2025, [https://ijisae.org/index.php/IJISAE/article/download/6423/5253/11574](https://ijisae.org/index.php/IJISAE/article/download/6423/5253/11574)  
22. Introduction to the Clang AST — Clang 21.0.0git documentation, accessed May 29, 2025, [https://clang.llvm.org/docs/IntroductionToTheClangAST.html](https://clang.llvm.org/docs/IntroductionToTheClangAST.html)  
23. MISIM: A NOVEL CODE SIMILARITY SYSTEM \- OpenReview, accessed May 29, 2025, [https://openreview.net/pdf?id=AZ4vmLoJft](https://openreview.net/pdf?id=AZ4vmLoJft)  
24. A Systematic Review on Code Clone Detection \- SciSpace, accessed May 29, 2025, [https://scispace.com/pdf/a-systematic-review-on-code-clone-detection-51ueuzckrv.pdf](https://scispace.com/pdf/a-systematic-review-on-code-clone-detection-51ueuzckrv.pdf)  
25. Revisiting Code Similarity Evaluation with Abstract Syntax Tree Edit Distance, accessed May 29, 2025, [https://aclanthology.org/2024.luhme-short.3/](https://aclanthology.org/2024.luhme-short.3/)  
26. Tutorial for building tools using LibTooling and LibASTMatchers \- Clang \- LLVM, accessed May 29, 2025, [https://clang.llvm.org/docs/LibASTMatchersTutorial.html](https://clang.llvm.org/docs/LibASTMatchersTutorial.html)  
27. Introduction to Embedding, Clustering, and Similarity | Towards Data Science, accessed May 29, 2025, [https://towardsdatascience.com/introduction-to-embedding-clustering-and-similarity-11dd80b00061/](https://towardsdatascience.com/introduction-to-embedding-clustering-and-similarity-11dd80b00061/)  
28. Ultimate Distance Metrics for Clustering \- Number Analytics, accessed May 29, 2025, [https://www.numberanalytics.com/blog/distance-metrics-clustering-guide](https://www.numberanalytics.com/blog/distance-metrics-clustering-guide)  
29. Detecting Semantic Code Clones by Building AST-based Markov Chains Model \- Yueming Wu, accessed May 29, 2025, [https://wu-yueming.github.io/Files/ASE2022\_Amain.pdf](https://wu-yueming.github.io/Files/ASE2022_Amain.pdf)  
30. Graph Kernels \- Journal of Machine Learning Research, accessed May 29, 2025, [https://jmlr.org/papers/volume11/vishwanathan10a/vishwanathan10a.pdf](https://jmlr.org/papers/volume11/vishwanathan10a/vishwanathan10a.pdf)  
31. G-Hash: Towards Fast Kernel-based Similarity Search in Large Graph Databases \- PMC, accessed May 29, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC2860326/](https://pmc.ncbi.nlm.nih.gov/articles/PMC2860326/)  
32. CCGraph: a PDG-based code clone detector with approximate graph matching \- Yinxing Xue's Homepage, accessed May 29, 2025, [https://yinxingxue.github.io/papers/ase2020\_CCGraph%20A%20PDG%20based%20Code%20Clone%20Detector%20With%20Approximate%20Graph%20Matching.pdf](https://yinxingxue.github.io/papers/ase2020_CCGraph%20A%20PDG%20based%20Code%20Clone%20Detector%20With%20Approximate%20Graph%20Matching.pdf)  
33. Control Flow Graph (CFG) – Software Engineering | GeeksforGeeks, accessed May 29, 2025, [https://www.geeksforgeeks.org/software-engineering-control-flow-graph-cfg/](https://www.geeksforgeeks.org/software-engineering-control-flow-graph-cfg/)  
34. cfgsim.cs.arizona.edu, accessed May 29, 2025, [http://cfgsim.cs.arizona.edu/qsic14.pdf](http://cfgsim.cs.arizona.edu/qsic14.pdf)  
35. Can Large Language Models Understand Intermediate Representations? \- arXiv, accessed May 29, 2025, [https://arxiv.org/html/2502.06854v1](https://arxiv.org/html/2502.06854v1)  
36. Code Complexity: An In-Depth Explanation and Metrics \- Codacy | Blog, accessed May 29, 2025, [https://blog.codacy.com/code-complexity](https://blog.codacy.com/code-complexity)  
37. Control flow \- Wikipedia, accessed May 29, 2025, [https://en.wikipedia.org/wiki/Control\_flow](https://en.wikipedia.org/wiki/Control_flow)  
38. Optimizing Code Runtime Performance through Context-Aware Retrieval-Augmented Generation \- arXiv, accessed May 29, 2025, [https://arxiv.org/pdf/2501.16692](https://arxiv.org/pdf/2501.16692)  
39. C++ Loops | GeeksforGeeks, accessed May 29, 2025, [https://www.geeksforgeeks.org/cpp-loops/](https://www.geeksforgeeks.org/cpp-loops/)  
40. Refining the Control Structure of Loops using Static Analysis. \- Stanford CS Theory, accessed May 29, 2025, [http://theory.stanford.edu/\~srirams/papers/loopRefinement2009.pdf](http://theory.stanford.edu/~srirams/papers/loopRefinement2009.pdf)  
41. A Structural Approach to Program Similarity Analysis \- CEUR-WS.org, accessed May 29, 2025, [https://ceur-ws.org/Vol-3845/paper01.pdf](https://ceur-ws.org/Vol-3845/paper01.pdf)  
42. Identifying Similar Code with Program Dependence Graphs | Request PDF \- ResearchGate, accessed May 29, 2025, [https://www.researchgate.net/publication/3919783\_Identifying\_Similar\_Code\_with\_Program\_Dependence\_Graphs](https://www.researchgate.net/publication/3919783_Identifying_Similar_Code_with_Program_Dependence_Graphs)  
43. Code metrics \- Cyclomatic complexity \- Visual Studio (Windows) | Microsoft Learn, accessed May 29, 2025, [https://learn.microsoft.com/en-us/visualstudio/code-quality/code-metrics-cyclomatic-complexity?view=vs-2022](https://learn.microsoft.com/en-us/visualstudio/code-quality/code-metrics-cyclomatic-complexity?view=vs-2022)  
44. Code Clone Detection and Analysis Using Software Metrics and Neural Network-A Literature Review, accessed May 29, 2025, [https://www.ijcstjournal.org/volume-3/issue-2/IJCST-V3I2P26.pdf](https://www.ijcstjournal.org/volume-3/issue-2/IJCST-V3I2P26.pdf)  
45. Understanding Code Metrics in CppDepend, accessed May 29, 2025, [https://www.cppdepend.com/documentation/code-metrics](https://www.cppdepend.com/documentation/code-metrics)  
46. Implementing a high-efficiency similarity analysis approach for firmware code \- PMC, accessed May 29, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7802928/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7802928/)  
47. GPLAG: detection of software plagiarism by program dependence graph analysis \- SciSpace, accessed May 29, 2025, [https://scispace.com/pdf/gplag-detection-of-software-plagiarism-by-program-dependence-4cms55onj4.pdf](https://scispace.com/pdf/gplag-detection-of-software-plagiarism-by-program-dependence-4cms55onj4.pdf)  
48. How to generate program dependence graph for C program? \- Stack Overflow, accessed May 29, 2025, [https://stackoverflow.com/questions/7904651/how-to-generate-program-dependence-graph-for-c-program](https://stackoverflow.com/questions/7904651/how-to-generate-program-dependence-graph-for-c-program)  
49. Optimising source code vulnerability detection using deep learning and deep graph network, accessed May 29, 2025, [https://www.tandfonline.com/doi/full/10.1080/09540091.2024.2447373?af=R](https://www.tandfonline.com/doi/full/10.1080/09540091.2024.2447373?af=R)  
50. Joern Documentation: Overview, accessed May 29, 2025, [https://docs.joern.io/](https://docs.joern.io/)  
51. Why You Should Add Joern to Your Source Code Audit Toolkit | Praetorian, accessed May 29, 2025, [https://www.praetorian.com/blog/why-you-should-add-joern-to-your-source-code-audit-toolkit/](https://www.praetorian.com/blog/why-you-should-add-joern-to-your-source-code-audit-toolkit/)  
52. Learning Type Inference for Enhanced Dataflow Analysis, accessed May 29, 2025, [https://www.mlsec.org/docs/2023-esorics.pdf](https://www.mlsec.org/docs/2023-esorics.pdf)  
53. Deep Learning for Code Intelligence: Survey, Benchmark and Toolkit \- arXiv, accessed May 29, 2025, [https://arxiv.org/html/2401.00288v1](https://arxiv.org/html/2401.00288v1)  
54. cnclabs/smore: SMORe: Modularize Graph Embedding for Recommendation \- GitHub, accessed May 29, 2025, [https://github.com/cnclabs/smore](https://github.com/cnclabs/smore)  
55. GraphBinMatch: Graph-based Similarity Learning for Cross-Language Binary and Source Code Matching, accessed May 29, 2025, [https://swapp.cs.iastate.edu/files/inline-files/Tehrani-ea-GraphBinMatch\_arXiv-April-2023.pdf](https://swapp.cs.iastate.edu/files/inline-files/Tehrani-ea-GraphBinMatch_arXiv-April-2023.pdf)  
56. Weisong Sun \- A Survey of Source Code Search: A 3-Dimensional Perspective, accessed May 29, 2025, [https://wssun.github.io/papers/2024-TOSEM-CodeSearchSurvey.pdf](https://wssun.github.io/papers/2024-TOSEM-CodeSearchSurvey.pdf)  
57. NeurIPS Poster Suitable is the Best: Task-Oriented Knowledge Fusion in Vulnerability Detection, accessed May 29, 2025, [https://neurips.cc/virtual/2024/poster/95375](https://neurips.cc/virtual/2024/poster/95375)  
58. A New Framework for Software Vulnerability Detection Based on an Advanced Computing, accessed May 29, 2025, [https://www.techscience.com/cmc/v79n3/57117/html](https://www.techscience.com/cmc/v79n3/57117/html)  
59. data-juicer/data\_juicer/ops/deduplicator/ray\_bts\_minhash\_deduplicator.py at main \- GitHub, accessed May 29, 2025, [https://github.com/modelscope/data-juicer/blob/main/data\_juicer/ops/deduplicator/ray\_bts\_minhash\_deduplicator.py](https://github.com/modelscope/data-juicer/blob/main/data_juicer/ops/deduplicator/ray_bts_minhash_deduplicator.py)  
60. shaltielshmid/MinHashSharp: A Robust Library in C\# for Similarity Estimation \- GitHub, accessed May 29, 2025, [https://github.com/shaltielshmid/MinHashSharp](https://github.com/shaltielshmid/MinHashSharp)  
61. What Is Vector Similarity Search? Benefits & Applications \- Couchbase, accessed May 29, 2025, [https://www.couchbase.com/blog/vector-similarity-search/](https://www.couchbase.com/blog/vector-similarity-search/)  
62. Ordered Semantically Diverse Sampling for Textual Data \- arXiv, accessed May 29, 2025, [https://arxiv.org/html/2503.10698](https://arxiv.org/html/2503.10698)  
63. arXiv:2503.10698v1 \[cs.CL\] 12 Mar 2025, accessed May 29, 2025, [https://arxiv.org/pdf/2503.10698](https://arxiv.org/pdf/2503.10698)  
64. EXAMPLE Feature Extraction/Selection & Clustering \- Kaggle, accessed May 29, 2025, [https://www.kaggle.com/code/unmoved/example-feature-extraction-selection-clustering](https://www.kaggle.com/code/unmoved/example-feature-extraction-selection-clustering)  
65. Clustering in Machine Learning | GeeksforGeeks, accessed May 29, 2025, [https://www.geeksforgeeks.org/clustering-in-machine-learning/](https://www.geeksforgeeks.org/clustering-in-machine-learning/)  
66. duhaime/minhash: Quickly estimate the similarity between many sets \- GitHub, accessed May 29, 2025, [https://github.com/duhaime/minhash](https://github.com/duhaime/minhash)  
67. Generate an AST in C++ \- Stack Overflow, accessed May 29, 2025, [https://stackoverflow.com/questions/28530284/generate-an-ast-in-c](https://stackoverflow.com/questions/28530284/generate-an-ast-in-c)  
68. standardese/cppast: Library to parse and work with the C++ AST \- GitHub, accessed May 29, 2025, [https://github.com/standardese/cppast](https://github.com/standardese/cppast)  
69. Libclang tutorial — Clang 21.0.0git documentation, accessed May 29, 2025, [https://clang.llvm.org/docs/LibClang.html](https://clang.llvm.org/docs/LibClang.html)  
70. LibToolingExample/Example.cpp at master \- GitHub, accessed May 29, 2025, [https://github.com/kevinaboos/LibToolingExample/blob/master/Example.cpp](https://github.com/kevinaboos/LibToolingExample/blob/master/Example.cpp)  
71. zzhzz/clang-cfg: A tool based on Clang used to generate control-flow-graph for C/C++ code, accessed May 29, 2025, [https://github.com/zzhzz/clang-cfg](https://github.com/zzhzz/clang-cfg)  
72. LibTooling — Clang 21.0.0git documentation, accessed May 29, 2025, [https://clang.llvm.org/docs/LibTooling.html](https://clang.llvm.org/docs/LibTooling.html)  
73. zhaosiying12138/PDG\_demo: A toy implementation about Program Dependence Graph using LLVM \- GitHub, accessed May 29, 2025, [https://github.com/zhaosiying12138/PDG\_demo](https://github.com/zhaosiying12138/PDG_demo)  
74. generating CFG for whole source code with LLVM \- Stack Overflow, accessed May 29, 2025, [https://stackoverflow.com/questions/26556356/generating-cfg-for-whole-source-code-with-llvm](https://stackoverflow.com/questions/26556356/generating-cfg-for-whole-source-code-with-llvm)  
75. LLVM traverse CFG \- Stack Overflow, accessed May 29, 2025, [https://stackoverflow.com/questions/15433417/llvm-traverse-cfg](https://stackoverflow.com/questions/15433417/llvm-traverse-cfg)  
76. LLVM's Analysis and Transform Passes — LLVM 21.0.0git documentation, accessed May 29, 2025, [https://llvm.org/docs/Passes.html](https://llvm.org/docs/Passes.html)  
77. Top 9 C++ Static Code Analysis Tools \- Incredibuild, accessed May 29, 2025, [https://www.incredibuild.com/blog/top-9-c-static-code-analysis-tools](https://www.incredibuild.com/blog/top-9-c-static-code-analysis-tools)  
78. List of tools for static code analysis \- Wikipedia, accessed May 29, 2025, [https://en.wikipedia.org/wiki/List\_of\_tools\_for\_static\_code\_analysis](https://en.wikipedia.org/wiki/List_of_tools_for_static_code_analysis)  
79. k-means clustering \- Wikipedia, accessed May 29, 2025, [https://en.wikipedia.org/wiki/K-means\_clustering](https://en.wikipedia.org/wiki/K-means_clustering)  
80. Implementing k-means clustering from scratch in C++ \- Reasonable Deviations, accessed May 29, 2025, [https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/](https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/)  
81. K-means++ Algorithm – ML | GeeksforGeeks, accessed May 29, 2025, [https://www.geeksforgeeks.org/ml-k-means-algorithm/](https://www.geeksforgeeks.org/ml-k-means-algorithm/)  
82. Bayesian Selection for Efficient MLIP Dataset Selection \- arXiv, accessed May 29, 2025, [https://arxiv.org/pdf/2502.21165](https://arxiv.org/pdf/2502.21165)  
83. k-medoids Archives \- QUSMA, accessed May 29, 2025, [https://qusma.com/tag/k-medoids/](https://qusma.com/tag/k-medoids/)  
84. Lallapallooza/clustering: Header-Only Collection of Clustering Algorithms for C++ \- GitHub, accessed May 29, 2025, [https://github.com/Lallapallooza/clustering](https://github.com/Lallapallooza/clustering)  
85. Diversity Maximization in the Presence of Outliers, accessed May 29, 2025, [https://ojs.aaai.org/index.php/AAAI/article/view/26454/26226](https://ojs.aaai.org/index.php/AAAI/article/view/26454/26226)  
86. MapReduce and Streaming Algorithms for Diversity Maximization in Metric Spaces of Bounded Doubling Dimension \- VLDB Endowment, accessed May 29, 2025, [http://www.vldb.org/pvldb/vol10/p469-ceccarello.pdf](http://www.vldb.org/pvldb/vol10/p469-ceccarello.pdf)