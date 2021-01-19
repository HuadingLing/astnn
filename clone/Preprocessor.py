


def expand_dict(d, seq):
    for i in d.keys():
        if isinstance(d[i], dict):
            seq.append(i)
            
def expand(nested_list):  # 生成器，用于展开嵌套的list
    for item in nested_list:
        if isinstance(item, list):
            yield from expand(item)
        elif item:
            yield item
            
            
#################################################################################################            
# For Python
#Logic1 = ['For', 'AsyncFor', 'While', 'If', 'With', 'AsyncWith', 'Try']
#Logic2 = ['FunctionDef', 'AsyncFunctionDef']
EXCLUDE_FIELDS = {'FunctionDef': ['body', 'decorator_list'],
                  'AsyncFunctionDef': ['body', 'decorator_list'],
                  'For': ['body', 'orelse'],
                  'AsyncFor': ['body', 'orelse'],
                  'While': ['body', 'orelse'], 
                  'If': ['body', 'orelse'],
                  'With': ['body'],
                  'AsyncWith': ['body'],
                  'Try': ['body', 'handlers', 'orelse', 'finalbody'],
                  'ExceptHandler': ['body']
                 }

class Node_python(object):
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(node, str)  # str => 叶子节点 => 无孩子节点
        self.token = self.get_token(node)
        self.children = self.add_children()

    def is_leaf(self):
        if self.is_str:
            return True
        return len(self.node.children) == 0

    def get_token(self, node):
        if self.is_str:
            return self.node
        import _ast
        if isinstance(node, _ast.AST):
            token = node.__class__.__name__
        else:
            try:
                token = str(node)
            except:
                token = ''
        return token

    def iter_fields(self, node, exclude=[]):
        """
        Yield a tuple of ``(fieldname, value)`` for each field in ``node._fields``
        that is present on *node*.
        """
        for field in [f for f in node._fields if f not in exclude]:
            try:
                yield getattr(node, field)
            except AttributeError:
                pass
            
    def ori_children(self, root, exclude=[]):
        import _ast
        #if isinstance(root, str):
        #    children = []
        if isinstance(root, list):
            children = root
        elif isinstance(root, _ast.AST):
            #children = [getattr(root, f) for f in root._fields if getattr(root, f)]
            children = self.iter_fields(root, exclude)
        elif isinstance(root, set):  # 猜测这条规则不会触发吧
            children = list(root)
        else:
            children = []
        return list(expand(children))

    def add_children(self):
        if self.is_str:  # str => 叶子节点 => 无孩子节点
            return []
        if self.token in EXCLUDE_FIELDS.keys():
            ef = EXCLUDE_FIELDS[self.token]
        else:
            ef = []
        children = self.ori_children(self.node, ef)
        return [Node_python(child) for child in children]
    

def fun_cleanout(code, fun_pos, mode=None, comment_open_close_pattern=None, comment_inline_pattern=None):
    if mode is None:  # 只清除结尾多余的空行
        pass
    elif mode == 'tailer_comment':  # 只清除结尾多余的空行和注释
        pass
    elif mode == 'comment': # 清除所有的注释
        pass
    elif mode == 'all':  # 清除所有注释和空行（紧凑形式）
        pass
    else:
        pass



# For Python
class Preprocessor_python():
    def __init__(self, vocab):
        self.vocab = vocab
        self.max_token = len(vocab)
    
    '''
    以下5个是从file到function_ast，也就是把一个文件中代码解析成AST并拆分出各个函数
    '''
    def file_check(self, filename):
        return filename[-3:].lower() == '.py'
    
    def file_to_code(self, filename):
        import os
        #assert os.path.isfile(filename)  # 对超长路径会出错
        assert self.file_check(filename)
        try:
            with open(filename, 'r', encoding="utf-8") as f:
                code = f.read()
            return code
        except:
            return None

    def file_to_ast(self, file):
        ast = self.code_to_ast(self.file_to_code(file))
        return ast
    
    def code_to_ast(self, code):
        from Lib import ast as python_ast
        ast = python_ast.parse(code)
        return ast
        
    def get_functions(self, ast):
        function_asts = []
        for c in ast.body:
            if c.__class__.__name__ == 'FunctionDef':
                function_asts.append(c)
            elif c.__class__.__name__ == 'ClassDef':
                function_asts.extend(self.get_functions(c))
        return function_asts

    def get_function_name(self, function_ast):
        return function_ast.name
    
    def extract_functions(self, code):
        '''
        输入代码，输出两个list：一个是函数的ast，一个是函数起止行
        '''
        import ast
        tree = None
        try:
            tree = ast.parse(code)
        except Exception as e:
            pass
            return None, None

        linecount = code.count("\n")
        if not code.endswith("\n"):
            linecount += 1

        function_nodes = []
        function_pos = []

        for index, stmt in enumerate(tree.body):
            if isinstance(stmt, ast.ClassDef):
                for idx, s in enumerate(stmt.body):
                    if isinstance(s, ast.FunctionDef):
                        start_lineno =  s.lineno
                        if idx == len(stmt.body)-1:
                            # this is the last one in stmt.body
                            if index == len(tree.body)-1:
                                # also the last stmt in tree.body
                                end_lineno = linecount
                            else:
                                # but not the last stmt in tree.body
                                end_lineno =  tree.body[index+1].lineno-1
                        else:
                            #not the last one in stmt.body
                            end_lineno = stmt.body[idx+1].lineno-1
                        function_nodes.append(s)
                        function_pos.append((start_lineno, end_lineno))

            if isinstance(stmt, ast.FunctionDef):
                start_lineno =  stmt.lineno
                if index == len(tree.body)-1:
                    # the last stmt in tree.body
                    end_lineno = linecount
                else:
                    end_lineno = tree.body[index+1].lineno-1
                function_nodes.append(stmt)
                function_pos.append((start_lineno, end_lineno))

        return function_nodes, function_pos
    

    '''
    以下两个函数分别获取节点token和孩子节点，是后续其他操作的基础
    '''
    def get_token(self, node):
        import _ast
        if isinstance(node, str):
            token = node
        elif isinstance(node, _ast.AST):
            token = node.__class__.__name__
        else:
            try:
                token = str(node)
            except:
                token = ''
        return token
        
    def iter_fields(self, node, exclude=[]):
        """
        Yield a tuple of ``(fieldname, value)`` for each field in ``node._fields``
        that is present on *node*.
        """
        for field in [f for f in node._fields if f not in exclude]:
            try:
                yield getattr(node, field)
            except AttributeError:
                pass
            
    def get_children(self, root, exclude=[]):
        import _ast
        #if isinstance(root, str):
        #    children = []
        if isinstance(root, list):
            children = root
        elif isinstance(root, _ast.AST):
            #children = [getattr(root, f) for f in root._fields if getattr(root, f)]
            children = self.iter_fields(root, exclude)
        elif isinstance(root, set):  # 猜测这条规则不会触发吧
            children = list(root)
        else:
            children = []
        return list(expand(children))

    
    '''
    以下函数，都是把ast变成ASTNN的输入结构，但是前两个使用token index，后两个使用token本身
    主要用前两个
    后两个是为了打印出来方便调试
    '''
    def ast_to_block(self, ast):
        blocks = []
        self.get_blocks(ast, blocks)
        tree = []
        for node in blocks:  # 这里的每个node就是对应一行代码？
            btree = self.replaced_by_index(node)
            tree.append(btree)
        return tree
    
    def replaced_by_index(self, node):
        # 返回的形式：[node, children1, children2, ...]
        # 一个大list，每个children又是一个子list
        token = node.token
        result = [self.vocab[token].index if token in self.vocab else self.max_token]
        children = node.children
        for child in children:
            result.append(self.replaced_by_index(child))
        return result
    
    def ast_to_token_block(self, ast):
        blocks = []
        self.get_blocks(ast, blocks)
        tree = []
        for node in blocks:
            btree = self.replaced_by_token(node)
            tree.append(btree)
        return tree
    
    def replaced_by_token(self, node):
        #result = [node.token if node.token in self.vocab else 'UNKNOWN']
        result = [node.token]
        children = node.children
        for child in children:
            result.append(self.replaced_by_token(child))
        return result
    
    
    '''
    最复杂的东东，根据当前的 node 获得一个或多个 Node_python 并添加到 block_seq 里面
    block_seq.append 的必定是一个 Node_python，可以猜测 Node_python 是一个把 ast.Node 转化成自定义的节点类
    '''
    def get_blocks(self, node, block_seq):
        name = self.get_token(node)
        block_seq.append(Node_python(node))
        if name in EXCLUDE_FIELDS.keys():
            ef = [f for f in node._fields if f not in EXCLUDE_FIELDS[name]]
            children = self.get_children(node, ef)
            for child in children:
                self.get_blocks(child, block_seq)
    
    '''
    以下两个函数对ast进行先序遍历获得先序遍历的token序列，用于 word embedding 的训练
    '''
    def get_sequence(self, node, sequence):  # 获取先序遍历结果，同时为一些特殊代码块加上'End'
        token, children = self.get_token(node), self.get_children(node)
        sequence.append(token)

        for child in children:
            self.get_sequence(child, sequence)

    def trans_to_sequences(self, ast):
        # 这个用于生成token列表，用于 word embedding 的训练
        sequence = []
        self.get_sequence(ast, sequence)  # 从根节点开始先序遍历
        return sequence
    
    
    '''
    以下函数待定，只是为了打印出来看一下ast或者block之类的，方便调试
    '''
    def visit_block(self, block):
        pass
        
    def visit_token_block(self, block):
        pass
    
    def block_to_embedded(self):
        pass
    
    
    
########################################################################################
# For C++
class Node_cpp(object):
    def __init__(self, node, add_children = True):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token()
        if add_children:
            self.children = self.add_children()
        else:
            self.children = []

    def is_leaf(self):
        if self.is_str:
            return True
        return len(self.node.children()) == 0

    def get_token(self, lower=True):
        if self.is_str:
            return self.node
        name = self.node.__class__.__name__
        token = name
        is_name = False
        if self.is_leaf():
            attr_names = self.node.attr_names
            if attr_names:
                if 'names' in attr_names:
                    token = self.node.names[0]
                elif 'name' in attr_names:
                    token = self.node.name
                    is_name = True
                else:
                    token = self.node.value
            else:
                token = name
        else:
            if name == 'TypeDecl':
                token = self.node.declname
            if self.node.attr_names:
                attr_names = self.node.attr_names
                if 'op' in attr_names:
                    if self.node.op[0] == 'p':
                        token = self.node.op[1:]
                    else:
                        token = self.node.op
        if token == None:
            token = name
        if lower and is_name:
            token = token.lower()
        return token

    def add_children(self):
        if self.is_str:
            return []
        children = self.node.children()
        if self.token in ['FuncDef', 'If', 'While', 'DoWhile']:
            return [Node_cpp(children[0][1])]
        elif self.token == 'For':
            return [Node_cpp(children[c][1]) for c in range(0, len(children)-1)]
        else:
            return [Node_cpp(child) for _, child in children]
        
# For C++
class Preprocessor_cpp():
    def __init__(self, vocab):
        self.vocab = vocab
        self.max_token = len(vocab)
    
    '''
    以下5个是从file到function_ast，也就是把一个文件中代码解析成AST并拆分出各个函数
    '''
    def file_check(self, filename):
        return filename[-4:].lower()=='.cpp'
    
    def file_to_code(self, filename):
        import os
        #assert os.path.isfile(filename)
        assert self.file_check(filename)
        try:
            with open(filename, 'r', encoding="utf-8") as f:
                code = f.read()
            return code
        except:
            return None

    def file_to_ast(self, file):
        #ast = self.code_to_ast(self.file_to_code(file))
        from pycparser import parse_file
        ast = parse_file(file, use_cpp=False,
            cpp_path='cpp'),
            #cpp_args=r'-Iutils/fake_libc_include')
        return ast
    
    def code_to_ast(self, code):
        from pycparser import c_parser
        parser = c_parser.CParser()
        ast = parser.parse(code)
        return ast
        '''
        try:
            from pycparser import c_parser
            parser = c_parser.CParser()
            ast = parser.parse(code)
            return ast
        except:
            return None
        '''


    def get_functions(self, ast):
        # 只要 function
        return [func_ast for func_ast in ast.ext if func_ast.__class__.__name__ == 'FuncDef']

    def get_function_name(self, function_ast):
        return function_ast.decl.name
    
    def extract_functions(self, code):
        '''
        输入代码，输出两个list：一个是函数的ast，一个是函数起止行
        '''
        import clang
        import clang.cindex
        from clang.cindex import CursorKind
        
        
        function_pos = []
        function_nodes = []
        try:
            index = clang.cindex.Index.create()
            tu = index.parse(path='0.cpp', unsaved_files=[('0.cpp',code)])
        except Exception as e:
            pass
            return None, None

        AST_root_node= tu.cursor
        file_string_split = code.split('\n')
        linecount = code.count("\n")
        if not code.endswith("\n"):
            linecount += 1
        ast_list = list(AST_root_node.get_children())

        for idx, cur in enumerate(ast_list):
            if cur.kind == CursorKind.FUNCTION_DECL:
                start_lineno = cur.location.line
                if idx == len(ast_list) - 1:
                    end_lineno = linecount
                else:
                    end_lineno = ast_list[idx+1].location.line - 1
                function_nodes.append(cur)
                function_pos.append((start_lineno, end_lineno))
                
            elif cur.kind == CursorKind.CLASS_DECL:
                ast_list_in_class = list(cur.get_children())
                for idx_in_class, cur_in_class in enumerate(ast_list_in_class):
                    if cur_in_class.kind == CursorKind.CXX_METHOD:
                        start_lineno = cur_in_class.location.line
                        if idx_in_class == len(ast_list_in_class) - 1: 
                            if idx == len(ast_list) - 1:
                                end_lineno = linecount
                            else:
                                end_lineno = ast_list[idx+1].location.line - 1
                            for lineno in range(end_lineno-1, 0, -1):
                                if file_string_split[lineno] and file_string_split[lineno][0]=='}':
                                    end_lineno = lineno
                                    break
                        else:
                            end_lineno = ast_list_in_class[idx_in_class+1].location.line - 1
                        function_nodes.append(cur_in_class)
                        function_pos.append((start_lineno, end_lineno))

        return function_nodes, function_pos

    
    '''
    以下函数，都是把ast变成ASTNN的输入结构，但是前两个使用token index，后两个使用token本身
    主要用前两个
    后两个是为了打印出来方便调试
    '''
    def ast_to_block(self, ast):
        blocks = []
        self.get_blocks(ast, blocks)
        tree = []
        for node in blocks:  # 这里的每个node就是对应一行代码？
            btree = self.replaced_by_index(node)
            tree.append(btree)
        return tree
    
    def replaced_by_index(self, node):
        # 返回的形式：[node, children1, children2, ...]
        # 一个大list，每个children又是一个子list
        token = node.token
        result = [self.vocab[token].index if token in self.vocab else self.max_token]
        children = node.children
        for child in children:
            result.append(self.replaced_by_index(child))
        return result
    
    def ast_to_token_block(self, ast):
        blocks = []
        self.get_blocks(ast, blocks)
        tree = []
        for node in blocks:
            btree = self.replaced_by_token(node)
            tree.append(btree)
        return tree
    
    def replaced_by_token(self, node):
        result = [node.token if node.token in self.vocab else 'UNKNOWN']
        children = node.children
        for child in children:
            result.append(self.replaced_by_token(child))
        return result
    
    
    '''
    
    '''
    def get_blocks(self, node, block_seq):
        children = node.children()
        name = node.__class__.__name__
        if name in ['FuncDef', 'If', 'For', 'While', 'DoWhile']:
            block_seq.append(Node_cpp(node))
            if name != 'For':
                skip = 1
            else:
                skip = len(children) - 1

            for i in range(skip, len(children)):
                child = children[i][1]
                if child.__class__.__name__ not in ['FuncDef', 'If', 'For', 'While', 'DoWhile', 'Compound']:
                    block_seq.append(Node_cpp(child))
                self.get_blocks(child, block_seq)
        elif name == 'Compound':
            block_seq.append(Node_cpp(name))
            for _, child in node.children():
                if child.__class__.__name__ not in ['If', 'For', 'While', 'DoWhile']:
                    block_seq.append(Node_cpp(child))
                self.get_blocks(child, block_seq)
            block_seq.append(Node_cpp('End'))
        else:
            for _, child in node.children():
                self.get_blocks(child, block_seq)

    
    '''
    以下两个函数对ast进行先序遍历获得先序遍历的token序列，用于 word embedding 的训练
    '''
    def get_sequence(self, node, sequence):
        current = Node_cpp(node, False)
        sequence.append(current.get_token())
        for _, child in node.children():
            self.get_sequence(child, sequence)
        if current.get_token().lower() == 'compound':
            sequence.append('End')
            # compound 代码段后面要加 End

    def trans_to_sequences(self, ast):
        # 这个用于生成token列表，用于 word embedding 的训练
        sequence = []
        self.get_sequence(ast, sequence)  # 从根节点开始先序遍历
        return sequence
    
    
    '''
    以下函数待定，只是为了打印出来看一下ast或者block之类的，方便调试
    '''
    def visit_block(self, block):
        pass
        
    def visit_token_block(self, block):
        block.show()
    
    def block_to_embedded(self):
        pass
    
    
    
########################################################################################
# For C
class Node_c(object):
    def __init__(self, node, add_children = True):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token()
        if add_children:
            self.children = self.add_children()
        else:
            self.children = []

    def is_leaf(self):
        if self.is_str:
            return True
        return len(self.node.children()) == 0

    def get_token(self, lower=True):
        if self.is_str:
            return self.node
        name = self.node.__class__.__name__
        token = name
        is_name = False
        if self.is_leaf():
            attr_names = self.node.attr_names
            if attr_names:
                if 'names' in attr_names:
                    token = self.node.names[0]
                elif 'name' in attr_names:
                    token = self.node.name
                    is_name = True
                else:
                    token = self.node.value
            else:
                token = name
        else:
            if name == 'TypeDecl':
                token = self.node.declname
            if self.node.attr_names:
                attr_names = self.node.attr_names
                if 'op' in attr_names:
                    if self.node.op[0] == 'p':
                        token = self.node.op[1:]
                    else:
                        token = self.node.op
        if token == None:
            token = name
        if lower and is_name:
            token = token.lower()
        return token

    def add_children(self):
        if self.is_str:
            return []
        children = self.node.children()
        if self.token in ['FuncDef', 'If', 'While', 'DoWhile']:
            return [Node_c(children[0][1])]
        elif self.token == 'For':
            return [Node_c(children[c][1]) for c in range(0, len(children)-1)]
        else:
            return [Node_c(child) for _, child in children]
        
# For C
class Preprocessor_c():
    def __init__(self, vocab):
        self.vocab = vocab
        self.max_token = len(vocab)
    
    '''
    以下5个是从file到function_ast，也就是把一个文件中代码解析成AST并拆分出各个函数
    '''
    def file_check(self, filename):
        return filename[-2:].lower()=='.c'
    
    def file_to_code(self, filename):
        import os
        #assert os.path.isfile(filename)
        assert self.file_check(filename)
        try:
            with open(filename, 'r', encoding="utf-8") as f:
                code = f.read()
            return code
        except:
            return None

    def file_to_ast(self, file):
        #ast = self.code_to_ast(self.file_to_code(file))
        from pycparser import parse_file
        ast = parse_file(file, use_cpp=False,
            cpp_path='c'),
            #cpp_args=r'-Iutils/fake_libc_include')
        return ast
    
    def code_to_ast(self, code):
        from pycparser import c_parser
        import re
        '''
        comment_inline_p1 = '#'
        comment_inline_p2 = '//'
        comment_open_tag_p = '/*'
        comment_close_tag_p = '*/'

        comment_inline_p1 = re.escape(comment_inline_p1)
        comment_inline_p2 = re.escape(comment_inline_p2)
        comment_inline_pattern = comment_inline_p1 + '.*?$' + '|' + comment_inline_p2 + '.*?$'
        comment_open_tag = re.escape(comment_open_tag_p)
        comment_close_tag = re.escape(comment_close_tag_p)
        comment_open_close_pattern = comment_open_tag + '.*?' + comment_close_tag
        '''
        code = re.sub('/\\*.*?\\*/', '', code, flags=re.DOTALL)
        code = re.sub('\\#.*?$|//.*?$', '', code, flags=re.MULTILINE)
        parser = c_parser.CParser()
        ast = parser.parse(code)
        return ast
    
        '''
        try:
            from pycparser import c_parser
            parser = c_parser.CParser()
            ast = parser.parse(code)
            return ast
        except:
            return None
        '''


    def get_functions(self, ast):
        # 只要 function
        return [func_ast for func_ast in ast.ext if func_ast.__class__.__name__ == 'FuncDef']

    def get_function_name(self, function_ast):
        return function_ast.decl.name
    
    def extract_functions(self, code):
        '''
        输入代码，输出两个list：一个是函数的ast，一个是函数起止行
        '''
        import clang
        import clang.cindex
        from clang.cindex import CursorKind
        
        
        function_pos = []
        function_nodes = []
        try:
            index = clang.cindex.Index.create()
            tu = index.parse(path='0.cpp', unsaved_files=[('0.cpp',code)])
        except Exception as e:
            pass
            return None, None

        AST_root_node= tu.cursor
        file_string_split = code.split('\n')
        linecount = code.count("\n")
        if not code.endswith("\n"):
            linecount += 1
        ast_list = list(AST_root_node.get_children())

        for idx, cur in enumerate(ast_list):
            if cur.kind == CursorKind.FUNCTION_DECL:
                start_lineno = cur.location.line
                if idx == len(ast_list) - 1:
                    end_lineno = linecount
                else:
                    end_lineno = ast_list[idx+1].location.line - 1
                #function_nodes.append(cur)
                function_pos.append((start_lineno, end_lineno))
                
            elif cur.kind == CursorKind.CLASS_DECL:
                ast_list_in_class = list(cur.get_children())
                for idx_in_class, cur_in_class in enumerate(ast_list_in_class):
                    if cur_in_class.kind == CursorKind.CXX_METHOD:
                        start_lineno = cur_in_class.location.line
                        if idx_in_class == len(ast_list_in_class) - 1: 
                            if idx == len(ast_list) - 1:
                                end_lineno = linecount
                            else:
                                end_lineno = ast_list[idx+1].location.line - 1
                            for lineno in range(end_lineno-1, 0, -1):
                                if file_string_split[lineno] and file_string_split[lineno][0]=='}':
                                    end_lineno = lineno
                                    break
                        else:
                            end_lineno = ast_list_in_class[idx_in_class+1].location.line - 1
                        #function_nodes.append(cur_in_class)
                        function_pos.append((start_lineno, end_lineno))
                        
        function_nodes = self.get_functions(self.code_to_ast(code))
        assert len(function_nodes) == len(function_pos)
        return function_nodes, function_pos

    
    '''
    以下函数，都是把ast变成ASTNN的输入结构，但是前两个使用token index，后两个使用token本身
    主要用前两个
    后两个是为了打印出来方便调试
    '''
    def ast_to_block(self, ast):
        blocks = []
        self.get_blocks(ast, blocks)
        tree = []
        for node in blocks:  # 这里的每个node就是对应一行代码？
            btree = self.replaced_by_index(node)
            tree.append(btree)
        return tree
    
    def replaced_by_index(self, node):
        # 返回的形式：[node, children1, children2, ...]
        # 一个大list，每个children又是一个子list
        token = node.token
        result = [self.vocab[token].index if token in self.vocab else self.max_token]
        children = node.children
        for child in children:
            result.append(self.replaced_by_index(child))
        return result
    
    def ast_to_token_block(self, ast):
        blocks = []
        self.get_blocks(ast, blocks)
        tree = []
        for node in blocks:
            btree = self.replaced_by_token(node)
            tree.append(btree)
        return tree
    
    def replaced_by_token(self, node):
        result = [node.token if node.token in self.vocab else 'UNKNOWN']
        children = node.children
        for child in children:
            result.append(self.replaced_by_token(child))
        return result
    
    
    '''
    
    '''
    def get_blocks(self, node, block_seq):
        children = node.children()
        name = node.__class__.__name__
        if name in ['FuncDef', 'If', 'For', 'While', 'DoWhile']:
            block_seq.append(Node_c(node))
            if name != 'For':
                skip = 1
            else:
                skip = len(children) - 1

            for i in range(skip, len(children)):
                child = children[i][1]
                if child.__class__.__name__ not in ['FuncDef', 'If', 'For', 'While', 'DoWhile', 'Compound']:
                    block_seq.append(Node_c(child))
                self.get_blocks(child, block_seq)
        elif name == 'Compound':
            block_seq.append(Node_c(name))
            for _, child in node.children():
                if child.__class__.__name__ not in ['If', 'For', 'While', 'DoWhile']:
                    block_seq.append(Node_c(child))
                self.get_blocks(child, block_seq)
            block_seq.append(Node_c('End'))
        else:
            for _, child in node.children():
                self.get_blocks(child, block_seq)

    
    '''
    以下两个函数对ast进行先序遍历获得先序遍历的token序列，用于 word embedding 的训练
    '''
    def get_sequence(self, node, sequence):
        current = Node_c(node, False)
        sequence.append(current.get_token())
        for _, child in node.children():
            self.get_sequence(child, sequence)
        if current.get_token().lower() == 'compound':
            sequence.append('End')
            # compound 代码段后面要加 End

    def trans_to_sequences(self, ast):
        # 这个用于生成token列表，用于 word embedding 的训练
        sequence = []
        self.get_sequence(ast, sequence)  # 从根节点开始先序遍历
        return sequence
    
    
    '''
    以下函数待定，只是为了打印出来看一下ast或者block之类的，方便调试
    '''
    def visit_block(self, block):
        pass
        
    def visit_token_block(self, block):
        block.show()
    
    def block_to_embedded(self):
        pass
    
    
    
######################################################################################
# For java
Logic1 = ['IfStatement', 'ForStatement', 'WhileStatement', 'DoStatement', 'SwitchStatement']
Logic2 = ['MethodDeclaration', 'ConstructorDeclaration']

class Node_java(object):
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(self.node, str)  # str => 叶子节点 => 无孩子节点
        self.token = self.get_token(node)
        self.children = self.add_children()

    def is_leaf(self):
        if self.is_str:
            return True
        return len(self.node.children) == 0

    def get_token(self, node):
        from javalang.ast import Node
        if isinstance(node, str):
            token = node
        elif isinstance(node, set): # 为什么？为什么是set的时候就是Modifier?
            token = 'Modifier'  # 访问修饰符，比如 public，private，static
        elif isinstance(node, Node):
            token = node.__class__.__name__
        else:
            token = ''
        return token

    def ori_children(self, root):
        from javalang.ast import Node
        if isinstance(root, Node):
            if self.token in Logic2:  # 这两个比较特殊？
                children = root.children[:-1]  # 最后一个丢弃？为什么？最后一个是什么？
            else:
                children = root.children
        elif isinstance(root, set):
            children = list(root)
        else:
            children = []

        return list(expand(children))

    def add_children(self):
        if self.is_str:  # str => 叶子节点 => 无孩子节点
            return []
        children = self.ori_children(self.node)
        
        # 下面是嵌套转化，把所有 javalang.ast.Node 全部转化成 Node_java
        if self.token in Logic1:
            return [Node_java(children[0])]
        elif self.token in Logic2:
            return [Node_java(child) for child in children]
        else:
            return [Node_java(child) for child in children if self.get_token(child) not in Logic1]  # What???
        
        
class Preprocessor_java():
    def __init__(self, vocab):
        self.vocab = vocab
        self.max_token = len(vocab)
    
    '''
    以下5个是从file到function_ast，也就是把一个文件中代码解析成AST并拆分出各个函数
    '''
    def file_check(self, filename):
        return filename[-5:].lower()=='.java'
    
    
    def file_to_code(self, filename):
        import os
        #assert os.path.isfile(filename)
        assert self.file_check(filename)
        try:
            with open(filename, 'r', encoding="utf-8") as f:
                code = f.read()
            return code
        except:
            return None

    def file_to_ast(self, file):
        ast = self.code_to_ast(self.file_to_code(file))
        return ast
    
    def code_to_ast(self, code):
        import javalang
        try:
            ast = javalang.parse.parse(code)
            #seq = trans2tokenseq(ast)
            return ast
        except:
            try:
                tokens = javalang.tokenizer.tokenize(code)
                parser = javalang.parser.Parser(tokens)
                ast = parser.parse_member_declaration()
                return ast
            except:
                return None
            
    def get_functions(self, ast):
        # 提取所有的function
        import javalang
        fun_list = list(ast.filter(javalang.tree.ConstructorDeclaration))
        fun_list.extend(ast.filter(javalang.tree.MethodDeclaration))
        return [f[1] for f in fun_list]
        # f[0] 好像是 path, f[1] 才是 node

    def get_function_name(self, function_ast):
        return function_ast.name
    
    def extract_functions(self, code):
        '''
        输入代码，输出两个list：一个是函数的ast，一个是函数起止行
        '''
    
        import re
        import javalang
        import itertools

        re_string = re.escape("\"") + '.*?' + re.escape("\"")

        comment_inline_p = '//'
        comment_inline = re.escape(comment_inline_p)
        comment_inline_pattern = comment_inline + '.*?$'

        function_nodes = []
        function_pos = []

        tree = None

        try:
            tree = javalang.parse.parse(code)
        except Exception as e:
            return None, None
            #logging.warning("File " + file_path + " cannot be parsed. (1)" + str(e))

        file_string_split = code.split('\n')
        nodes = itertools.chain(tree.filter(
            javalang.tree.ConstructorDeclaration), tree.filter(javalang.tree.MethodDeclaration))

        for path, node in nodes:
            (start_lineno, b) = node.position
            
            ##################################
            end_lineno = start_lineno
            closed = 0
            openned = 0

            for line in file_string_split[start_lineno-1:]:
                if len(line.strip()) == 0:
                    continue
                
                line_re = re.sub(re_string, '', line, flags=re.DOTALL)
                # 先删字符串再删注释
                line_re = re.sub(comment_inline_pattern, '', line_re, flags=re.MULTILINE)

                closed += line_re.count('}')
                openned += line_re.count('{')

                if closed == openned:
                    break
                else:
                    end_lineno += 1
            ###################################

            function_pos.append((start_lineno, end_lineno))
            function_nodes.append(node)


        return function_nodes, function_pos
    
    
    '''
    以下两个函数分别获取节点token和孩子节点，是后续其他操作的基础
    '''
    def get_token(self, node):
        from javalang.ast import Node
        if isinstance(node, str):
            token = node
        elif isinstance(node, set):  # 为什么？为什么是set的时候就是Modifier?
            token = 'Modifier'#node.pop()
        elif isinstance(node, Node):
            token = node.__class__.__name__  # 直接用类名
        else:
            token = ''

        return token
    
    def get_children(self, root):
        from javalang.ast import Node
        if isinstance(root, Node):
            children = root.children
        elif isinstance(root, set):  # 按照get_token，这里应该是Modifier，就把Modifier的子节点直接转成list
            children = list(root)
        else:
            children = []

        return list(expand(children))
    

    
    '''
    以下函数，都是把ast变成ASTNN的输入结构，但是前两个使用token index，后两个使用token本身
    主要用前两个
    后两个是为了打印出来方便调试
    '''
    def ast_to_block(self, ast):
        blocks = []
        self.get_blocks(ast, blocks)
        tree = []
        for node in blocks:  # 这里的每个node就是对应一行代码？
            btree = self.replaced_by_index(node)
            tree.append(btree)
        return tree
    
    def replaced_by_index(self, node):
        # 返回的形式：[node, children1, children2, ...]
        # 一个大list，每个children又是一个子list
        token = node.token
        result = [self.vocab[token].index if token in self.vocab else self.max_token]
        children = node.children
        for child in children:
            result.append(self.replaced_by_index(child))
        return result
    
    def ast_to_token_block(self, ast):
        blocks = []
        self.get_blocks(ast, blocks)
        tree = []
        for node in blocks:
            btree = self.replaced_by_token(node)
            tree.append(btree)
        return tree
    
    def replaced_by_token(self, node):
        result = [node.token if node.token in self.vocab else 'UNKNOWN']
        children = node.children
        for child in children:
            result.append(self.replaced_by_token(child))
        return result
    
    
    '''
    最复杂的东东，根据当前的 node 获得一个或多个 Node_java 并添加到 block_seq 里面
    block_seq.append 的必定是一个 Node_java，可以猜测 Node_java 是一个把 javalang.ast.Node 转化成自定义的节点类
    '''
    def get_blocks(self, node, block_seq):
        name, children = self.get_token(node), self.get_children(node)
        
        # 分4种情况，前3种又进一步对孩子进行细分
        if name in Logic2:
            block_seq.append(Node_java(node))
            body = node.body
            for child in body:
                if self.get_token(child) not in Logic1 and not hasattr(child, 'block'):
                    block_seq.append(Node_java(child))
                else:
                    self.get_blocks(child, block_seq)
                    
        elif name in Logic1:
            block_seq.append(Node_java(node))
            for child in children[1:]:
                token = self.get_token(child)
                if not hasattr(node, 'block') and token not in Logic1 + ['BlockStatement']:
                    block_seq.append(Node_java(child))
                else:
                    self.get_blocks(child, block_seq)
                block_seq.append(Node_java('End'))
                
        elif name == 'BlockStatement' or hasattr(node, 'block'):
            block_seq.append(Node_java(name))
            for child in children:
                if self.get_token(child) not in Logic1:
                    block_seq.append(Node_java(child))
                else:
                    self.get_blocks(child, block_seq)

        else:
            for child in children:
                self.get_blocks(child, block_seq)

    
    '''
    以下两个函数对ast进行先序遍历获得先序遍历的token序列，用于 word embedding 的训练
    '''
    def get_sequence(self, node, sequence):  # 获取先序遍历结果，同时为一些特殊代码块加上'End'
        token, children = self.get_token(node), self.get_children(node)
        sequence.append(token)

        for child in children:
            self.get_sequence(child, sequence)

        if token in Logic1:
            sequence.append('End')  
            # 因为 Logic1 之后会紧接着一个 'BlockStatement' (表示左大括号)
            # 所以在后面要加上一个 'End' (表示右大括号)

    def trans_to_sequences(self, ast):
        # 这个用于生成token列表，用于 word embedding 的训练
        sequence = []
        self.get_sequence(ast, sequence)  # 从根节点开始先序遍历
        return sequence
    
    
    '''
    以下函数待定，只是为了打印出来看一下ast或者block之类的，方便调试
    '''
    def visit_block(self, block):
        pass
        
    def visit_token_block(self, block):
        pass
    
    def block_to_embedded(self):
        pass