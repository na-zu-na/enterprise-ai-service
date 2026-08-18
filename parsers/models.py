from dataclasses import dataclass, field
from typing import Any

#文件解析出来的段落
@dataclass
class ParsedSection:
    title:str | None
    content:str
    level:int |None=None
    page_number:int |None=None
    metadata:dict[str, Any]=field(default_factory=dict)

#解析之后文件规范
@dataclass
class ParsedDocument:
    document_id: int
    knowledge_base_id: int
    name: str
    original_name: str
    file_type: str
    file_size: int
    storage_path: str
    version: int
    title: str | None
    content: str

    sections:list[ParsedSection]=field(default_factory=list)
    metadata:dict[str,Any]=field(default_factory=dict)

# 文件上传格式
@dataclass
class DocumentParseContext:
    document_id: int
    knowledge_base_id: int
    name: str
    original_name: str
    file_type: str
    file_size: int
    storage_path: str
    version: int
