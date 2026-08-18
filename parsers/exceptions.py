class DocumentParseException(Exception):
    """
    文档解析基础异常
    """
    pass


class UnsupportedFileTypeException(DocumentParseException):
    """
    不支持的文件类型
    """
    pass


class DocumentNotFoundException(DocumentParseException):
    """
    文件不存在
    """
    pass


class DocumentReadException(DocumentParseException):
    """
    文件读取失败
    """
    pass