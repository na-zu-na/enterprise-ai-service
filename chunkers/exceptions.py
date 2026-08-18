class DocumentChunkException(Exception):
    """
    Chunk 模块基础异常。
    """
    pass


class InvalidChunkConfigException(DocumentChunkException):
    """
    Chunk 配置异常。
    """
    pass