from abc import ABC, abstractmethod
from pathlib import Path

from parsers.exceptions import DocumentNotFoundException, DocumentReadException
from parsers.models import DocumentParseContext, ParsedDocument


class BaseDocumentParser(ABC):
    @abstractmethod
    def parse(self,
              context:DocumentParseContext
              )->ParsedDocument:
        """
        将原始文件转换为统一 ParsedDocument。
        """
        pass

    def validate_path(self,
                      context:DocumentParseContext
                      )->Path:

        path = Path(context.storage_path)

        if not path.exists():
            raise DocumentNotFoundException(
                f"文档文件不存在: {context.storage_path}"
            )

        if not path.is_file():
            raise DocumentReadException(
                f"文档路径不是有效文件: {context.storage_path}"
            )

        return path