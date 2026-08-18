from pathlib import Path
from pydoc import text

from parsers.base import BaseDocumentParser
from parsers.exceptions import DocumentNotFoundException, DocumentReadException
from parsers.models import DocumentParseContext, ParsedDocument


class TxtParser(BaseDocumentParser):
    def parse(self,
              context: DocumentParseContext
              )->ParsedDocument:
        path=self.validate_path(context)

        #读取文件
        try:
            content=path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentReadException(
                f"TXT 文件编码不是 UTF-8: {context.original_name}"
            ) from exc
        except OSError as exc:
            raise DocumentReadException(
                f"读取文档失败: {context.original_name}"
            ) from exc

        content=self._clean_text(content)

        return ParsedDocument(
            document_id=context.document_id,
            knowledge_base_id=context.knowledge_base_id,

            name=context.name,
            original_name=context.original_name,

            file_type=context.file_type,
            file_size=context.file_size,
            storage_path=context.storage_path,
            version=context.version,

            title=self._get_title(context),
            content=content,

            sections=[],

            metadata={
                "char_count": len(content),
                "parser": "TxtParser",
            }
        )


    @staticmethod
    def _clean_text(text:str)->str:
        #统一换行符
        text=text.replace("\r\n", "\n")
        text=text.replace("\r", "\n")

        #删除每行末尾多于空白
        lines=[
            line.strip()
            for line in text.split("\n")
        ]

        return "\n".join(lines).strip()

    @staticmethod
    def _get_title(context:DocumentParseContext)->str:
        if context.name:
            return context.name

        return Path(context.storage_path).stem