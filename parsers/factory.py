from parsers.base import BaseDocumentParser
from parsers.exceptions import UnsupportedFileTypeException
from parsers.markdown_parser import MarkdownParser
from parsers.txt_parser import TxtParser


class ParserFactory:
    _parsers: dict[str, type[BaseDocumentParser]] = {
        "txt": TxtParser,
        "text": TxtParser,
        "md": MarkdownParser,
        "markdown": MarkdownParser,
    }

    @classmethod
    def get_parser(cls, file_type: str) -> BaseDocumentParser:
        normalized_type = file_type.strip().lower().replace(".", "")
        parser_class = cls._parsers.get(normalized_type)

        if parser_class is None:
            raise UnsupportedFileTypeException(
                f"暂不支持的文件类型: {file_type}"
            )

        return parser_class()
