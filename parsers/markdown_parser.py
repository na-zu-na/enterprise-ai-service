from pathlib import Path
import re

from parsers.base import BaseDocumentParser
from parsers.exceptions import DocumentReadException
from parsers.models import DocumentParseContext, ParsedDocument, ParsedSection


class MarkdownParser(BaseDocumentParser):
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def parse(self,
              context:DocumentParseContext
              )->ParsedDocument:
        path=self.validate_path(context)

        try:
            content=path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentReadException(
                f"Markdown 文件编码不是 UTF-8: "
                f"{context.original_name}"
            ) from exc
        except OSError as exc:
            raise DocumentReadException(
                f"读取 Markdown 文件失败: "
                f"{context.original_name}"
            ) from exc

        content=self._clen_text(content)
        sections=self._parse_sections(content)
        title=self._get_title(context,sections)

        return ParsedDocument(
            document_id=context.document_id,
            knowledge_base_id=context.knowledge_base_id,

            name=context.name,
            original_name=context.original_name,

            file_type=context.file_type,
            file_size=context.file_size,
            storage_path=context.storage_path,
            version=context.version,

            title=title,
            content=content,

            sections=sections,

            metadata={
                         "char_count": len(content),
                         "section_count": len(sections),
                         "parser": "MarkdownParser",
                     }
        )

    @staticmethod
    def _clen_text(text:str):
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        return text.strip()

    def _parse_sections(self,content:str)->list[ParsedSection]:
        sections:list[ParsedSection]=[]

        current_title:str |None = None
        current_level:int |None = None
        current_content:list[str] = []

        for line in content.split("\n"):
            match=self.HEADING_PATTERN.match(line)
            if match:
                self._append_section(
                    sections=sections,
                    title=current_title,
                    level=current_level,
                    content_lines=current_content
                )

                current_title=match.group(2).strip()
                current_level=len(match.group(1))
                current_content=[]
            else:
                current_content.append(line)

        self._append_section(
            sections=sections,
            title=current_title,
            level=current_level,
            content_lines=current_content,
        )

        return sections

    @staticmethod
    def _append_section(sections: list[ParsedSection],
                        title: str | None,
                        level: int | None,
                        content_lines: list[str]):
        content="\n".join(content_lines).strip()

        #如果不存在标题和内容就不存
        if title is None and not content:
            return

        sections.append(
            ParsedSection(
                title=title,
                level=level,
                content=content,
            )
        )


    @staticmethod
    def _get_title(context, sections:list[ParsedSection])->str:
        #markdown第一层 标题优先
        for section in sections:
            if section.level==1 and section.title:
                return section.title

            if context.name:
                return context.name

            return Path(context.original_name).stem

