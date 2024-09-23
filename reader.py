import re

from pypdf import PdfReader

import config


class Reader:
    @staticmethod
    def remove_header(text: str) -> str:
        header_match = re.search(pattern=r"\n", string=text)
        header_idx = header_match.span()[1] if header_match else 0
        if re.match(pattern=r"^\d+$", string=text[:header_idx]):
            header_match = re.search(pattern=r"\n", string=text[header_idx:])
            header_idx += header_match.span()[1] if header_match else 0
        return text[header_idx:]

    @staticmethod
    def remove_footer(text: str) -> str:
        footer_match = re.search(pattern=r"\n", string=text[::-1])
        footer_idx = footer_match.span()[1] if footer_match else -len(text)
        return text[:- footer_idx]

    @staticmethod
    def remove_links(text: str) -> str:
        return re.sub(pattern=r"<https?://.*[\r\n]*.*>", repl="", string=text)

    @staticmethod
    def fix_paragraphs(text: str) -> str:
        return (text
                .replace(".\n", ".\n\n")
                .replace("?\n", "?\n\n")
                .replace("!\n", "!\n\n"))

    @staticmethod
    def fix_whitespaces(text: str) -> str:
        return (text
                .replace(" \n", " ")
                .replace("\xa0", " "))

    @staticmethod
    def fix_hyphen_usage(text: str) -> str:
        return (text
                .replace("–", "-")
                .replace(" -\n", "")
                .replace("-\n", "-"))

    @staticmethod
    def fix_slash_usage(text: str) -> str:
        return text.replace("/\n", "/")

    @staticmethod
    def remove_references_all(text_all: str) -> str:
        return re.sub(pattern=r"References\n.*\d+.*(\n\n.*\d+.*)+", repl="", string=text_all)

    @classmethod
    def read_pdf(cls, path: str) -> str:
        result = ""
        with open(path, mode="rb") as f:
            reader = PdfReader(f)

            for i, page in enumerate(reader.pages):
                if i in config.PAGES_TO_SKIP:
                    continue

                raw_text = page.extract_text(extraction_mode="plain")

                raw_text = cls.remove_header(text=raw_text)
                raw_text = cls.remove_footer(text=raw_text)
                raw_text = cls.remove_links(text=raw_text)
                raw_text = cls.fix_whitespaces(text=raw_text)
                raw_text = cls.fix_paragraphs(text=raw_text)
                raw_text = cls.fix_hyphen_usage(text=raw_text)
                raw_text = cls.fix_slash_usage(text=raw_text)

                result += raw_text

        result = cls.remove_references_all(text_all=result)
        return result
