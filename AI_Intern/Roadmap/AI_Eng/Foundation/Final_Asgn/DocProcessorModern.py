from pathlib import Path
from dotenv import load_dotenv
from llama_cloud import LlamaCloud
from typing import List
from html import unescape
from dataclasses import dataclass, field
from os.path import basename

import re

@dataclass
class Heading:
    text: str
    type:str
    number: str

@dataclass
class Section:
    title: str
    heading_type: str
    level: int
    owner: str | None = None
    content: list[str] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)

class DocProcessorModern:
    '''Modern class to process documents using API. This helps
       reduce stress on local machine and can handle OCR operations.
    '''
    def __init__(self):
        llama_api_keys_path = Path(".secrets/LLAMA_API_Keys.secrets")
        load_dotenv(llama_api_keys_path)

        self.client = LlamaCloud()

    def loadDoc(self, filePath: str):
        '''Load a SINGLE document then return the result'''
        file = self.client.files.create(
            file=Path(filePath),
            purpose="parse"
        )

        job = self.client.parsing.create(
            tier="agentic",
            version="latest",
            file_id=file.id
        )

        job = self.client.parsing.wait_for_completion(job.id)

        result = self.client.parsing.get(
            job_id=job.id,
            expand=["markdown"]
        )

        return result

    def chunkDocHardCoded(self, pages: List[MarkdownPage], fileName: str) -> dict[str: str]:
        '''
        Hard coded method to chunk documents used in this assignment
        '''
        roots: list[Section] = []
        stack: list[Section] = [] # Keep track of the current level

        for page in pages:
            if page.page_number == 1: # Skip the first page
                continue

            page_content = self.pre_filter_hardCoded(page.markdown)

            for line in page_content.splitlines():
                line = line.strip()

                if not line:
                    continue

                heading = self.parse_heading(line)

                # Append content if not a Heading
                if heading is None:
                    if stack:
                        stack[-1].content.append(line)
                    continue

                # Add a new heading
                level = self.get_heading_level(
                    heading=heading, 
                    current_stack=stack)

                # Remove deeper section
                while len(stack) > level:
                    stack.pop()

                section = Section(
                    title=heading.text,
                    heading_type=heading.type,
                    level=level,
                    owner=(
                        fileName
                        if level == 0
                        else stack[-1].title if stack else fileName
                    ),
                    content=[heading.text]
                )

                if level == 0:
                    roots.append(section)
                elif stack:
                    stack[-1].children.append(section)
                else:
                    roots.append(section)

                stack.append(section)

        return roots

    def pre_filter_hardCoded(self, page_content: str) -> str:
        '''
        Hard coded filter for the contents.
        '''
        unwanted = [
            "FPT Software logo",
            "A person in a suit holding a glowing digital "
            "interface representing human resources and connectivity",
            "icon",
            "End graphic",
            "End",
            "A person using a tablet in front of stacked shipping containers with "
            "digital supply chain icons overlaid",
            "FPT SOFTWARE COMPANY LIMITED",
            "A person using a tablet in front of stacked shipping containers with digital supply chain s overlaid",
        ]

        for item in unwanted:
            page_content = page_content.replace(item, "")

         # Remove Markdown formatting
        page_content = re.sub(r"^#{1,6}\s*", "", page_content, flags=re.MULTILINE)
        page_content = re.sub(r"\*\*(.*?)\*\*", r"\1", page_content)
        page_content = re.sub(r"(?<!\*)\*(?!\*)(.*?)\*(?!\*)", r"\1", page_content)

        return page_content.strip()

    def convert_html_table(self, html_table: str) -> list[str]:
        '''
        Convert HTML code for a table into a flat list. First half will be header and second half will be corresponded values.
        e.g., ['No.', 'Effective Date', 'Version', 'Change Description', 'Reason for Changes', 
        'Reviewer', 'Approver', '1', '', '1.0', 'Newly Create', '', '', ''] 
        '''
        cells = re.findall(
            r"<(?:td|th)[^>]*>(.*?)</(?:td|th)>",
            html_table,
            flags=re.DOTALL | re.IGNORECASE,
        )

        return [
            unescape(
                re.sub(r"<[^>]+>", "", cell)
            ).strip()
            for cell in cells
        ]

    def parse_heading(self, line: str) -> Heading | None:
        '''
        Check if this line is a heading and return the heading type
        '''
        line = line.strip()

        if not line:
            return None

        # Numeric
        #
        # 1 Policy
        # 1.1 Policy
        # 1.1.1 Policy
        #
        match = re.match(
            r"^(\d+(?:\.\d+)*\.?)\s+(.+)$",
            line,
        )

        if match:
            number = match.group(1).rstrip(".")

            return Heading(
                text=line,
                type="numeric",
                number=number
            )

        # Roman
        #
        # i. Policy
        # ii) Policy
        # (iii) Policy
        #
        match = re.match(
            r"^(?:\(([ivxlcdm]+)\)|([ivxlcdm]+)[.)])\s+(.+)$",
            line,
            re.IGNORECASE,
        )

        if match:
            number = (
                match.group(1)
                or match.group(2)
            ).lower()

            return Heading(
                text=line,
                type="roman",
                number=number,
            )
        # Letter
        #
        # a. Policy
        # a) Policy
        # (a) Policy
        #
        match = re.match(
            r"^(?:\(([a-z])\)|([a-z])[.)])\s+(.+)$",
            line,
            re.IGNORECASE,
        )

        if match:
            number = (
                match.group(1)
                or match.group(2)
            ).lower()

            return Heading(
                text=line,
                type="letter",
                number=number,
            )

        # Article
        match = re.match(
            r"^Article\s+(\d+)(?:\s*[-.:]\s*(.*))?$",
            line,
            re.IGNORECASE,
        )

        if match:
            number = match.group(1)

            return Heading(
                text=line,
                type="article",
                number=number,
            )

        return None

    def get_heading_level(self, heading: Heading, current_stack: list[Section]) -> int:
        '''
        Return the current heading level, with 0 being the highest
        '''

        if heading.type == "article":
            return 0

        if heading.type == "numeric":
            return heading.number.count(".")

        if heading.type == "letter" or heading.type == "roman":
            return len(current_stack)
        