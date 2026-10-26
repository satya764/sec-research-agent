from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Section = Literal["business", "risk_factors", "mdna"]
QuestionType = Literal["lookup", "comparison", "trend", "calculation", "narrative", "other"]


class Company(BaseModel):
    cik: str  # zero-padded to 10 digits
    ticker: str
    name: str

    @property
    def cik_int(self) -> int:
        return int(self.cik)


class FilingRef(BaseModel):
    cik: str
    ticker: str
    company: str
    form: str
    accession: str
    filing_date: date
    report_date: date | None = None
    primary_document: str

    @property
    def fiscal_year(self) -> int:
        return (self.report_date or self.filing_date).year

    @property
    def folder_url(self) -> str:
        return (
            f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/"
            f"{self.accession.replace('-', '')}"
        )

    @property
    def url(self) -> str:
        return f"{self.folder_url}/{self.primary_document}"


class FactPoint(BaseModel):
    metric: str
    concept: str
    unit: str
    value: float
    start: date | None = None
    end: date
    fiscal_year: int
    period: Literal["annual", "quarterly"]
    form: str
    filed: date
    accession: str


class Chunk(BaseModel):
    ticker: str
    cik: str
    company: str
    form: str
    filing_date: date
    fiscal_year: int
    accession: str
    section: Section
    chunk_index: int
    url: str
    content: str

    def embedding_text(self) -> str:
        label = SECTION_LABELS[self.section]
        return f"{self.company} {self.form} fiscal {self.fiscal_year}, {label}\n{self.content}"


SECTION_LABELS: dict[str, str] = {
    "business": "Business",
    "risk_factors": "Risk Factors",
    "mdna": "Management's Discussion and Analysis",
}


class SearchHit(BaseModel):
    chunk_id: int
    score: float
    chunk: Chunk

    @property
    def source_id(self) -> str:
        return f"C{self.chunk_id}"


class Filters(BaseModel):
    ticker: str | None = None
    fiscal_year: int | None = None
    form: str | None = None
    section: Section | None = None


class ParsedQuestion(BaseModel):
    companies: list[str] = Field(
        default_factory=list, description="Ticker symbols if known, otherwise company names"
    )
    metric: str | None = Field(None, description="One of the known metric keys, or null")
    fiscal_years: list[int] = Field(default_factory=list)
    question_type: QuestionType = "other"


class Citation(BaseModel):
    source_id: str
    kind: Literal["filing", "fact"]
    company: str
    form: str
    filing_date: date
    section: str | None = None
    url: str
    snippet: str


class AskResult(BaseModel):
    question: str
    answer: str
    found: bool
    value: float | None = None
    unit: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    parsed: ParsedQuestion | None = None
    model: str = ""
    route_reason: str = ""
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    blocked: bool = False
    redacted: list[str] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)
