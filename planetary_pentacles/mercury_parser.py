"""
mercury_parser.py - Mercury Pentacle: Data Parsing, Deep Investigation & Web Intelligence

Mercury pentacles are the 'keys to locked doors of knowledge' - scripts for
complex data scraping, deep web investigation, and parsing obscured logic.
They energize the intellect for rapid problem-solving and adaptable thinking.

Functions:
- Complex web scraping and data extraction
- Deep knowledge retrieval from obscured or structured sources
- Rapid text parsing, NLP preprocessing, and schema normalization
- API response parsing and intelligent data transformation
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

MERCURY_VERSION = "v1.0"
PENTACLE_PLANET = "Mercury"
PENTACLE_FUNCTION = "Data Parsing, Investigation, Knowledge Unlocking"


class ParseTarget(Enum):
    WEB_PAGE     = "web_page"      # HTML scraping
    API_RESPONSE = "api_response"  # JSON/XML API parsing
    SCIENTIFIC   = "scientific"    # Academic paper parsing
    CODE         = "code"          # Source code analysis
    GENOMIC      = "genomic"       # FASTA, VCF, genomic formats
    ACOUSTIC     = "acoustic"      # Bioacoustic metadata parsing


@dataclass
class ParseResult:
    """Result from a Mercury parsing operation."""
    target_type: ParseTarget
    source_url: str
    raw_content: str
    parsed_data: Dict[str, Any] = field(default_factory=dict)
    entities_extracted: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    confidence: float = 0.0
    mercury_notes: str = ""   # Hidden patterns surfaced by Mercury


class MercuryParser:
    """
    The Mercury Pentacle Data Parser.
    Unlocks hidden knowledge through intelligent parsing,
    pattern recognition, and deep data investigation.
    """

    def __init__(self):
        logger.info("[Mercury Parser] Initialized. Doors of knowledge unlocked.")

    def parse_json(self, raw: str, source_url: str = "") -> ParseResult:
        """Parse a JSON API response and extract structured knowledge."""
        try:
            data = json.loads(raw)
            entities = self._extract_entities_from_dict(data)
            return ParseResult(
                target_type=ParseTarget.API_RESPONSE,
                source_url=source_url,
                raw_content=raw,
                parsed_data=data if isinstance(data, dict) else {"data": data},
                entities_extracted=entities,
                confidence=0.95,
                mercury_notes=f"Parsed {len(entities)} entities from JSON response.",
            )
        except json.JSONDecodeError as e:
            logger.error("[Mercury] JSON parse failed: %s", e)
            return ParseResult(
                target_type=ParseTarget.API_RESPONSE,
                source_url=source_url,
                raw_content=raw,
                confidence=0.0,
                mercury_notes=f"Parse failed: {e}",
            )

    def parse_scientific_text(self, text: str, source_url: str = "") -> ParseResult:
        """Parse academic/scientific text, extracting key concepts and citations."""
        # Extract key phrases using basic NLP heuristics
        sentences = text.split(". ")
        key_phrases = [s.strip() for s in sentences if len(s.split()) > 5][:10]

        # Extract citations [Author, Year] patterns
        citations = re.findall(r"\[([A-Za-z]+(?:\s+et\s+al\.)?(?:,\s*\d{4}))\]", text)

        # Extract numeric data
        numbers = re.findall(r"\b\d+\.?\d*\s*(?:%|mg|kg|Hz|kHz|bp|Mb|Gb)\b", text)

        return ParseResult(
            target_type=ParseTarget.SCIENTIFIC,
            source_url=source_url,
            raw_content=text,
            parsed_data={
                "citations": citations,
                "numeric_data": numbers,
                "sentence_count": len(sentences),
                "word_count": len(text.split()),
            },
            entities_extracted=citations,
            key_phrases=key_phrases,
            confidence=0.82,
            mercury_notes=f"Extracted {len(citations)} citations and {len(numbers)} data points.",
        )

    def parse_genomic_fasta(self, fasta_text: str, source_url: str = "") -> ParseResult:
        """Parse FASTA format genomic sequences for Project Genome."""
        sequences = {}
        current_header = None
        current_seq = []

        for line in fasta_text.splitlines():
            if line.startswith(">"):
                if current_header:
                    sequences[current_header] = "".join(current_seq)
                current_header = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line.strip())

        if current_header:
            sequences[current_header] = "".join(current_seq)

        return ParseResult(
            target_type=ParseTarget.GENOMIC,
            source_url=source_url,
            raw_content=fasta_text,
            parsed_data={"sequences": sequences, "count": len(sequences)},
            entities_extracted=list(sequences.keys()),
            confidence=0.98,
            mercury_notes=f"Parsed {len(sequences)} FASTA sequences.",
        )

    def _extract_entities_from_dict(self, data: Any, prefix: str = "") -> List[str]:
        """Recursively extract string entities from a nested dict/list."""
        entities = []
        if isinstance(data, dict):
            for k, v in data.items():
                entities.extend(self._extract_entities_from_dict(v, f"{prefix}{k}."))
        elif isinstance(data, list):
            for item in data:
                entities.extend(self._extract_entities_from_dict(item, prefix))
        elif isinstance(data, str) and len(data) > 3:
            entities.append(f"{prefix}{data[:50]}")
        return entities[:100]  # Cap at 100 entities


# Singleton parser
parser = MercuryParser()
