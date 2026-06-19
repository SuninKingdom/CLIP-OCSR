import json
import re
import time
import logging

from openai import OpenAI

from config import Config

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert chemistry assistant specializing in patent Markush structures. \
Given OCR-extracted text containing definitions of chemical substituents from a Markush structure, \
extract all substituent variable definitions as structured JSON.

Rules:
1. Identify each variable label (e.g., R1, R2, B1, X, L, m, n)
2. For each variable, list ALL possible substituent values mentioned
3. Return a JSON object where keys are variable labels and values are arrays of substituent strings
4. Preserve the exact chemical terminology from the text
5. For integer ranges like "m represents 0, 1 or 2", return ["0", "1", "2"]
6. For ranges like "n = 1-3", return ["1", "2", "3"]
7. If a variable is described as "as described hereinabove", include that phrase as-is
8. Return ONLY the JSON object, no explanation"""

USER_PROMPT_TEMPLATE = """Extract the substituent definitions from this Markush structure text.

Text:
{text}

Return a JSON object mapping variable labels to their possible values.
Example output format:
{{"R1": ["hydrogen", "methyl"], "R2": ["halogen", "C1-C6 alkyl"], "m": ["0", "1", "2"]}}"""


class SubstituentExtractor:
    """Uses a text LLM to extract substituent definitions from OCR text."""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        )

    def extract_substituents(self, ocr_text: str) -> dict | None:
        """Send OCR text to LLM, return substituent dict."""
        user_prompt = USER_PROMPT_TEMPLATE.format(text=ocr_text)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(self.config.llm_max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=messages,
                    temperature=self.config.llm_temperature,
                    max_tokens=2048,
                )
                text = response.choices[0].message.content
                result = self._parse_response(text)
                if result is not None:
                    return result
                logger.warning(f"Failed to parse LLM response: {text[:200]}")
            except Exception as e:
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
                if attempt < self.config.llm_max_retries - 1:
                    time.sleep(2 ** attempt)

        return None

    def _parse_response(self, text: str) -> dict | None:
        """Parse LLM response into a substituent dict."""
        if not text:
            return None

        # Try markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try direct JSON parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try finding first { ... } block
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None
