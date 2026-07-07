from dataclasses import dataclass, asdict
import spacy
from src.constants import (
    COMPARE_KEYWORDS,
    LIST_KEYWORDS,
    BOOLEAN_KEYWORDS,
    MULTIHOP_KEYWORDS,
    CONSTRAINT_KEYWORDS,
    REASONING_KEYWORDS
)
from src.utils import logger, validate_text

@dataclass
class QueryFeatures:
    token_count: int
    character_count: int
    average_word_length: float
    sentence_count: int
    clause_count: int
    entity_count: int
    person_count: int
    organization_count: int
    location_count: int
    date_count: int
    comparison: int
    boolean: int
    listing: int
    multihop: int
    reasoning: int
    constraint_count: int
    intent: str

class QueryAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def analyze(self, query: str) -> QueryFeatures:
        query = validate_text(query)
        doc = self.nlp(query)
        tokens = [token for token in doc if not token.is_space]
        token_count = len(tokens)
        character_count = len(query)
        words = [t.text for t in tokens if t.is_alpha]
        average_word_length = 0.0
        if words:
            average_word_length = round(sum(len(w) for w in words) / len(words), 2)
        sentence_count = max(1, len(list(doc.sents)))
        clause_count = sum(1 for token in doc if token.pos_ in ("VERB", "AUX"))
        entity_count = len(doc.ents)
        person_count = 0
        organization_count = 0
        location_count = 0
        date_count = 0
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                person_count += 1
            elif ent.label_ == "ORG":
                organization_count += 1
            elif ent.label_ in ("GPE", "LOC"):
                location_count += 1
            elif ent.label_ in ("DATE", "TIME"):
                date_count += 1
        token_set = {t.text.lower() for t in tokens}
        comparison_val = int(any(w in token_set for w in COMPARE_KEYWORDS))
        boolean_val = int(any(w in token_set for w in BOOLEAN_KEYWORDS))
        listing_val = int(any(w in token_set for w in LIST_KEYWORDS))
        multihop_val = int(any(w in token_set for w in MULTIHOP_KEYWORDS))
        reasoning_val = sum(1 for w in token_set if w in REASONING_KEYWORDS)
        constraint_val = sum(1 for w in token_set if w in CONSTRAINT_KEYWORDS)
        intent = "factoid"
        if comparison_val:
            intent = "comparison"
        elif reasoning_val > 0:
            intent = "explanation"
        elif listing_val:
            intent = "listing"
        elif boolean_val:
            intent = "boolean"
        return QueryFeatures(
            token_count=token_count,
            character_count=character_count,
            average_word_length=average_word_length,
            sentence_count=sentence_count,
            clause_count=clause_count,
            entity_count=entity_count,
            person_count=person_count,
            organization_count=organization_count,
            location_count=location_count,
            date_count=date_count,
            comparison=comparison_val,
            boolean=boolean_val,
            listing=listing_val,
            multihop=multihop_val,
            reasoning=reasoning_val,
            constraint_count=constraint_val,
            intent=intent
        )

    @staticmethod
    def to_dict(features: QueryFeatures) -> dict:
        return asdict(features)
