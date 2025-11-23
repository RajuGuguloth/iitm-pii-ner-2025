"""
Label definitions and utilities for NER task.
Implements BIO tagging scheme for token-level classification.
"""

from typing import List, Dict, Tuple

# Entity types
ENTITY_TYPES = [
    "CREDIT_CARD",
    "PHONE",
    "EMAIL",
    "PERSON_NAME",
    "DATE",
    "CITY",
    "LOCATION"
]

# PII entity types (require high precision)
PII_TYPES = {"CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE"}

# Non-PII entity types
NON_PII_TYPES = {"CITY", "LOCATION"}


def get_bio_labels() -> List[str]:
    """
    Generate BIO labels for all entity types.
    
    Returns:
        List of labels: ['O', 'B-CREDIT_CARD', 'I-CREDIT_CARD', ...]
    """
    labels = ['O']  # Outside any entity
    
    for entity_type in ENTITY_TYPES:
        labels.append(f'B-{entity_type}')  # Begin
        labels.append(f'I-{entity_type}')  # Inside
    
    return labels


# Generate label list
LABELS = get_bio_labels()

# Create label to ID mapping
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}

# Create ID to label mapping
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# Number of labels
NUM_LABELS = len(LABELS)


def align_labels_with_tokens(
    text: str,
    entities: List[Dict],
    token_offsets: List[Tuple[int, int]]
) -> List[str]:
    """
    Convert character-level entity spans to token-level BIO labels.
    
    Args:
        text: Original text string
        entities: List of entity dicts with 'start', 'end', 'label'
        token_offsets: List of (start, end) character offsets for each token
        
    Returns:
        List of BIO labels for each token
    """
    # Initialize all tokens as 'O' (Outside)
    labels = ['O'] * len(token_offsets)
    
    # Process each entity
    for entity in entities:
        entity_start = entity['start']
        entity_end = entity['end']
        entity_label = entity['label']
        
        # Find which tokens overlap with this entity
        entity_tokens = []
        for token_idx, (token_start, token_end) in enumerate(token_offsets):
            # Check if token overlaps with entity span
            if token_start < entity_end and token_end > entity_start:
                entity_tokens.append(token_idx)
        
        # Assign BIO labels to entity tokens
        if entity_tokens:
            # First token gets B- (Begin) tag
            labels[entity_tokens[0]] = f'B-{entity_label}'
            
            # Remaining tokens get I- (Inside) tags
            for token_idx in entity_tokens[1:]:
                labels[token_idx] = f'I-{entity_label}'
    
    return labels


def get_label_ids(labels: List[str]) -> List[int]:
    """
    Convert label strings to label IDs.
    
    Args:
        labels: List of BIO label strings
        
    Returns:
        List of label IDs
    """
    return [LABEL2ID[label] for label in labels]


def get_entity_type(bio_label: str) -> str:
    """
    Extract entity type from BIO label.
    
    Args:
        bio_label: BIO label like 'B-EMAIL' or 'I-PHONE'
        
    Returns:
        Entity type like 'EMAIL' or 'PHONE', or 'O' for outside
    """
    if bio_label == 'O':
        return 'O'
    return bio_label.split('-', 1)[1]


def is_pii_entity(entity_type: str) -> bool:
    """
    Check if entity type is PII.
    
    Args:
        entity_type: Entity type string
        
    Returns:
        True if PII entity, False otherwise
    """
    return entity_type in PII_TYPES


if __name__ == "__main__":
    # Test the label system
    print("Entity Types:", ENTITY_TYPES)
    print(f"\nTotal Labels: {NUM_LABELS}")
    print("\nBIO Labels:", LABELS)
    print("\nLabel to ID mapping (first 5):")
    for label in LABELS[:5]:
        print(f"  {label}: {LABEL2ID[label]}")
    
    # Test alignment
    print("\n--- Testing Label Alignment ---")
    test_text = "my email is john at test dot com"
    test_entities = [{"start": 12, "end": 32, "label": "EMAIL"}]
    test_tokens = ["my", "email", "is", "john", "at", "test", "dot", "com"]
    # Approximate offsets
    test_offsets = [(0, 2), (3, 8), (9, 11), (12, 16), (17, 19), (20, 24), (25, 28), (29, 32)]
    
    aligned_labels = align_labels_with_tokens(test_text, test_entities, test_offsets)
    
    print(f"Text: {test_text}")
    print(f"Tokens: {test_tokens}")
    print(f"Labels: {aligned_labels}")
    
    # Show PII classification
    print("\n--- PII Classification ---")
    for entity_type in ENTITY_TYPES:
        pii_status = "PII" if is_pii_entity(entity_type) else "Non-PII"
        print(f"{entity_type}: {pii_status}")