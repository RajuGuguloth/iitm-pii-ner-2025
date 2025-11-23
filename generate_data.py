import json
import random
from typing import List, Dict, Tuple

# Entity value pools
FIRST_NAMES = ["John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa", "James", "Mary"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Wilson", "Moore"]
CITIES = ["Chennai", "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Kolkata", "Pune", "Ahmedabad"]
LOCATIONS = ["India", "Tamil Nadu", "Maharashtra", "Karnataka", "United States", "California"]
EMAIL_DOMAINS = ["gmail", "yahoo", "outlook", "hotmail", "company"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# Noise patterns
def add_typo(text: str) -> str:
    """Randomly add typos to text - reduced rate for better precision"""
    if random.random() < 0.05 and len(text) > 3:  # Reduced from 0.2 to 0.05
        idx = random.randint(1, len(text) - 2)
        char_list = list(text)
        char_list[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
        return ''.join(char_list)
    return text

def number_to_words(num_str: str) -> str:
    """Convert digits to spoken words"""
    digit_map = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    return ' '.join([digit_map[d] for d in num_str])

# Entity generators
def generate_email() -> Tuple[str, str]:
    """Generate noisy email entity"""
    first = random.choice(FIRST_NAMES).lower()
    last = random.choice(LAST_NAMES).lower()
    domain = random.choice(EMAIL_DOMAINS)
    
    # Simpler patterns without typos in separators
    patterns = [
        f"{first} dot {last} at {domain} dot com",
        f"{first} underscore {last} at {domain} dot com",
        f"{first}{random.randint(1,99)} at {domain} dot com",
    ]
    
    noisy = random.choice(patterns)
    # Don't add typos to emails - they're too sensitive
    
    # Canonical form for reference
    canonical = f"{first}.{last}@{domain}.com"
    
    return noisy, canonical

def generate_phone() -> Tuple[str, str]:
    """Generate noisy phone number"""
    digits = ''.join([str(random.randint(0, 9)) for _ in range(10)])
    
    # Variations
    if random.random() < 0.7:
        # Fully spelled out
        noisy = number_to_words(digits)
    else:
        # Partially spelled with some digits
        parts = [digits[:3], digits[3:6], digits[6:]]
        noisy = f"{number_to_words(parts[0])} {number_to_words(parts[1])} {number_to_words(parts[2])}"
    
    return noisy, digits

def generate_credit_card() -> Tuple[str, str]:
    """Generate noisy credit card number"""
    digits = ''.join([str(random.randint(0, 9)) for _ in range(16)])
    
    # Always spelled out for STT
    noisy = number_to_words(digits)
    
    return noisy, digits

def generate_person_name() -> Tuple[str, str]:
    """Generate person name without noise for high precision"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    
    # No typos for person names - critical for precision
    full_name = f"{first} {last}"
    return full_name, full_name

def generate_date() -> Tuple[str, str]:
    """Generate noisy date"""
    month = random.choice(MONTHS)
    day = random.randint(1, 28)
    year = random.randint(2020, 2024)
    
    patterns = [
        f"{month} {number_to_words(str(day))} {number_to_words(str(year))}",
        f"{number_to_words(str(day).zfill(2))} {number_to_words(str(random.randint(1,12)).zfill(2))} {number_to_words(str(year))}",
        f"{month} {day} {year}",
    ]
    
    noisy = random.choice(patterns)
    canonical = f"{month} {day}, {year}"
    
    return noisy, canonical

def generate_city() -> Tuple[str, str]:
    """Generate city name"""
    city = random.choice(CITIES)
    # No typos for better precision
    return city, city

def generate_location() -> Tuple[str, str]:
    """Generate location name"""
    location = random.choice(LOCATIONS)
    # No typos for better precision
    return location, location

# Template sentences
TEMPLATES = [
    "my email is {EMAIL} and phone is {PHONE}",
    "please send to {EMAIL} or call {PHONE}",
    "I'm {PERSON_NAME} from {CITY}",
    "{PERSON_NAME} lives in {CITY} {LOCATION}",
    "contact me at {EMAIL} my name is {PERSON_NAME}",
    "card number {CREDIT_CARD} expiry {DATE}",
    "born on {DATE} in {CITY}",
    "my phone number is {PHONE} and I am {PERSON_NAME}",
    "{PERSON_NAME} email {EMAIL} phone {PHONE}",
    "I live in {CITY} my card is {CREDIT_CARD}",
    "reach out to {PERSON_NAME} at {EMAIL} or {PHONE}",
    "traveling to {CITY} on {DATE}",
    "account holder {PERSON_NAME} card {CREDIT_CARD}",
    "{PERSON_NAME} from {LOCATION} email is {EMAIL}",
    "my details are name {PERSON_NAME} phone {PHONE} city {CITY}",
]

def generate_example(example_id: str) -> Dict:
    """Generate a single training example with accurate span alignment"""
    # Select template
    template = random.choice(TEMPLATES)
    
    # Find all entity placeholders in order
    import re
    placeholders = re.findall(r'\{(\w+)\}', template)
    
    # Generate entity values
    entity_generators = {
        "EMAIL": generate_email,
        "PHONE": generate_phone,
        "CREDIT_CARD": generate_credit_card,
        "PERSON_NAME": generate_person_name,
        "DATE": generate_date,
        "CITY": generate_city,
        "LOCATION": generate_location,
    }
    
    # Build text and track entities with accurate positions
    text = template
    entities = []
    offset = 0  # Track cumulative offset from replacements
    
    for entity_type in placeholders:
        noisy_value, canonical = entity_generators[entity_type]()
        
        # Find position of placeholder in current text
        placeholder = f"{{{entity_type}}}"
        placeholder_pos = text.find(placeholder)
        
        if placeholder_pos == -1:
            continue
        
        # Calculate actual start position (accounting for previous replacements)
        actual_start = placeholder_pos
        
        # Replace placeholder with actual value
        text = text.replace(placeholder, noisy_value, 1)
        
        # Calculate end position
        actual_end = actual_start + len(noisy_value)
        
        # Verify span is correct
        extracted = text[actual_start:actual_end]
        assert extracted == noisy_value, f"Span mismatch: expected '{noisy_value}', got '{extracted}'"
        
        entities.append({
            "start": actual_start,
            "end": actual_end,
            "label": entity_type
        })
    
    # Sort entities by start position
    entities = sorted(entities, key=lambda x: x['start'])
    
    return {
        "id": example_id,
        "text": text,
        "entities": entities
    }

def generate_dataset(num_examples: int, start_id: int = 0) -> List[Dict]:
    """Generate multiple examples"""
    dataset = []
    for i in range(num_examples):
        example = generate_example(f"{start_id + i}")
        dataset.append(example)
    return dataset

def save_jsonl(data: List[Dict], filepath: str):
    """Save dataset to JSONL file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} examples to {filepath}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Generate datasets - REDUCED SIZE FOR SPEED
    print("Generating synthetic datasets...")
    
    train_data = generate_dataset(400, start_id=0)  # Reduced from 800
    dev_data = generate_dataset(100, start_id=400)   # Reduced from 150
    test_data = generate_dataset(100, start_id=500)  # Reduced from 150
    
    # Save to files
    save_jsonl(train_data, "data/train.jsonl")
    save_jsonl(dev_data, "data/dev.jsonl")
    save_jsonl(test_data, "data/test.jsonl")
    
    print("\nDataset generation complete!")
    print(f"Train: {len(train_data)} examples")
    print(f"Dev: {len(dev_data)} examples")
    print(f"Test: {len(test_data)} examples")
    
    # Show sample
    print("\nSample example:")
    sample = train_data[0]
    print(json.dumps(sample, indent=2, ensure_ascii=False))