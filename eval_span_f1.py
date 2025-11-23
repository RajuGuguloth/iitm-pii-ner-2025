"""
Span-level evaluation for NER predictions.
Calculates precision, recall, and F1 for exact span matches.
Separates metrics for PII and non-PII entities.
"""

import json
from typing import List, Dict, Tuple, Set
from collections import defaultdict

from labels import PII_TYPES, NON_PII_TYPES, ENTITY_TYPES


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def entity_to_tuple(entity: Dict, example_id: str) -> Tuple:
    """
    Convert entity dict to tuple for comparison.
    
    Args:
        entity: Entity dict with 'start', 'end', 'label'
        example_id: Example ID
        
    Returns:
        Tuple of (example_id, start, end, label)
    """
    return (example_id, entity['start'], entity['end'], entity['label'])


def get_entity_sets(data: List[Dict]) -> Dict[str, Set[Tuple]]:
    """
    Extract entity sets from data, grouped by type.
    
    Args:
        data: List of examples with entities
        
    Returns:
        Dict mapping entity_type to set of entity tuples
    """
    entity_sets = defaultdict(set)
    all_entities = set()
    
    for example in data:
        example_id = example['id']
        for entity in example.get('entities', []):
            entity_tuple = entity_to_tuple(entity, example_id)
            entity_type = entity['label']
            
            entity_sets[entity_type].add(entity_tuple)
            all_entities.add(entity_tuple)
    
    entity_sets['ALL'] = all_entities
    
    return entity_sets


def calculate_metrics(
    true_entities: Set[Tuple],
    pred_entities: Set[Tuple]
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1.
    
    Args:
        true_entities: Set of ground truth entity tuples
        pred_entities: Set of predicted entity tuples
        
    Returns:
        Dict with precision, recall, f1
    """
    # True positives: entities in both sets
    tp = len(true_entities & pred_entities)
    
    # False positives: predicted but not in ground truth
    fp = len(pred_entities - true_entities)
    
    # False negatives: in ground truth but not predicted
    fn = len(true_entities - pred_entities)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'support': tp + fn  # Number of true entities
    }


def evaluate_predictions(
    true_data: List[Dict],
    pred_data: List[Dict]
) -> Dict:
    """
    Evaluate predictions against ground truth.
    
    Args:
        true_data: Ground truth examples
        pred_data: Predicted examples
        
    Returns:
        Dict with evaluation metrics
    """
    # Get entity sets
    true_entity_sets = get_entity_sets(true_data)
    pred_entity_sets = get_entity_sets(pred_data)
    
    # Calculate metrics for each entity type
    results = {}
    
    for entity_type in ENTITY_TYPES + ['ALL']:
        true_set = true_entity_sets[entity_type]
        pred_set = pred_entity_sets[entity_type]
        
        metrics = calculate_metrics(true_set, pred_set)
        results[entity_type] = metrics
    
    # Calculate PII-specific metrics
    pii_true = set()
    pii_pred = set()
    
    for entity_type in PII_TYPES:
        pii_true |= true_entity_sets[entity_type]
        pii_pred |= pred_entity_sets[entity_type]
    
    results['PII'] = calculate_metrics(pii_true, pii_pred)
    
    # Calculate Non-PII metrics
    non_pii_true = set()
    non_pii_pred = set()
    
    for entity_type in NON_PII_TYPES:
        non_pii_true |= true_entity_sets[entity_type]
        non_pii_pred |= pred_entity_sets[entity_type]
    
    results['NON_PII'] = calculate_metrics(non_pii_true, non_pii_pred)
    
    return results


def print_results(results: Dict):
    """
    Print evaluation results in a formatted table.
    
    Args:
        results: Evaluation results dict
    """
    print("\n" + "="*80)
    print("SPAN-LEVEL EVALUATION RESULTS")
    print("="*80)
    
    # Overall metrics
    print("\n📊 OVERALL METRICS")
    print("-"*80)
    all_metrics = results['ALL']
    print(f"Precision: {all_metrics['precision']:.4f}")
    print(f"Recall:    {all_metrics['recall']:.4f}")
    print(f"F1 Score:  {all_metrics['f1']:.4f}")
    print(f"Support:   {all_metrics['support']} entities")
    
    # PII vs Non-PII
    print("\n🔒 PII vs NON-PII BREAKDOWN")
    print("-"*80)
    
    pii_metrics = results['PII']
    print(f"\nPII Entities (CREDIT_CARD, PHONE, EMAIL, PERSON_NAME, DATE):")
    print(f"  Precision: {pii_metrics['precision']:.4f} ⭐ (Target: ≥0.80)")
    print(f"  Recall:    {pii_metrics['recall']:.4f}")
    print(f"  F1 Score:  {pii_metrics['f1']:.4f}")
    print(f"  Support:   {pii_metrics['support']} entities")
    
    non_pii_metrics = results['NON_PII']
    print(f"\nNon-PII Entities (CITY, LOCATION):")
    print(f"  Precision: {non_pii_metrics['precision']:.4f}")
    print(f"  Recall:    {non_pii_metrics['recall']:.4f}")
    print(f"  F1 Score:  {non_pii_metrics['f1']:.4f}")
    print(f"  Support:   {non_pii_metrics['support']} entities")
    
    # Per-entity type metrics
    print("\n📋 PER-ENTITY TYPE METRICS")
    print("-"*80)
    print(f"{'Entity Type':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-"*80)
    
    for entity_type in ENTITY_TYPES:
        metrics = results[entity_type]
        pii_marker = "🔒" if entity_type in PII_TYPES else "  "
        print(f"{pii_marker} {entity_type:<17} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} "
              f"{metrics['support']:<10}")
    
    # Check if PII precision meets target
    print("\n" + "="*80)
    if pii_metrics['precision'] >= 0.80:
        print("✅ PII PRECISION TARGET MET: {:.4f} ≥ 0.80".format(pii_metrics['precision']))
    else:
        print("❌ PII PRECISION TARGET NOT MET: {:.4f} < 0.80".format(pii_metrics['precision']))
    print("="*80)


def save_results(results: Dict, output_path: str):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Evaluation results dict
        output_path: Path to save results
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main evaluation function."""
    
    # Configuration
    TRUE_DATA_PATH = "data/test.jsonl"
    PRED_DATA_PATH = "out/predictions.jsonl"
    OUTPUT_PATH = "out/evaluation_results.json"
    
    print("="*80)
    print("NER SPAN-LEVEL EVALUATION")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    true_data = load_jsonl(TRUE_DATA_PATH)
    pred_data = load_jsonl(PRED_DATA_PATH)
    
    print(f"   Ground truth examples: {len(true_data)}")
    print(f"   Prediction examples: {len(pred_data)}")
    
    # Evaluate
    print("\n2. Evaluating predictions...")
    results = evaluate_predictions(true_data, pred_data)
    
    # Print results
    print_results(results)
    
    # Save results
    print("\n3. Saving results...")
    save_results(results, OUTPUT_PATH)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()