MAX_SEQ_LENGTH = 1024
def analyze_sequence_lengths(dataset, tokenizer):
    """
    Analyze sequence lengths in the dataset to determine if MAX_SEQ_LENGTH needs adjustment
    """
    print("\n" + "="*60)
    print("SEQUENCE LENGTH ANALYSIS")
    print("="*60)
    
    lengths = []
    for item in dataset:
        try:
            text = f"{item['prompt']}{item['completion']}"
        except KeyError:
            text = item['text']
        tokens = tokenizer.encode(text, add_special_tokens=True)
        lengths.append(len(tokens))
    
    lengths.sort()
    
    max_len = max(lengths)
    min_len = min(lengths)
    avg_len = sum(lengths) / len(lengths)
    median_len = lengths[len(lengths) // 2]
    
    # Calculate percentiles
    p50 = lengths[int(len(lengths) * 0.50)]
    p75 = lengths[int(len(lengths) * 0.75)]
    p90 = lengths[int(len(lengths) * 0.90)]
    p95 = lengths[int(len(lengths) * 0.95)]
    p99 = lengths[int(len(lengths) * 0.99)]
    
    # Count sequences that will be truncated
    truncated_count = sum(1 for l in lengths if l > MAX_SEQ_LENGTH)
    truncated_pct = (truncated_count / len(lengths)) * 100
    
    print(f"Total examples: {len(lengths)}")
    print(f"\nSequence Length Statistics:")
    print(f"  Min length:     {min_len:6d} tokens")
    print(f"  Max length:     {max_len:6d} tokens")
    print(f"  Mean length:    {avg_len:6.1f} tokens")
    print(f"  Median length:  {median_len:6d} tokens")
    print(f"\nPercentiles:")
    print(f"  50th percentile: {p50:6d} tokens")
    print(f"  75th percentile: {p75:6d} tokens")
    print(f"  90th percentile: {p90:6d} tokens")
    print(f"  95th percentile: {p95:6d} tokens")
    print(f"  99th percentile: {p99:6d} tokens")
    print(f"\nTruncation Analysis (MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}):")
    print(f"  Sequences to be truncated: {truncated_count}/{len(lengths)} ({truncated_pct:.2f}%)")
    
    if truncated_count > 0:
        print(f"\n⚠️  WARNING: {truncated_count} sequences will be truncated!")
        print(f"   Consider increasing MAX_SEQ_LENGTH to at least {p95} (95th percentile)")
        print(f"   or {max_len} (to avoid any truncation)")
    else:
        print(f"\n✓ All sequences fit within MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}")
        # Suggest optimization if there's lots of room
        if p99 < MAX_SEQ_LENGTH * 0.7:
            print(f"   You could reduce MAX_SEQ_LENGTH to {p99} to save memory")
    
    print("="*60 + "\n")
    
    return {
        'max': max_len,
        'min': min_len,
        'mean': avg_len,
        'median': median_len,
        'p95': p95,
        'p99': p99,
        'truncated_count': truncated_count,
        'truncated_pct': truncated_pct
    }
