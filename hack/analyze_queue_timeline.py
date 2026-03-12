#!/usr/bin/env python3
"""Analyze timeline differences between queue_size=-1 and queue_size=0."""

import re
import sys
import time
from typing import List, Tuple, Dict
from datetime import datetime

# matplotlib is optional
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, skipping visualization")

class TimelineEvent:

    def __init__(self, timestamp: float, event_type: str, batch_id: int, thread: str, details: str = ""):
        self.timestamp = timestamp
        self.event_type = event_type
        self.batch_id = batch_id
        self.thread = thread
        self.details = details
    
    def __repr__(self):
        return f"TimelineEvent({self.timestamp:.3f}, {self.event_type}, batch={self.batch_id}, thread={self.thread})"

def parse_log_file(log_file: str) -> List[TimelineEvent]:

    events = []
    
    # Pattern: [PG0] Batch 0: Producer starting add_filenames
    pattern = r'\[PG\d+\]\s+Batch\s+(\d+):\s+(Producer|Consumer)\s+(\w+)\s+(.*)'
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                batch_id = int(match.group(1))
                thread = match.group(2)
                event_type = match.group(3)
                details = match.group(4)
                
                # extract timestamp from line start
                timestamp_match = re.search(r'(\d+\.\d+)', line)
                if timestamp_match:
                    timestamp = float(timestamp_match.group(1))
                    events.append(TimelineEvent(timestamp, event_type, batch_id, thread, details))
    
    return events

def visualize_timeline(events: List[TimelineEvent], queue_size: int, output_file: str):

    if not HAS_MATPLOTLIB:
        print("Skipping visualization: matplotlib not installed")
        return
    
    if not events:
        print("No events to visualize")
        return
    

    batches = {}
    for event in events:
        if event.batch_id not in batches:
            batches[event.batch_id] = []
        batches[event.batch_id].append(event)
    

    fig, ax = plt.subplots(figsize=(15, len(batches) * 2))
    

    colors = {
        'Producer': {'add_filenames': '#FF6B6B', 'copy_files': '#4ECDC4', 'queue_put': '#45B7D1'},
        'Consumer': {'queue_get': '#FFA07A', 'get_tensor': '#98D8C8', 'close': '#F7DC6F'}
    }
    

    base_time = min(e.timestamp for e in events)
    

    y_pos = 0
    for batch_id in sorted(batches.keys()):
        batch_events = sorted(batches[batch_id], key=lambda e: e.timestamp)
        

        producer_events = [e for e in batch_events if e.thread == 'Producer']
        for i in range(len(producer_events) - 1):
            start = producer_events[i].timestamp - base_time
            end = producer_events[i + 1].timestamp - base_time
            event_type = producer_events[i].event_type
            color = colors['Producer'].get(event_type, '#CCCCCC')
            
            ax.barh(y_pos, end - start, left=start, height=0.4, 
                   color=color, edgecolor='black', linewidth=0.5)
            ax.text(start + (end - start) / 2, y_pos, event_type, 
                   ha='center', va='center', fontsize=8)
        

        consumer_events = [e for e in batch_events if e.thread == 'Consumer']
        for i in range(len(consumer_events) - 1):
            start = consumer_events[i].timestamp - base_time
            end = consumer_events[i + 1].timestamp - base_time
            event_type = consumer_events[i].event_type
            color = colors['Consumer'].get(event_type, '#CCCCCC')
            
            ax.barh(y_pos + 0.5, end - start, left=start, height=0.4,
                   color=color, edgecolor='black', linewidth=0.5)
            ax.text(start + (end - start) / 2, y_pos + 0.5, event_type,
                   ha='center', va='center', fontsize=8)
        
        y_pos += 1
    

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Batch ID', fontsize=12)
    ax.set_title(f'ParallelLoader Timeline (queue_size={queue_size})', fontsize=14)
    ax.set_yticks(range(len(batches)))
    ax.set_yticklabels([f'Batch {i}' for i in sorted(batches.keys())])
    

    legend_elements = [
        mpatches.Patch(color='#FF6B6B', label='Producer: add_filenames'),
        mpatches.Patch(color='#4ECDC4', label='Producer: copy_files'),
        mpatches.Patch(color='#45B7D1', label='Producer: queue_put'),
        mpatches.Patch(color='#FFA07A', label='Consumer: queue_get'),
        mpatches.Patch(color='#98D8C8', label='Consumer: get_tensor'),
        mpatches.Patch(color='#F7DC6F', label='Consumer: close'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Timeline saved to: {output_file}")

def analyze_concurrency(events: List[TimelineEvent]) -> Dict:

    if not events:
        return {}
    

    producer_intervals = []
    consumer_intervals = []
    
    current_producer_start = None
    current_consumer_start = None
    
    for event in sorted(events, key=lambda e: e.timestamp):
        if event.thread == 'Producer':
            if event.event_type == 'starting':
                current_producer_start = event.timestamp
            elif event.event_type == 'completed' and current_producer_start:
                producer_intervals.append((current_producer_start, event.timestamp))
                current_producer_start = None
        
        elif event.thread == 'Consumer':
            if event.event_type == 'starting':
                current_consumer_start = event.timestamp
            elif event.event_type == 'completed' and current_consumer_start:
                consumer_intervals.append((current_consumer_start, event.timestamp))
                current_consumer_start = None
    

    overlap_time = 0
    for p_start, p_end in producer_intervals:
        for c_start, c_end in consumer_intervals:
            overlap_start = max(p_start, c_start)
            overlap_end = min(p_end, c_end)
            if overlap_start < overlap_end:
                overlap_time += overlap_end - overlap_start
    
    total_producer_time = sum(end - start for start, end in producer_intervals)
    total_consumer_time = sum(end - start for start, end in consumer_intervals)
    
    return {
        'total_producer_time': total_producer_time,
        'total_consumer_time': total_consumer_time,
        'overlap_time': overlap_time,
        'concurrency_ratio': overlap_time / max(total_producer_time, total_consumer_time) if max(total_producer_time, total_consumer_time) > 0 else 0
    }

def main():

    if len(sys.argv) < 2:
        print("Usage: python analyze_queue_timeline.py <log_file_queue_0> <log_file_queue_neg1>")
        print("Or: python analyze_queue_timeline.py --run-test <safetensors_files>")
        sys.exit(1)
    
    if sys.argv[1] == '--run-test':
        print("Running test mode...")
        # TODO: auto-run test and collect logs
        print("Please run test_queue_performance.py manually and save logs first")
        sys.exit(1)
    

    log_file_0 = sys.argv[1]
    log_file_neg1 = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Parsing log file: {log_file_0}")
    events_0 = parse_log_file(log_file_0)
    print(f"Found {len(events_0)} events")
    

    visualize_timeline(events_0, queue_size=0, output_file='timeline_queue_0.png')
    

    analysis_0 = analyze_concurrency(events_0)
    print(f"\nqueue_size=0 concurrency analysis:")
    print(f"  Producer total time: {analysis_0.get('total_producer_time', 0):.2f}s")
    print(f"  Consumer total time: {analysis_0.get('total_consumer_time', 0):.2f}s")
    print(f"  Overlap time: {analysis_0.get('overlap_time', 0):.2f}s")
    print(f"  Concurrency ratio: {analysis_0.get('concurrency_ratio', 0):.1%}")
    
    if log_file_neg1:
        print(f"\nParsing log file: {log_file_neg1}")
        events_neg1 = parse_log_file(log_file_neg1)
        print(f"Found {len(events_neg1)} events")
        
        visualize_timeline(events_neg1, queue_size=-1, output_file='timeline_queue_neg1.png')
        
        analysis_neg1 = analyze_concurrency(events_neg1)
        print(f"\nqueue_size=-1 concurrency analysis:")
        print(f"  Producer total time: {analysis_neg1.get('total_producer_time', 0):.2f}s")
        print(f"  Consumer total time: {analysis_neg1.get('total_consumer_time', 0):.2f}s")
        print(f"  Overlap time: {analysis_neg1.get('overlap_time', 0):.2f}s")
        print(f"  Concurrency ratio: {analysis_neg1.get('concurrency_ratio', 0):.1%}")
        

        print(f"\nConcurrency ratio comparison:")
        print(f"  queue_size=0:  {analysis_0.get('concurrency_ratio', 0):.1%}")
        print(f"  queue_size=-1: {analysis_neg1.get('concurrency_ratio', 0):.1%}")
        print(f"  Difference: {(analysis_0.get('concurrency_ratio', 0) - analysis_neg1.get('concurrency_ratio', 0)):.1%}")

if __name__ == "__main__":
    main()
