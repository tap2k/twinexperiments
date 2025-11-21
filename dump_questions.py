"""
Dump all Wave 4 questions from the dataset, optionally filtered by block.

Usage:
    python dump_questions.py [options]

Options:
    --block BLOCK           Filter by block name (case-insensitive partial match)
    --output FILE           Output CSV file (default: data/questions_dump.csv)
    --personas N            Number of personas to extract from (default: all)
    --data-dir PATH         Data directory (default: ../Twin-2K-500)
    --list-blocks           List all unique block names and exit

Examples:
    python dump_questions.py --list-blocks
    python dump_questions.py --block "False consensus"
    python dump_questions.py --block anchoring --output data/anchoring_questions.csv
    python dump_questions.py --personas 10
"""

import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from utils import extract_wave4_questions


DEFAULT_DATA_DIR = Path(__file__).parent.parent / "Twin-2K-500"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "questions_dump.csv"


def load_wave4_ground_truth(data_dir):
    """Load Wave 4 ground truth from parquet files."""
    wave4_chunks = list((data_dir / "wave_split/chunks").glob("*.parquet"))

    if not wave4_chunks:
        raise FileNotFoundError(f"No parquet files found in {data_dir / 'wave_split/chunks'}")

    # Load all chunks and concatenate
    dfs = [pd.read_parquet(chunk) for chunk in wave4_chunks]
    return pd.concat(dfs, ignore_index=True)


def list_all_blocks(wave4_df):
    """List all unique block names in the dataset."""
    all_blocks = set()

    for idx, row in wave4_df.iterrows():
        wave4_json = row['wave4_Q_wave4_A']
        if not wave4_json:
            continue

        questions = extract_wave4_questions(wave4_json)
        for q in questions:
            all_blocks.add(q.get('block_name', 'Unknown'))

    return sorted(all_blocks)


def dump_questions(wave4_df, block_filter=None, max_personas=None):
    """Extract all questions from the dataset.

    Args:
        wave4_df: DataFrame with Wave 4 ground truth
        block_filter: Optional block name to filter questions
        max_personas: Optional limit on number of personas to process

    Returns:
        List of question dictionaries
    """
    all_questions = []
    personas_processed = 0

    for idx, row in wave4_df.iterrows():
        if max_personas and personas_processed >= max_personas:
            break

        pid = row['pid']
        wave4_json = row['wave4_Q_wave4_A']

        if not wave4_json:
            continue

        questions = extract_wave4_questions(wave4_json, block_filter=block_filter)

        for q in questions:
            all_questions.append({
                'pid': pid,
                'block_name': q.get('block_name', 'Unknown'),
                'question_id': q['question_id'],
                'question_type': q['question_type'],
                'question_text': q['question_text'],
                'answer': q['answer'],
                'options': '|'.join(q.get('options', [])) if q.get('options') else ''
            })

        personas_processed += 1

    return all_questions


def main():
    """Dump Wave 4 questions to CSV."""
    parser = argparse.ArgumentParser(
        description='Dump Wave 4 questions from the dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dump_questions.py --list-blocks
  python dump_questions.py --block "False consensus"
  python dump_questions.py --block anchoring --output data/anchoring_questions.csv
  python dump_questions.py --personas 10
        """
    )
    parser.add_argument('--block', type=str, default=None,
                        help='Filter by block name (case-insensitive partial match)')
    parser.add_argument('--output', type=str, default=None,
                        help=f'Output CSV file (default: auto-generated based on filters)')
    parser.add_argument('--personas', type=int, default=None,
                        help='Number of personas to extract from (default: all)')
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help=f'Data directory (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--list-blocks', action='store_true',
                        help='List all unique block names and exit')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Auto-generate output filename if not provided
    if args.output:
        output_file = Path(args.output)
    else:
        # Create descriptive filename based on filters
        if args.block:
            # Sanitize block name for filename (remove spaces, special chars)
            safe_block = args.block.lower().replace(' ', '_').replace('-', '_')
            safe_block = ''.join(c for c in safe_block if c.isalnum() or c == '_')
            output_file = Path(__file__).parent / "data" / f"questions_{safe_block}.csv"
        else:
            output_file = DEFAULT_OUTPUT

    print("=" * 80)
    print("WAVE 4 QUESTIONS DUMP")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print()

    # Load data
    print("Loading Wave 4 data...")
    wave4_df = load_wave4_ground_truth(data_dir)
    print(f"  Loaded {len(wave4_df)} Wave 4 responses")
    print()

    # List blocks if requested
    if args.list_blocks:
        print("Extracting all unique block names...")
        blocks = list_all_blocks(wave4_df)
        print(f"\nFound {len(blocks)} unique blocks:")
        print("-" * 80)
        for i, block in enumerate(blocks, 1):
            print(f"{i:2}. {block}")
        return

    # Dump questions
    if args.block:
        print(f"Filtering by block: {args.block}")
    if args.personas:
        print(f"Processing first {args.personas} personas")
    print()

    print("Extracting questions...")
    questions = dump_questions(wave4_df, block_filter=args.block, max_personas=args.personas)

    if not questions:
        print("No questions found!")
        return

    # Save to CSV
    df = pd.DataFrame(questions)

    # Create output directory if needed
    output_file.parent.mkdir(exist_ok=True)

    df.to_csv(output_file, index=False)

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"Total questions extracted: {len(questions)}")
    print(f"Unique personas: {df['pid'].nunique()}")
    print(f"Unique blocks: {df['block_name'].nunique()}")
    print(f"Unique question types: {df['question_type'].nunique()}")
    print()

    # Question type breakdown
    print("Question type breakdown:")
    print(df['question_type'].value_counts().to_string())
    print()

    # Block breakdown (if not filtering by block)
    if not args.block:
        print("Questions per block:")
        print(df['block_name'].value_counts().to_string())
        print()

    print(f"Output saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
