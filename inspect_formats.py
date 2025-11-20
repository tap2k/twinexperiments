"""
Inspect different persona formats to see what they contain.

Usage:
    python inspect_formats.py [options] [format_name]

Options:
    --persona-id PID        Persona ID to inspect (default: first persona)
    --max-chars N           Max characters to show (default: 500 for all, 2000 for single)
    --data-dir PATH         Data directory (default: ../Twin-2K-500)

Examples:
    python inspect_formats.py                           # Show all formats
    python inspect_formats.py demographics_big5         # Show specific format
    python inspect_formats.py --max-chars 1000 summary  # Show more chars
"""

import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from persona_formatter import format_persona, PERSONA_FORMATS

# Default configuration
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "Twin-2K-500"
DEFAULT_MAX_CHARS_ALL = 300
DEFAULT_MAX_CHARS_SINGLE = 2000


def load_persona_data(data_dir, num_personas=1):
    """Load persona summaries from parquet files."""
    persona_chunks = list((data_dir / "full_persona/chunks").glob("*.parquet"))

    if not persona_chunks:
        raise FileNotFoundError(f"No parquet files found in {data_dir / 'full_persona/chunks'}")

    # Load first chunk and get first N personas
    df = pd.read_parquet(persona_chunks[0])
    return df.head(num_personas)


def inspect_format(persona_row, format_name, max_chars=500):
    """Inspect a single format."""
    result = format_persona(persona_row, format_name)

    print(f"\n{'=' * 80}")
    print(f"FORMAT: {format_name}")
    print(f"{'=' * 80}")
    print(f"Length: {len(result):,} characters")
    print(f"\nContent (first {max_chars} chars):")
    print("-" * 80)
    if result:
        print(result[:max_chars])
        if len(result) > max_chars:
            print(f"\n... ({len(result) - max_chars:,} more characters)")
    else:
        print("(empty)")
    print()


def main():
    """Show persona format contents."""
    parser = argparse.ArgumentParser(
        description='Inspect persona formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_formats.py                           # Show all formats
  python inspect_formats.py demographics_big5         # Show specific format
  python inspect_formats.py --max-chars 1000 summary  # Show more chars
  python inspect_formats.py --persona-id 1348 empty   # Inspect different persona
        """
    )
    parser.add_argument('format', type=str, nargs='?', default=None,
                        help='Format name to inspect (if not provided, shows all)')
    parser.add_argument('--persona-id', type=int, default=None,
                        help='Persona ID to inspect (default: first persona)')
    parser.add_argument('--max-chars', type=int, default=None,
                        help=f'Max characters to show (default: {DEFAULT_MAX_CHARS_ALL} for all, {DEFAULT_MAX_CHARS_SINGLE} for single)')
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help=f'Data directory (default: {DEFAULT_DATA_DIR})')

    args = parser.parse_args()

    # Load persona data
    data_dir = Path(args.data_dir)
    persona_df = load_persona_data(data_dir, num_personas=100 if args.persona_id else 1)

    # Select persona
    if args.persona_id:
        # Find persona by ID
        persona_row = persona_df[persona_df['pid'] == args.persona_id]
        if len(persona_row) == 0:
            print(f"Error: Persona ID {args.persona_id} not found")
            return
        persona_row = persona_row.iloc[0]
    else:
        persona_row = persona_df.iloc[0]

    pid = persona_row['pid']

    print("=" * 80)
    print(f"PERSONA FORMAT INSPECTOR (PID: {pid})")
    print("=" * 80)

    # If format specified, show just that one with more detail
    if args.format:
        format_name = args.format
        if format_name not in PERSONA_FORMATS:
            print(f"\nError: Unknown format '{format_name}'")
            print(f"Available formats: {', '.join(sorted(PERSONA_FORMATS.keys()))}")
            return

        max_chars = args.max_chars if args.max_chars else DEFAULT_MAX_CHARS_SINGLE
        inspect_format(persona_row, format_name, max_chars=max_chars)
    else:
        # Show all formats with preview
        max_chars = args.max_chars if args.max_chars else DEFAULT_MAX_CHARS_ALL

        formats_by_size = [
            ('empty', 'demographics_only'),
            ('demographics_big5', 'demographics_qualitative', 'demographics_big5_qualitative'),
            ('demographics_personality', 'demographics_cognitive', 'demographics_economic'),
            ('demographics_cognitive_economic', 'all_scores_no_demographics'),
            ('summary', 'full_text'),
        ]

        for format_group in formats_by_size:
            for fmt in format_group:
                if fmt in PERSONA_FORMATS:
                    inspect_format(persona_row, fmt, max_chars=max_chars)

        print("=" * 80)
        print("\nTo see full content of a format, run:")
        print("  python inspect_formats.py <format_name>")
        print(f"\nAvailable formats: {', '.join(sorted(PERSONA_FORMATS.keys()))}")
        print("=" * 80)


if __name__ == "__main__":
    main()
