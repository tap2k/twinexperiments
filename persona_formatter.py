"""
Flexible persona formatting system for Twin-2K experiments.

Allows mixing and matching different persona information components.
"""

import re


def extract_demographics(persona_summary: str) -> str:
    """Extract just the demographics section."""
    lines = persona_summary.split('\n')
    demographics = []
    in_demographics = False

    for line in lines:
        if "demographics are the following" in line.lower():
            in_demographics = True
            continue
        elif in_demographics and line.strip() and not line.startswith('The person'):
            demographics.append(line)
        elif in_demographics and line.startswith('The person'):
            break

    if demographics:
        return "The person's demographics are the following...\n" + '\n'.join(demographics)
    return ""


def extract_big5(persona_summary: str) -> str:
    """Extract Big 5 personality scores."""
    pattern = r"The person's Big 5 scores are the following:.*?(?=\n\nThe person's|$)"
    match = re.search(pattern, persona_summary, re.DOTALL)
    return match.group(0) if match else ""


def extract_qualitative(persona_summary: str) -> str:
    """Extract the three qualitative self-description questions."""
    pattern = r"The person also answered three purely qualitative questions.*$"
    match = re.search(pattern, persona_summary, re.DOTALL)
    return match.group(0) if match else ""


def extract_cognitive_scores(persona_summary: str) -> str:
    """Extract cognitive/intelligence scores (fluid, crystallized, CRT, etc.)."""
    sections = []

    # CRT score
    pattern = r"The person's CRT score is the following:.*?(?=\n\nThe person's|$)"
    match = re.search(pattern, persona_summary, re.DOTALL)
    if match:
        sections.append(match.group(0))

    # Fluid and crystallized intelligence
    pattern = r"The person's fluid and crystallized intelligence scores.*?(?=\n\nThe person's|$)"
    match = re.search(pattern, persona_summary, re.DOTALL)
    if match:
        sections.append(match.group(0))

    # Syllogism
    pattern = r"The person's syllogism score.*?(?=\n\nThe person's|$)"
    match = re.search(pattern, persona_summary, re.DOTALL)
    if match:
        sections.append(match.group(0))

    # Total intelligence and overconfidence
    pattern = r"The person's total intelligence scores.*?(?=\n\nThe person's|$)"
    match = re.search(pattern, persona_summary, re.DOTALL)
    if match:
        sections.append(match.group(0))

    # Numeracy
    pattern = r"The person's numeracy score.*?(?=\n\nThe person's|$)"
    match = re.search(pattern, persona_summary, re.DOTALL)
    if match:
        sections.append(match.group(0))

    # Financial literacy
    pattern = r"The person's financial literacy score.*?(?=\n\nThe person's|$)"
    match = re.search(pattern, persona_summary, re.DOTALL)
    if match:
        sections.append(match.group(0))

    return '\n\n'.join(sections)


def extract_economic_scores(persona_summary: str) -> str:
    """Extract economic game scores (ultimatum, dictator, trust, etc.)."""
    sections = []

    patterns = [
        r"The person's ultimatum game scores.*?(?=\n\nThe person's|$)",
        r"The person's trust game scores.*?(?=\n\nThe person's|$)",
        r"The person's dictator game score.*?(?=\n\nThe person's|$)",
        r"The person's mental accounting score.*?(?=\n\nThe person's|$)",
        r"The person's discount rate and present bias.*?(?=\n\nThe person's|$)",
        r"The person's risk aversion score.*?(?=\n\nThe person's|$)",
        r"The person's loss aversion score.*?(?=\n\nThe person's|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, persona_summary, re.DOTALL)
        if match:
            sections.append(match.group(0))

    return '\n\n'.join(sections)


def extract_personality_scores(persona_summary: str) -> str:
    """Extract personality and values scores (agency, communion, empathy, etc.)."""
    sections = []

    patterns = [
        r"The person's need for cognition score.*?(?=\n\nThe person's|$)",
        r"The person's agentic / communal value scores.*?(?=\n\nThe person's|$)",
        r"The person's minimalism score.*?(?=\n\nThe person's|$)",
        r"The person's basic empathy scale score.*?(?=\n\nThe person's|$)",
        r"The person's G\.R\.E\.E\.N\. score.*?(?=\n\nThe person's|$)",
        r"The person's individualism vs collectivism scores.*?(?=\n\nThe person's|$)",
        r"The person's need for uniqueness score.*?(?=\n\nThe person's|$)",
        r"The person's need for closure score.*?(?=\n\nThe person's|$)",
        r"The person's maximization scale score.*?(?=\n\nThe person's|$)",
        r"The person's regulatory focus scale score.*?(?=\n\nThe person's|$)",
        r"The person's tightwad-spendthrift score.*?(?=\n\nThe person's|$)",
        r"The person's self-monitoring score.*?(?=\n\nThe person's|$)",
        r"The person's self-concept clarity score.*?(?=\n\nThe person's|$)",
        r"The person's social desirability score.*?(?=\n\nThe person's|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, persona_summary, re.DOTALL)
        if match:
            sections.append(match.group(0))

    return '\n\n'.join(sections)


def extract_wellbeing_scores(persona_summary: str) -> str:
    """Extract wellbeing scores (anxiety, depression)."""
    sections = []

    patterns = [
        r"The person's Beck anxiety score.*?(?=\n\nThe person's|$)",
        r"The person's Beck depression score.*?(?=\n\nThe person's|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, persona_summary, re.DOTALL)
        if match:
            sections.append(match.group(0))

    return '\n\n'.join(sections)


# Predefined persona format combinations
PERSONA_FORMATS = {
    # Minimal formats
    'empty': lambda row: "",
    'demographics_only': lambda row: extract_demographics(row['persona_summary']),

    # Small formats
    'demographics_big5': lambda row: '\n\n'.join(filter(None, [
        extract_demographics(row['persona_summary']),
        extract_big5(row['persona_summary'])
    ])),

    'demographics_qualitative': lambda row: '\n\n'.join(filter(None, [
        extract_demographics(row['persona_summary']),
        extract_qualitative(row['persona_summary'])
    ])),

    # Medium formats
    'demographics_personality': lambda row: '\n\n'.join(filter(None, [
        extract_demographics(row['persona_summary']),
        extract_big5(row['persona_summary']),
        extract_personality_scores(row['persona_summary'])
    ])),

    'demographics_cognitive': lambda row: '\n\n'.join(filter(None, [
        extract_demographics(row['persona_summary']),
        extract_cognitive_scores(row['persona_summary'])
    ])),

    'demographics_economic': lambda row: '\n\n'.join(filter(None, [
        extract_demographics(row['persona_summary']),
        extract_economic_scores(row['persona_summary'])
    ])),

    # Full formats
    'summary': lambda row: row['persona_summary'],
    'full_text': lambda row: row['persona_text'],

    # Custom combinations
    'demographics_big5_qualitative': lambda row: '\n\n'.join(filter(None, [
        extract_demographics(row['persona_summary']),
        extract_big5(row['persona_summary']),
        extract_qualitative(row['persona_summary'])
    ])),

    'demographics_cognitive_economic': lambda row: '\n\n'.join(filter(None, [
        extract_demographics(row['persona_summary']),
        extract_cognitive_scores(row['persona_summary']),
        extract_economic_scores(row['persona_summary'])
    ])),

    'all_scores_no_demographics': lambda row: '\n\n'.join(filter(None, [
        extract_big5(row['persona_summary']),
        extract_personality_scores(row['persona_summary']),
        extract_cognitive_scores(row['persona_summary']),
        extract_economic_scores(row['persona_summary']),
        extract_wellbeing_scores(row['persona_summary'])
    ])),
}


def format_persona(persona_row, format_name='summary'):
    """
    Format a persona using a predefined or custom format.

    Args:
        persona_row: DataFrame row containing persona data
        format_name: One of the predefined format names or 'custom'

    Returns:
        Formatted persona string
    """
    if format_name in PERSONA_FORMATS:
        return PERSONA_FORMATS[format_name](persona_row)
    else:
        raise ValueError(f"Unknown format: {format_name}. Available formats: {list(PERSONA_FORMATS.keys())}")


def create_custom_format(components):
    """
    Create a custom persona format from a list of component names.

    Args:
        components: List of component names like ['demographics', 'big5', 'qualitative']

    Returns:
        Function that formats a persona row
    """
    component_extractors = {
        'demographics': extract_demographics,
        'big5': extract_big5,
        'qualitative': extract_qualitative,
        'cognitive': extract_cognitive_scores,
        'economic': extract_economic_scores,
        'personality': extract_personality_scores,
        'wellbeing': extract_wellbeing_scores,
    }

    def custom_formatter(row):
        sections = []
        for component in components:
            if component in component_extractors:
                section = component_extractors[component](row['persona_summary'])
                if section:
                    sections.append(section)
        return '\n\n'.join(sections)

    return custom_formatter


# List available formats for easy reference
def list_formats():
    """Print all available persona formats."""
    print("Available persona formats:")
    print("\nMinimal formats:")
    print("  - empty: No persona information")
    print("  - demographics_only: Just demographics (region, gender, age, etc.)")

    print("\nSmall formats:")
    print("  - demographics_big5: Demographics + Big 5 personality")
    print("  - demographics_qualitative: Demographics + self-descriptions")

    print("\nMedium formats:")
    print("  - demographics_personality: Demographics + Big 5 + personality scores")
    print("  - demographics_cognitive: Demographics + cognitive/intelligence scores")
    print("  - demographics_economic: Demographics + economic game scores")

    print("\nFull formats:")
    print("  - summary: Full persona summary (~13.6KB)")
    print("  - full_text: Complete question/answer history (~129KB)")

    print("\nCustom combinations:")
    print("  - demographics_big5_qualitative")
    print("  - demographics_cognitive_economic")
    print("  - all_scores_no_demographics")

    print("\nOr create your own with create_custom_format(['demographics', 'big5', ...])")
