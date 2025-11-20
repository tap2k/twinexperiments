"""
Shared utility functions for Twin-2K experiments.
"""

import json


def create_prompt(persona_summary: str, question: str, question_type: str, options: list = None) -> str:
    """Create a prompt for the LLM to answer a question as a persona.

    Args:
        persona_summary: Formatted persona information
        question: The question text
        question_type: Type of question (MC, Matrix, TE, Slider)
        options: List of answer options (if applicable)

    Returns:
        Formatted prompt string
    """
    options_text = ""
    if options and len(options) > 0:
        if question_type == "Slider":
            options_text = f"\n{options[0]}"
        else:
            options_text = "\n\nOPTIONS:\n" + "\n".join([f"- {opt}" for opt in options])

    # Create instruction based on question type
    if question_type == "MC":
        instruction = "Respond with ONLY the exact text of your chosen option from the list above. Do not include any explanation or preamble."
    elif question_type == "Matrix":
        instruction = "Respond with ONLY the exact text of your chosen option from the list above. Do not include any explanation or preamble."
    elif question_type == "TE":
        instruction = "Respond with only your answer (a number or short text). Do not include any explanation or preamble."
    elif question_type == "Slider":
        instruction = "Respond with ONLY a number within the range shown above. Do not include any explanation or preamble."
    else:
        instruction = "Respond with only your answer. Do not include any explanation or preamble."

    return f"""You are simulating a person based on their survey responses. Answer the following question as this person would.

PERSONA INFORMATION:
{persona_summary}

QUESTION:
{question}{options_text}

INSTRUCTIONS:
{instruction}"""


def extract_wave4_questions(wave4_json_str, block_filter=None):
    """Extract questions from Wave 4 JSON data with answer options and block categories.

    Args:
        wave4_json_str: JSON string containing Wave 4 questions
        block_filter: Optional block name to filter questions (case-insensitive partial match)

    Returns:
        List of question dictionaries with keys: question_id, question_text, question_type,
        answer, options, block_name
    """
    if not wave4_json_str:
        return []

    # Parse JSON string
    data = json.loads(wave4_json_str)

    questions = []
    # Iterate through blocks
    for block in data:
        if block.get('ElementType') == 'Block':
            block_name = block.get('BlockName', 'Unknown')

            # Skip block if filter is specified and doesn't match
            if block_filter and block_filter.lower() not in block_name.lower():
                continue

            # Iterate through questions in the block
            for question in block.get('Questions', []):
                # Skip descriptive questions
                if question.get('is_descriptive'):
                    continue

                q_text = question.get('QuestionText', '')
                q_type = question.get('QuestionType', '')
                answers = question.get('Answers', {})

                # Extract answer options and ground truth answer
                answer = None
                options = []

                if q_type == 'MC':
                    answer = answers.get('SelectedText', '')
                    options = question.get('Options', [])
                elif q_type == 'Matrix':
                    # For matrix, use first selected text and first row
                    selected_texts = answers.get('SelectedText', [])
                    if selected_texts:
                        answer = selected_texts[0] if isinstance(selected_texts, list) else selected_texts
                    options = question.get('Columns', [])
                    # Include first row for context
                    rows = question.get('Rows', [])
                    if rows:
                        q_text = f"{q_text}\n\n{rows[0]}"
                elif q_type == 'TE':
                    answer = answers.get('Text', '')
                    # No options for text entry
                elif q_type == 'Slider':
                    values = answers.get('Values', [])
                    if values:
                        answer = values[0] if isinstance(values, list) else values
                    # Include slider range
                    range_info = question.get('Range', {})
                    if range_info:
                        min_val = range_info.get('Min', 0)
                        max_val = range_info.get('Max', 100)
                        options = [f"Range: {min_val} to {max_val}"]

                if q_text and answer:
                    questions.append({
                        'question_id': question.get('QuestionID', ''),
                        'question_text': q_text,
                        'question_type': q_type,
                        'answer': str(answer),
                        'options': options,
                        'block_name': block_name
                    })

    return questions
