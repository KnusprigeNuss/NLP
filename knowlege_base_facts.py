import json

def extract_evidence_sentences_with_labels(file_path, desired_labels=['SUPPORTS']):
    """
    Extracts 'evidence' sentences from a CLIMATE-FEVER JSONL dataset file,
    filtering by their 'evidence_label'.

    Args:
        file_path (str): The path to the CLIMATE-FEVER dataset file.
        desired_labels (list): A list of evidence_labels to include (e.g., ['SUPPORTS']).
                               Valid labels are 'SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO'.

    Returns:
        list: A list of unique evidence sentences that match the desired labels.
    """
    evidence_sentences = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'evidences' in data and data['evidences'] is not None:
                    for evidence_item in data['evidences']:
                        if (
                            'evidence' in evidence_item and evidence_item['evidence'] is not None and
                            'evidence_label' in evidence_item and evidence_item['evidence_label'] in desired_labels
                        ):
                            sentence = evidence_item['evidence'].strip()
                            # Remove surrounding quotes if present from the original data
                            if sentence.startswith('"') and sentence.endswith('"'):
                                sentence = sentence[1:-1]
                            evidence_sentences.add(sentence)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the path.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file '{file_path}': {e}")
        print(f"Problematic line (or near): {line.strip()}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

    return list(evidence_sentences)

def save_facts_in_json_format(facts, output_file="climate_change_facts_db.txt"):
    """
    Saves a list of facts into a .txt file in the specified JSON-like format.

    Args:
        facts (list): A list of strings, where each string is a factual statement.
        output_file (str): The name of the output file.
    """
    # Create the dictionary structure
    output_data = {"support": facts}

    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            # Use json.dumps to correctly handle escaping of internal quotes
            # indent=2 for pretty printing, ensure_ascii=False to preserve non-ASCII chars
            json_string = json.dumps(output_data, indent=2, ensure_ascii=False)
            outfile.write(json_string)
        print(f"Facts successfully saved to '{output_file}' in JSON format.")
    except Exception as e:
        print(f"Error saving facts to file: {e}")


if __name__ == "__main__":
    dataset_file = 'climate-fever.jsonl' # Make sure this path is correct

    # Extract only 'SUPPORTS' facts, as these are the most direct facts for your KB
    print(f"Extracting SUPPORTS facts from: {dataset_file}")
    supported_facts = extract_evidence_sentences_with_labels(dataset_file, desired_labels=['SUPPORTS',"REFUTES"])

    if supported_facts:
        print(f"\nExtracted {len(supported_facts)} unique SUPPORTS facts.")
        print("\nFirst 5 SUPPORTS facts:")
        for i, fact in enumerate(supported_facts[:5]):
            print(f"{i+1}. {fact}")

        # Save these facts to a file in the desired JSON-like format
        save_facts_in_json_format(supported_facts, "climate_change_supported_refutes_facts_db.txt")
    else:
        print("No facts were extracted. Please check the file path and content.")