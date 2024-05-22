import json

def process_conversation(input_file_path, output_file_path):
    structured_conversation = []

    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{input_file_path}' not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    for line in lines:
        line = line.strip()  # To remove extra spaces and newline characters

        if (line.startswith("User:") or line.startswith("Polaris:")):
            speaker, text = line.split(":", 1)
            text = text.strip()
            text = line

        if text.endswith("?"):
            structured_conversation.append({"from": speaker, "text": text, "type": "question"})
        elif text.endswith("."):
            structured_conversation.append({"from": speaker, "text": text, "type": "statement"})
        elif text.endswith("!"):
            structured_conversation.append({"from": speaker, "text": text, "type": "exclamation"})
        elif text.endswith("..."):
            structured_conversation.append({"from": speaker, "text": text, "type": "suspense"})

    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(structured_conversation, outfile, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error writing to file: {e}")

input_file_path = 'input.txt'
output_file_path = 'output.json'
process_conversation(input_file_path, output_file_path)
