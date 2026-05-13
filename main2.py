from typing import List, Dict, Any, Optional
import json
import os
import re

from llm_sdk.llm_sdk import Small_LLM_Model


# ==============================
# LOAD JSON
# ==============================
def load_json(path: str) -> List[Dict[str, Any]]:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ==============================
# BUILD PROMPT
# ==============================
def build_prompt(
    functions: List[Dict[str, Any]],
    user_prompt: str
) -> str:
    """Build the LLM prompt."""
    fn_text = "Available functions:\n"

    for fn in functions:
        params = ", ".join(
            f"{k} ({v['type']})"
            for k, v in fn["parameters"].items()
        )

        fn_text += (
            f"- {fn['name']}\n"
            f"  parameters: {params}\n"
            f"  description: {fn['description']}\n\n"
        )

    prompt = (
        "You are a function calling system.\n\n"
        "You MUST choose EXACTLY one function from the list above.\n\n"
        "STRICT RULES:\n"
        "- Use EXACT function name\n"
        "- Use EXACT parameter names\n"
        "- Include ALL required parameters\n"
        "- DO NOT invent new parameters\n"
        "- Respect parameter types\n\n"
        "Output MUST be valid JSON:\n"
        "{\n"
        '  "name": "function_name",\n'
        '  "parameters": {\n'
        '    "param": value\n'
        "  }\n"
        "}\n\n"
        "Examples:\n\n"
        "Input: Greet John\n"
        'Output: {"name": "fn_greet", '
        '"parameters": {"name": "John"}}\n\n'
        "Input: Reverse the string 'hello'\n"
        'Output: {"name": "fn_reverse_string", '
        '"parameters": {"s": "hello"}}\n\n'
        "Input: What is the sum of 2 and 3?\n"
        'Output: {"name": "fn_add_numbers", '
        '"parameters": {"a": 2, "b": 3}}\n\n'
        "Input: What is the square root of 16?\n"
        'Output: {"name": "fn_get_square_root", '
        '"parameters": {"a": 16}}\n\n'
        f"User request:\n{user_prompt}\n\n"
        "Output:\n"
    )

    return fn_text + "\n" + prompt


# ==============================
# GENERATE WITH CONSTRAINTS
# ==============================
def generate(
    model: Small_LLM_Model,
    prompt: str,
    functions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate with delayed constrained decoding."""

    try:
        input_ids: List[int] = model.encode(prompt)[0].tolist()
        generated_ids: List[int] = []

        # Encode tokens (multi-token safe)
        tok_open = model.encode("{")[0].tolist()
        tok_close = model.encode("}")[0].tolist()
        tok_name = model.encode('"name"')[0].tolist()
        tok_params = model.encode('"parameters"')[0].tolist()
        tok_colon = model.encode(":")[0].tolist()
        tok_comma = model.encode(",")[0].tolist()

        encoded_functions = [
            model.encode(f'"{fn["name"]}"')[0].tolist()
            for fn in functions
        ]

        state = "free"
        prefix: List[int] = []

        def next_tokens(prefix_seq: List[int], sequences: List[List[int]]) -> List[int]:
            valid = [
                seq for seq in sequences
                if seq[:len(prefix_seq)] == prefix_seq
            ]
            return list({
                seq[len(prefix_seq)]
                for seq in valid
                if len(seq) > len(prefix_seq)
            })

        for _ in range(300):

            logits = model.get_logits_from_input_ids(input_ids)

            # ==============================
            # FREE MODE (NO CONSTRAINT)
            # ==============================
            if state == "free":
                next_token = max(range(len(logits)), key=lambda i: logits[i])
                input_ids.append(next_token)
                generated_ids.append(next_token)

                text = model.decode(generated_ids)

                # detect JSON start
                if "{" in text:
                    idx = text.index("{")

                    json_part = text[idx:]

                    # reset generation to JSON only
                    generated_ids = model.encode(json_part)[0].tolist()
                    input_ids = model.encode(prompt)[0].tolist() + generated_ids

                    state = "start"
                    prefix = []

                continue

            # ==============================
            # CONSTRAINED STATES
            # ==============================
            allowed_ids: List[int] = []

            if state == "start":
                allowed_ids = next_tokens(prefix, [tok_open])

            elif state == "name_key":
                allowed_ids = next_tokens(prefix, [tok_name])

            elif state == "name_colon":
                allowed_ids = next_tokens(prefix, [tok_colon])

            elif state == "function_name":
                allowed_ids = next_tokens(prefix, encoded_functions)

            elif state == "after_function":
                allowed_ids = next_tokens(prefix, [tok_comma])

            elif state == "params_key":
                allowed_ids = next_tokens(prefix, [tok_params])

            elif state == "params_colon":
                allowed_ids = next_tokens(prefix, [tok_colon])

            elif state == "params_open":
                allowed_ids = next_tokens(prefix, [tok_open])

            elif state == "params_close":
                allowed_ids = next_tokens(prefix, [tok_close])

            elif state == "end":
                break

            # ==============================
            # FILTER LOGITS
            # ==============================
            if allowed_ids:
                for i in range(len(logits)):
                    if i not in allowed_ids:
                        logits[i] = float("-inf")

            next_token = max(range(len(logits)), key=lambda i: logits[i])

            input_ids.append(next_token)
            generated_ids.append(next_token)

            prefix.append(next_token)

            def matched(seq: List[int]) -> bool:
                return prefix == seq

            # ==============================
            # STATE TRANSITIONS
            # ==============================
            if state == "start" and matched(tok_open):
                state = "name_key"
                prefix = []

            elif state == "name_key" and matched(tok_name):
                state = "name_colon"
                prefix = []

            elif state == "name_colon" and matched(tok_colon):
                state = "function_name"
                prefix = []

            elif state == "function_name":
                for fn, seq in zip(functions, encoded_functions):
                    if matched(seq):
                        state = "after_function"
                        prefix = []
                        break

            elif state == "after_function" and matched(tok_comma):
                state = "params_key"
                prefix = []

            elif state == "params_key" and matched(tok_params):
                state = "params_colon"
                prefix = []

            elif state == "params_colon" and matched(tok_colon):
                state = "params_open"
                prefix = []

            elif state == "params_open" and matched(tok_open):
                state = "params_close"
                prefix = []

            elif state == "params_close" and matched(tok_close):
                state = "end"
                prefix = []

        output_text = model.decode(generated_ids)

        match = re.search(r"\{.*\}", output_text, re.DOTALL)

        if match:
            return json.loads(match.group())

        return {"name": "", "parameters": {}}

    except Exception as exc:
        return {"name": "", "parameters": {}, "error": str(exc)}
# ==============================
# VALIDATION
# ==============================
def validate_output(
    output: Dict[str, Any],
    functions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Validate schema."""
    fn_name: str = output.get("name", "")
    params: Dict[str, Any] = output.get("parameters", {})

    fn = next(
        (f for f in functions if f["name"] == fn_name),
        None
    )

    if fn is None:
        return {"name": "", "parameters": {}}

    expected = set(fn["parameters"].keys())

    if set(params.keys()) != expected:
        return {"name": "", "parameters": {}}

    return output


# ==============================
# SAVE
# ==============================
def save_results(
    path: str,
    results: List[Dict[str, Any]]
) -> None:
    """Save results."""
    dir_path = os.path.dirname(path)

    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


# ==============================
# MAIN
# ==============================
def main() -> None:
    """Main entry."""
    functions = load_json(
        "data/input/functions_definition.json"
    )
    prompts = load_json(
        "data/input/function_calling_tests.json"
    )

    model = Small_LLM_Model()

    results: List[Dict[str, Any]] = []

    for item in prompts:
        prompt_text: str = item["prompt"]

        full_prompt = build_prompt(
            functions,
            prompt_text
        )

        output = generate(model, full_prompt, functions)
        output = validate_output(output, functions)

        print("prompt:", prompt_text)
        print("output:", output)
        print("************")

        results.append({
            "prompt": prompt_text,
            "name": output.get("name", ""),
            "parameters": output.get("parameters", {}),
        })

    save_results(
        "data/output/function_calling_results.json",
        results
    )


if __name__ == "__main__":
    main()