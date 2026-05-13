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
    full_prompt = "Available functions:\n"
    for fn in functions:

        params = ", ".join(
            f"{k} ({v['type']})"
            for k, v in fn["parameters"].items()
        )

        full_prompt += (
            f"- {fn['name']}\n"
            f"  parameters: {params}\n"
            f"  description: {fn['description']}\n\n"
        )

    full_prompt += (
        "\n"
        "Available functions:\n\n"
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
        "Input: Greet Jack\n"
        'Output: {"name": "fn_greet", '
        '"parameters": {"name": "Jack"}}\n\n'
        "Input: Reverse the string 'Hola'\n"
        'Output: {"name": "fn_reverse_string", '
        '"parameters": {"s": "Hola"}}\n\n'
        "Input: What is the sum of 4 and 7?\n"
        'Output: {"name": "fn_add_numbers", '
        '"parameters": {"a": 4, "b": 7}}\n\n'
        "Input: What is the square root of 7?\n"
        'Output: {"name": "fn_get_square_root", '
        '"parameters": {"a": 7}}\n\n'
        'Input: Replace numbers in "Hello 12" with SMTHG\n'
        'Output: {"name": "fn_substitute_string_with_regex", '
        '"parameters": {"source_string": "Hello 12", '
        '"regex": "\\\\d+", "replacement": "SMTHG"}}\n\n'
        f"User request:\n{user_prompt}\n\n"
        "Output:\n"
    )

    return full_prompt


# ==============================
# GENERATE
# ==============================
def generate(
    model: Small_LLM_Model,
    prompt: str,
    functions: List[Dict[str, Any]]
) -> Dict[str, Any]:

    import re

    input_ids = model.encode(prompt)[0].tolist()
    generated_ids: List[int] = []

    def enc(s: str) -> List[int]:
        return model.encode(s)[0].tolist()

    # === STRUCTURE TOKENS ===
    tok_open = enc("{")
    tok_close = enc("}")
    tok_name = enc('"name"')
    tok_params = enc('"parameters"')
    tok_colon = enc(":")
    tok_comma = enc(",")

    # === FUNCTION TOKENS (fixed split) ===
    fn_tokens = {
        "fn_add_numbers": enc('"fn_add_numbers"'),
        "fn_greet": enc('"fn_greet"'),
        "fn_reverse_string": enc('"fn_reverse_string"'),
        "fn_get_square_root": enc('"fn_get_square_root"'),
        "fn_substitute_string_with_regex": enc('"fn_substitute_string_with_regex"'),
    }

    state = "start"
    prefix: List[int] = []

    selected_fn: Optional[str] = None
    param_keys: List[str] = []
    param_index = 0

    def next_token_from_seq(prefix_seq, seq):
        if seq[:len(prefix_seq)] == prefix_seq:
            if len(seq) > len(prefix_seq):
                return [seq[len(prefix_seq)]]
        return []

    for _ in range(300):

        logits = model.get_logits_from_input_ids(input_ids)
        allowed_ids: List[int] = []

        # =========================
        # STATE MACHINE
        # =========================

        if state == "start":
            allowed_ids = next_token_from_seq(prefix, tok_open)

        elif state == "name_key":
            allowed_ids = next_token_from_seq(prefix, tok_name)

        elif state == "name_colon":
            allowed_ids = next_token_from_seq(prefix, tok_colon)

        elif state == "function_name":
            for seq in fn_tokens.values():
                allowed_ids += next_token_from_seq(prefix, seq)

        elif state == "after_function":
            allowed_ids = next_token_from_seq(prefix, tok_comma)

        elif state == "params_key":
            allowed_ids = next_token_from_seq(prefix, tok_params)

        elif state == "params_colon":
            allowed_ids = next_token_from_seq(prefix, tok_colon)

        elif state == "params_open":
            allowed_ids = next_token_from_seq(prefix, tok_open)

        elif state == "param_name":
            for key in param_keys:
                allowed_ids += next_token_from_seq(prefix, enc(f'"{key}"'))

        elif state == "param_colon":
            allowed_ids = next_token_from_seq(prefix, tok_colon)

        elif state == "param_value":
            ptype = next(f for f in functions if f["name"] == selected_fn)["parameters"][param_keys[param_index]]["type"]

            if ptype == "number":
                for d in range(10):
                    allowed_ids += enc(str(d))

            elif ptype == "string":
                allowed_ids = enc('"')  # force string start

        elif state == "after_value":
            if param_index < len(param_keys) - 1:
                allowed_ids = next_token_from_seq(prefix, tok_comma)
            else:
                allowed_ids = next_token_from_seq(prefix, tok_close)

        elif state == "end":
            break

        # =========================
        # FILTER LOGITS
        # =========================
        if allowed_ids:
            for i in range(len(logits)):
                if i not in allowed_ids:
                    logits[i] = float("-inf")

        next_token = max(range(len(logits)), key=lambda i: logits[i])

        input_ids.append(next_token)
        generated_ids.append(next_token)

        prefix.append(next_token)

        def matched(seq):
            return prefix == seq

        # =========================
        # STATE TRANSITIONS
        # =========================

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
            for name, seq in fn_tokens.items():
                if matched(seq):
                    selected_fn = name
                    param_keys = list(
                        next(f for f in functions if f["name"] == name)["parameters"].keys()
                    )
                    param_index = 0
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
            state = "param_name"
            prefix = []

        elif state == "param_name":
            for key in param_keys:
                if matched(enc(f'"{key}"')):
                    state = "param_colon"
                    prefix = []
                    break

        elif state == "param_colon" and matched(tok_colon):
            state = "param_value"
            prefix = []

        elif state == "param_value":
            state = "after_value"
            prefix = []

        elif state == "after_value":
            if param_index < len(param_keys) - 1:
                if matched(tok_comma):
                    param_index += 1
                    state = "param_name"
                    prefix = []
            else:
                if matched(tok_close):
                    state = "end"
                    prefix = []

    output_text = model.decode(generated_ids)

    try:
        match = re.search(r"\{.*\}", output_text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        print("INVALID JSON:", output_text)

    return {"name": "", "parameters": {}}


# ==============================
# VALIDATION
# ==============================
def validate_output(
    output: Dict[str, Any],
    functions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Validate output against schema."""
    fn_name: str = output.get("name", "")
    params: Dict[str, Any] = output.get("parameters", {})

    fn = next(
        (f for f in functions if f["name"] == fn_name),
        None
    )

    if fn is None:
        return {"name": "", "parameters": {}}

    expected_params = set(fn["parameters"].keys())

    if set(params.keys()) != expected_params:
        return {"name": "", "parameters": {}}

    return output


# ==============================
# SAVE RESULTS
# ==============================
def save_results(
    path: str,
    results: List[Dict[str, Any]]
) -> None:
    """Save results to file."""
    dir_path: str = os.path.dirname(path)

    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


# ==============================
# MAIN
# ==============================
def main() -> None:
    """Main entry point."""
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

        full_prompt: str = build_prompt(
            functions,
            prompt_text
        )

        output = generate(model, full_prompt,functions)
        #output = validate_output(output, functions)

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