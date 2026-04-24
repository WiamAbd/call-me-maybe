from typing import List, Dict, Any
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
        'Input: Replace numbers in "Hello 12" with NUM\n'
        'Output: {"name": "fn_substitute_string_with_regex", '
        '"parameters": {"source_string": "Hello 12", '
        '"regex": "\\\\d+", "replacement": "NUM"}}\n\n'
        f"User request:\n{user_prompt}\n\n"
        "Output:\n"
    )

    return fn_text + "\n" + prompt


# ==============================
# GENERATE
# ==============================
def generate(
    model: Small_LLM_Model,
    prompt: str
) -> Dict[str, Any]:
    """Generate function call using LLM."""
    try:
        input_ids: List[int] = model.encode(prompt)[0].tolist()
        generated_ids: List[int] = []

        for _ in range(300):
            logits: List[float] = model.get_logits_from_input_ids(
                input_ids
            )
            next_token: int = max(
                range(len(logits)),
                key=lambda i: logits[i]
            )

            input_ids.append(next_token)
            generated_ids.append(next_token)

            text = model.decode(generated_ids)

            if "}" in text:
                break

        output_text: str = model.decode(generated_ids)

        match = re.search(r"\{.*\}", output_text, re.DOTALL)

        if match:
            return json.loads(match.group())

        return {"name": "", "parameters": {}}

    except Exception as exc:
        return {
            "name": "",
            "parameters": {},
            "error": str(exc),
        }


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

        output = generate(model, full_prompt)
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