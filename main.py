from typing import List, Dict, Any, Optional
import json
import os

from llm_sdk.llm_sdk import Small_LLM_Model


# ==============================
# LOAD JSON
# ==============================
def load_json(path: str) -> List[Dict[str, Any]]:
    """Load JSON file safely."""

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ==============================
# BUILD PROMPT
# ==============================
def build_prompt(
    functions: List[Dict[str, Any]],
    user_prompt: str
) -> str:
    """Build prompt for the model."""

    prompt = "Available functions:\n\n"

    for fn in functions:

        params = ", ".join(
            f"{k} ({v['type']})"
            for k, v in fn["parameters"].items()
        )

        prompt += (
            f"- {fn['name']}\n"
            f"  parameters: {params}\n"
            f"  description: {fn['description']}\n\n"
        )

    prompt += (
        "You are a function calling system.\n\n"
        "You MUST choose EXACTLY one function.\n\n"
        "Return ONLY valid JSON.\n\n"
        "Format:\n"
        "{\n"
        '  "name": "function_name",\n'
        '  "parameters": {\n'
        '    "param": value\n'
        "  }\n"
        "}\n\n"
        f"User request:\n{user_prompt}\n\n"
        "Output:\n"
    )

    return prompt


# ==============================
# CONSTRAINED DECODING
# ==============================
def generate(
    model: Small_LLM_Model,
    prompt: str,
    functions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate structured JSON using constrained decoding."""

    input_ids: List[int] = model.encode(prompt)[0].tolist()
    generated_ids: List[int] = []

    def enc(text: str) -> List[int]:
        return model.encode(text)[0].tolist()

    # ==========================
    # FIXED TOKENS
    # ==========================
    tok_open = enc("{")
    tok_close = enc("}")
    tok_name = enc('"name"')
    tok_params = enc('"parameters"')
    tok_colon = enc(":")
    tok_comma = enc(",")

    # ==========================
    # FUNCTION TOKENS
    # ==========================
    fn_tokens = {
        fn["name"]: enc(f'"{fn["name"]}"')
        for fn in functions
    }

    state = "start"

    prefix: List[int] = []

    selected_fn: Optional[str] = None
    param_keys: List[str] = []
    param_index = 0

    # ==========================
    # PREFIX MATCHING
    # ==========================
    def next_tokens(
        prefix_seq: List[int],
        sequences: List[List[int]]
    ) -> List[int]:

        valid_sequences = [
            seq for seq in sequences
            if seq[:len(prefix_seq)] == prefix_seq
        ]

        return list({
            seq[len(prefix_seq)]
            for seq in valid_sequences
            if len(seq) > len(prefix_seq)
        })

    # ==========================
    # GENERATION LOOP
    # ==========================
    for _ in range(200):

        logits = model.get_logits_from_input_ids(input_ids)

        allowed_ids: List[int] = []

        # ==========================
        # STATE MACHINE
        # ==========================
        if state == "start":

            allowed_ids = next_tokens(
                prefix,
                [tok_open]
            )

        elif state == "name_key":

            allowed_ids = next_tokens(
                prefix,
                [tok_name]
            )

        elif state == "name_colon":

            allowed_ids = next_tokens(
                prefix,
                [tok_colon]
            )

        elif state == "function_name":

            allowed_ids = next_tokens(
                prefix,
                list(fn_tokens.values())
            )

        elif state == "after_function":

            allowed_ids = next_tokens(
                prefix,
                [tok_comma]
            )

        elif state == "params_key":

            allowed_ids = next_tokens(
                prefix,
                [tok_params]
            )

        elif state == "params_colon":

            allowed_ids = next_tokens(
                prefix,
                [tok_colon]
            )

        elif state == "params_open":

            allowed_ids = next_tokens(
                prefix,
                [tok_open]
            )

        elif state == "param_name":

            allowed_ids = next_tokens(
                prefix,
                [
                    enc(f'"{key}"')
                    for key in param_keys
                ]
            )

        elif state == "param_colon":

            allowed_ids = next_tokens(
                prefix,
                [tok_colon]
            )

        elif state == "param_value":

            # allow free generation for values
            allowed_ids = list(range(len(logits)))

        elif state == "after_value":

            if param_index < len(param_keys) - 1:

                allowed_ids = next_tokens(
                    prefix,
                    [tok_comma]
                )

            else:

                allowed_ids = next_tokens(
                    prefix,
                    [tok_close]
                )

        elif state == "end":
            break

        # ==========================
        # FILTER INVALID TOKENS
        # ==========================
        if allowed_ids:

            for i in range(len(logits)):

                if i not in allowed_ids:
                    logits[i] = float("-inf")

        # ==========================
        # TOKEN SELECTION
        # ==========================
        next_token = max(
            range(len(logits)),
            key=lambda i: logits[i]
        )

        input_ids.append(next_token)
        generated_ids.append(next_token)

        prefix.append(next_token)

        token_text = model.decode([next_token])

        # ==========================
        # MATCH HELPER
        # ==========================
        def matched(seq: List[int]) -> bool:
            return prefix == seq

        # ==========================
        # STATE TRANSITIONS
        # ==========================
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
                        next(
                            f for f in functions
                            if f["name"] == name
                        )["parameters"].keys()
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

            if len(param_keys) == 0:

                state = "end"

            else:

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

            # detect actual separators
            if token_text == ",":

                prefix = []

                if param_index < len(param_keys) - 1:

                    param_index += 1
                    state = "param_name"

            elif token_text == "}":

                prefix = []
                state = "end"

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

        # ==========================
        # STOP CONDITION
        # ==========================
        current_text = model.decode(generated_ids)

        if (
            current_text.count("{") > 0
            and current_text.count("{")
            == current_text.count("}")
        ):
            break

    # ==========================
    # FINAL JSON EXTRACTION
    # ==========================
    output_text = model.decode(generated_ids)

    start = output_text.find("{")
    end = output_text.rfind("}")

    if start != -1 and end != -1 and end > start:

        json_text = output_text[start:end + 1]

        try:
            return json.loads(json_text)

        except Exception:
            print("INVALID JSON:", json_text)

    return {
        "name": "",
        "parameters": {}
    }


# ==============================
# VALIDATION
# ==============================
def validate_output(
    output: Dict[str, Any],
    functions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Validate output schema."""

    fn_name = output.get("name", "")
    params = output.get("parameters", {})

    fn = next(
        (
            f for f in functions
            if f["name"] == fn_name
        ),
        None
    )

    if fn is None:

        return {
            "name": "",
            "parameters": {}
        }

    expected = set(fn["parameters"].keys())

    if set(params.keys()) != expected:

        return {
            "name": "",
            "parameters": {}
        }

    return output


# ==============================
# SAVE RESULTS
# ==============================
def save_results(
    path: str,
    results: List[Dict[str, Any]]
) -> None:
    """Save results."""

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as file:

        json.dump(
            results,
            file,
            indent=2
        )


# ==============================
# MAIN
# ==============================
def main() -> None:

    functions = load_json(
        "data/input/functions_definition.json"
    )

    prompts = load_json(
        "data/input/function_calling_tests.json"
    )

    model = Small_LLM_Model()

    results: List[Dict[str, Any]] = []

    for item in prompts:

        prompt_text = item["prompt"]

        full_prompt = build_prompt(
            functions,
            prompt_text
        )

        output = generate(
            model,
            full_prompt,
            functions
        )

        output = validate_output(
            output,
            functions
        )

        formatted_params = {
            k: float(v)
            if isinstance(v, (int, float))
            else v
            for k, v in output.get(
                "parameters",
                {}
            ).items()
        }

        result = {
            "prompt": prompt_text,
            "name": output.get("name", ""),
            "parameters": formatted_params
        }

        print(result)
        print("************")

        results.append(result)

    save_results(
        "data/output/function_calling_results.json",
        results
    )


if __name__ == "__main__":
    main()