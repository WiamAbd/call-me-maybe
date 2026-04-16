from typing import List, Dict, Any
import json

from llm_sdk.llm_sdk import Small_LLM_Model


def load_json(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON file safely.

    Args:
        path (str): Path to the JSON file.

    Returns:
        List[Dict[str, Any]]: Parsed JSON content.

    Raises:
        RuntimeError: If file cannot be read or parsed.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        raise RuntimeError(f"Failed to load JSON file: {path}") from exc


def build_prompt(functions: List[Dict[str, Any]], user_prompt: str) -> str:
    """
    Build the LLM prompt including function definitions.

    Args:
        functions (List[Dict[str, Any]]): List of available functions.
        user_prompt (str): User input prompt.

    Returns:
        str: Full prompt string.
    """
    lines: List[str] = ["Available functions:"]

    for fn in functions:
        params = ", ".join(fn["parameters"].keys())
        lines.append(
            f"- {fn['name']}({params}): {fn['description']}"
        )

    lines.append("\nUser:")
    lines.append(user_prompt)
    lines.append("\nReturn a JSON function call:")

    return "\n".join(lines)


def load_vocab(model: Small_LLM_Model) -> Dict[str, int]:
    """
    Load vocabulary from the LLM tokenizer.

    Args:
        model (Small_LLM_Model): LLM instance.

    Returns:
        Dict[str, int]: Token to ID mapping.
    """
    try:
        path = model.get_path_to_vocab_file()
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        raise RuntimeError("Failed to load vocabulary") from exc


def generate(
    model: Small_LLM_Model,
    prompt: str,
    functions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate a function call using constrained decoding.

    Args:
        model (Small_LLM_Model): LLM model.
        prompt (str): Full prompt string.
        functions (List[Dict[str, Any]]): Available functions.

    Returns:
        Dict[str, Any]: Generated function call.
    """
    try:
        vocab = load_vocab(model)
        id_to_token: Dict[int, str] = {v: k for k, v in vocab.items()}

        input_ids: List[int] = model.encode(prompt)[0].tolist()
        generated_ids: List[int] = []

        structure: List[str] = [
            "{",
            '"name"',
            ":",
            "FUNCTION",
            ",",
            '"parameters"',
            ":",
            "{",
            "}",
            "}",
        ]

        step: int = 0
        selected_function: Dict[str, Any] | None = None

        for _ in range(200):
            logits: List[float] = model.get_logits_from_input_ids(input_ids)

            allowed_tokens: List[str] = []

            # ---------- STRUCTURE ----------
            if structure[step] == "{":
                allowed_tokens = ["{"]

            elif structure[step] == '"name"':
                allowed_tokens = ['"name"']

            elif structure[step] == ":":
                allowed_tokens = [":"]

            elif structure[step] == "FUNCTION":
                allowed_tokens = [
                    f'"{fn["name"]}"' for fn in functions
                ]

            elif structure[step] == ",":
                allowed_tokens = [","]

            elif structure[step] == '"parameters"':
                allowed_tokens = ['"parameters"']

            elif structure[step] == "}":
                allowed_tokens = ["}"]

            # ---------- TOKEN → IDS ----------
            allowed_ids: List[int] = [
                vocab[token]
                for token in allowed_tokens
                if token in vocab
            ]

            if not allowed_ids:
                allowed_ids = list(range(len(logits)))

            # ---------- FILTER LOGITS ----------
            for idx in range(len(logits)):
                if idx not in allowed_ids:
                    logits[idx] = float("-inf")

            next_token_id: int = int(max(
                range(len(logits)), key=lambda i: logits[i]
            ))

            input_ids.append(next_token_id)
            generated_ids.append(next_token_id)

            token: str = id_to_token.get(next_token_id, "")

            # ---------- CAPTURE FUNCTION ----------
            if structure[step] == "FUNCTION":
                fn_name = token.replace('"', "")
                selected_function = next(
                    (fn for fn in functions if fn["name"] == fn_name),
                    None,
                )

            if step < len(structure) - 1:
                step += 1
            else:
                break

        text: str = model.decode(generated_ids)

        try:
            parsed: Dict[str, Any] = json.loads(text)
        except Exception:
            parsed = {"name": "", "parameters": {}}

        return parsed

    except Exception as exc:
        return {"name": "", "parameters": {}, "error": str(exc)}


def save_results(path: str, results: List[Dict[str, Any]]) -> None:
    """
    Save results to a JSON file.

    Args:
        path (str): Output file path.
        results (List[Dict[str, Any]]): Results data.
    """
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
    except Exception as exc:
        raise RuntimeError("Failed to save results") from exc


def main() -> None:
    """
    Main entry point of the program.
    """
    try:
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

            full_prompt = build_prompt(functions, prompt_text)

            output = generate(model, full_prompt, functions)

            results.append({
                "prompt": prompt_text,
                "name": output.get("name", ""),
                "parameters": output.get("parameters", {}),
            })

        save_results(
            "data/output/function_calling_results.json",
            results,
        )

    except Exception as exc:
        print(f"Fatal error: {exc}")


if __name__ == "__main__":
    main()